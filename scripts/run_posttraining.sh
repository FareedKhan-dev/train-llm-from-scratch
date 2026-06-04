#!/usr/bin/env bash
# Turnkey post-training pipeline: SFT -> Reward Model -> {DPO, PPO} -> GRPO -> eval table.
# Assumes the base model is already pretrained (scripts/pretrain_base.py ->
# /ephemeral/ckpts/base_pretrained.pt) and the datasets are prepared (scripts/prepare_*).
#
# Usage (from repo root):
#   bash scripts/run_posttraining.sh            # use both GPUs (torchrun)
#   NPROC=1 bash scripts/run_posttraining.sh    # single GPU
#
# Each stage writes a checkpoint to /ephemeral/ckpts and metrics JSONL to /ephemeral/logs.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=. HF_HOME=/ephemeral/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/venv/bin/python
NPROC=${NPROC:-2}

run() {  # run a training script single- or multi-GPU
  if [ "$NPROC" -gt 1 ]; then
    /ephemeral/venv/bin/torchrun --standalone --nproc_per_node="$NPROC" "$@"
  else
    $PY "$@"
  fi
}

echo "############ 1/5  SFT ############"
run scripts/train_sft.py

echo "############ 2/5  Reward Model ############"
run scripts/train_reward.py

echo "############ 3/5  DPO ############"
run scripts/train_dpo.py --loss_type dpo

echo "############ 4/5  PPO (GSM8K, verifier reward) ############"
run scripts/train_ppo.py --reward_source verifier

echo "############ 5/5  GRPO (arithmetic curriculum -> GSM8K) ############"
run scripts/train_grpo.py

echo "############ Eval: GSM8K accuracy across stages ############"
TABLE=/ephemeral/logs/stage_table.jsonl
rm -f "$TABLE"
for s in base_pretrained sft dpo ppo grpo; do
  [ -f "/ephemeral/ckpts/$s.pt" ] && \
    $PY scripts/eval_post_training.py --ckpt "/ephemeral/ckpts/$s.pt" --label "$s" --limit 200 --append "$TABLE"
done
$PY scripts/eval_post_training.py --table "$TABLE"
echo "Done. Metrics in /ephemeral/logs, checkpoints in /ephemeral/ckpts."

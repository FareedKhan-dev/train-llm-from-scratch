#!/usr/bin/env bash
# Turnkey post-training pipeline: SFT -> Reward Model -> {DPO, PPO} -> GRPO -> eval table.
# Assumes the base model is already pretrained (scripts/pretrain_base.py ->
# models/base_pretrained.pt) and the datasets are prepared (scripts/prepare_*).
#
# Usage (from repo root):
#   bash scripts/run_posttraining.sh                         # use both GPUs (torchrun)
#   NPROC=1 bash scripts/run_posttraining.sh                 # single GPU
#   PYTHON=.venv/bin/python bash scripts/run_posttraining.sh # pick the interpreter
#
# Each stage writes a checkpoint to models/ and metrics JSONL to logs/ (paths come from
# configs/*.json, so edit those to use another disk).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY=${PYTHON:-python}
NPROC=${NPROC:-2}
CKPTS=${CKPTS:-models}

run() {  # run a training script single- or multi-GPU
  if [ "$NPROC" -gt 1 ]; then
    $PY -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$@"
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
TABLE=logs/stage_table.jsonl
mkdir -p logs
rm -f "$TABLE"
for s in base_pretrained sft dpo ppo grpo; do
  if [ -f "$CKPTS/$s.pt" ]; then
    $PY scripts/eval_post_training.py --ckpt "$CKPTS/$s.pt" --label "$s" --limit 200 --append "$TABLE"
  fi
done
$PY scripts/eval_post_training.py --table "$TABLE"
echo "Done. Metrics in logs/, checkpoints in $CKPTS/."

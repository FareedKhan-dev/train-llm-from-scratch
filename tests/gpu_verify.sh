#!/usr/bin/env bash
# End-to-end GPU verification: run EVERY post-training script for a few steps on the GPU,
# on the REAL prepared datasets, with a tiny model (fits in the spare VRAM next to the
# running pretraining). Proves the full pipeline works on GPU + real data, not just CPU.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. HF_HOME=/ephemeral/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=/ephemeral/venv/bin/python
DIM="--vocab_size 50304 --context_length 1024 --n_embed 128 --n_head 4 --n_blocks 2 --device cuda --amp_dtype bf16"
CK=/ephemeral/ckpts/tiny_base.pt
ok() { echo "  [PASS] $1"; }; bad() { echo "  [FAIL] $1"; }

echo "== 1/6 SFT on real sft_packed.h5 (GPU) =="
$PY scripts/train_sft.py $DIM --pretrained_ckpt $CK --data_path /ephemeral/data/sft_packed.h5 \
  --out_ckpt /ephemeral/ckpts/gpuverify_sft.pt --batch_size 8 --epochs 1 --max_steps 12 \
  --eval_steps 1000 --warmup_steps 2 --save_every 1000 2>&1 | grep -E "step [0-9]+/|Done SFT" | tail -2 \
  && [ -f /ephemeral/ckpts/gpuverify_sft.pt ] && ok "SFT ran on GPU + saved" || bad "SFT"

echo "== 2/6 Reward Model on real preferences.jsonl (GPU) =="
$PY scripts/train_reward.py $DIM --sft_ckpt $CK --pref_path /ephemeral/data/prefs_small.jsonl \
  --out_ckpt /ephemeral/ckpts/gpuverify_reward.pt --batch_size 8 --epochs 1 --eval_steps 1000 \
  --warmup_steps 2 --max_len 512 --save_every 1000 2>&1 | grep -E "step [0-9]+/|Done RM" | tail -2 \
  && [ -f /ephemeral/ckpts/gpuverify_reward.pt ] && ok "Reward ran on GPU + saved" || bad "Reward"
# truncate to a few steps by capping rows via a tiny pref file
echo "== 3/6 DPO on real preferences.jsonl (GPU) =="
$PY scripts/train_dpo.py $DIM --sft_ckpt $CK --pref_path /ephemeral/data/prefs_small.jsonl \
  --out_ckpt /ephemeral/ckpts/gpuverify_dpo.pt --loss_type dpo --beta 0.1 --batch_size 8 --epochs 1 \
  --eval_steps 1000 --warmup_steps 2 --max_len 512 --save_every 1000 2>&1 | grep -E "step [0-9]+/|Done DPO" | tail -2 \
  && [ -f /ephemeral/ckpts/gpuverify_dpo.pt ] && ok "DPO ran on GPU + saved" || bad "DPO"

echo "== 4/6 PPO on real GSM8K prompts, verifier reward (GPU) =="
$PY scripts/train_ppo.py $DIM --sft_ckpt $CK --prompt_path /ephemeral/data/rl_prompts_train.jsonl \
  --reward_source verifier --iterations 2 --prompts_per_iter 4 --rollout_len 48 --ppo_epochs 2 \
  --minibatch_size 2 --eval_every 1000 --save_every 1000 --out_ckpt /ephemeral/ckpts/gpuverify_ppo.pt \
  2>&1 | grep -E "iter [0-9]+|Done PPO" | tail -2 \
  && [ -f /ephemeral/ckpts/gpuverify_ppo.pt ] && ok "PPO ran on GPU + saved" || bad "PPO"

echo "== 5/6 GRPO on real arithmetic+GSM8K prompts (GPU) =="
$PY scripts/train_grpo.py $DIM --sft_ckpt $CK --curriculum_path /ephemeral/data/arithmetic_prompts.jsonl \
  --prompt_path /ephemeral/data/rl_prompts_train.jsonl --curriculum_iters 1 --iterations 2 \
  --prompts_per_iter 2 --group_size 4 --rollout_len 48 --grpo_epochs 1 --eval_every 1000 --save_every 1000 \
  --out_ckpt /ephemeral/ckpts/gpuverify_grpo.pt 2>&1 | grep -E "iter [0-9]+|Done GRPO" | tail -2 \
  && [ -f /ephemeral/ckpts/gpuverify_grpo.pt ] && ok "GRPO ran on GPU + saved" || bad "GRPO"

echo "== 6/6 GSM8K eval benchmark on GPU =="
$PY scripts/eval_post_training.py --ckpt $CK --label tiny_base --limit 20 --max_new_tokens 48 --device cuda --samples 0 \
  2>&1 | grep -E "GSM8K test accuracy" | tail -1 \
  && ok "GSM8K eval ran on GPU" || bad "eval"

echo "== cleanup =="; rm -f /ephemeral/ckpts/gpuverify_*.pt; echo "done"

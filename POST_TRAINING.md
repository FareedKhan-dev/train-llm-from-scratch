# Post-Training from Scratch: SFT · Reward Model · PPO · DPO · GRPO

This adds a complete, **from-scratch** (pure PyTorch — no `trl`/`peft`/`transformers`)
post-training suite on top of the repo's own `Transformer`, covering the full modern
pipeline used to turn a base LM into an aligned, reasoning model:

```
Base (pretrained)  ──►  SFT  ──►  Reward Model ──►  PPO ┐
                          │                            ├─►  GRPO / RLVR  (GSM8K)
                          └────────►  DPO / ORPO / KTO ─┘
```

Everything is self-contained on the repo's custom model and trained/evaluated on **real
public datasets** (Alpaca, Dolly, Anthropic HH-RLHF, UltraFeedback, GSM8K). The headline
metric is **greedy GSM8K accuracy across stages**.

> **Expectation:** a ~400M model pretrained from scratch on 2×H100 is coherent and
> instruction-followable and shows real before/after gains at each stage, but its
> *absolute* GSM8K score stays modest — frontier numbers need far more pretraining
> compute. The value here is the authentic end-to-end pipeline on real data with real eval.

---

## 0. Environment

H100s need CUDA-12 wheels (the original `requirements.txt` pins cu118 for the legacy
pretraining path and is left untouched). Use a separate venv + `requirements-post.txt`,
and keep all large artifacts on the big `/ephemeral` disk.

```bash
python3 -m venv /ephemeral/venv && source /ephemeral/venv/bin/activate
pip install -r requirements-post.txt        # torch cu121 + datasets/wandb/tiktoken/h5py
export HF_HOME=/ephemeral/hf_cache
```

Run everything from the repo root with `PYTHONPATH=.`. Single GPU: `python scripts/X.py`.
Both GPUs: `torchrun --standalone --nproc_per_node=2 scripts/X.py` (DDP + bf16, one code path).

Config lives in [config/post_training_config.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/config/post_training_config.py) as
per-stage dataclasses; **any field is a CLI override**, e.g. `--lr 2e-5 --batch_size 16`.
The default base model is ~400M (`n_embed=1024, n_head=16, n_blocks=24, context_length=1024`).

---

## 1. Pretrain the base (the long pole)

```bash
# one-time data prep (Pile -> flat-token HDF5 on /ephemeral)
PYTHONPATH=. python scripts/prepare_pretrain_data.py --split val   --out /ephemeral/data/pile_dev.h5
PYTHONPATH=. python scripts/prepare_pretrain_data.py --split train --num_shards 1 --out /ephemeral/data/pile_train.h5

# pretrain (multi-day for full quality; checkpoints continuously to /ephemeral/ckpts/base_pretrained.pt)
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/pretrain_base.py
```

[scripts/pretrain_base.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/scripts/pretrain_base.py) upgrades the original
`train_transformer.py` recipe with DDP, bf16 autocast, gradient accumulation, a cosine LR
schedule with warmup, and periodic checkpointing — everything needed to train a mid-size
model on 2×H100. The original training script is untouched.

---

## 2. SFT (instruction tuning)

```bash
PYTHONPATH=. python scripts/prepare_sft_data.py --context_length 1024      # Alpaca+Dolly+GSM8K -> packed HDF5
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_sft.py # -> /ephemeral/ckpts/sft.pt
```

- **From-scratch loss:** prompt-masked next-token CE in
  [src/post_training/sft.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/sft.py) (`sft_loss`). Only assistant tokens
  are trained (mask from the chat template); sequence **packing** fills each row to the
  context length.
- **Chat format:** [src/post_training/chat_template.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/chat_template.py).
  The r50k_base tokenizer has only `<|endoftext|>` as a special token, so role markers
  (`<|user|>`, `<|assistant|>`, `<think>`, `<answer>`) are ordinary tokens the model learns.
  GSM8K is reformatted into `<think>…</think><answer>N</answer>` so the model learns the
  exact output structure the RL verifier rewards.
- **Eval:** masked dev perplexity + greedy GSM8K dev accuracy.

## 3. Reward Model

```bash
PYTHONPATH=. python scripts/prepare_preference_data.py --source both   # HH-RLHF + UltraFeedback -> JSONL
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_reward.py  # -> reward.pt
```

- **From-scratch:** a scalar reward head on the SFT backbone
  ([src/post_training/reward_model.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/reward_model.py)), trained with the
  **Bradley-Terry** pairwise loss in
  [src/post_training/reward_train.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/reward_train.py). The reward is read
  off the last real token (causal attention makes right-padding safe — no attention mask
  needed). **Eval:** held-out preference accuracy (expect ~0.65–0.75 on noisy real data).

## 4. DPO / ORPO / KTO

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_dpo.py --loss_type dpo --beta 0.1
#   --loss_type orpo   (reference-free, folds SFT+alignment into one stage)
#   --loss_type kto    (unpaired, reference-KL baseline)
```

- **From-scratch:** all three objectives in [src/post_training/dpo.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/dpo.py).
  Policy initialized from SFT; a frozen deep copy is the reference (ORPO needs none).
  Operates on summed response log-probs from
  [src/post_training/rollout.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/rollout.py) (`sequence_logprobs`).
  **Eval:** implicit-reward accuracy/margins + GSM8K dev.

## 5. PPO (classic RLHF)

```bash
PYTHONPATH=. python scripts/prepare_rl_prompts.py                 # GSM8K + arithmetic warm-up -> JSONL
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_ppo.py --reward_source verifier
#   --reward_source rm   to use the trained reward model instead of the GSM8K checker
```

- **From-scratch:** GAE, clipped policy/value losses in
  [src/post_training/ppo.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/ppo.py); the actor-critic shares the backbone
  via [src/post_training/value_head.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/value_head.py). Each iteration:
  roll out → score (verifier or RM) → add per-token **KL-to-reference** penalty → GAE →
  several clipped-surrogate epochs. **Eval:** reward / KL / clip-fraction / value-loss curves
  and greedy GSM8K test accuracy.

## 6. GRPO / RLVR (the 2025 frontier; DeepSeek-R1 style)

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_grpo.py --group_size 8
```

- **From-scratch:** group-relative advantages + token-level clipped surrogate with a k3 KL
  penalty in [src/post_training/grpo.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/grpo.py). **No critic** — the
  baseline is each prompt's own group of G samples. An **arithmetic curriculum** runs for
  the first `curriculum_iters` iterations so the policy has non-zero reward variance before
  full GSM8K. **Eval:** mean group reward, informative-group fraction, KL, GSM8K test accuracy.

---

## 7. Inference / chat (any stage checkpoint)

[scripts/chat.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/scripts/chat.py) loads **any** checkpoint (base/sft/dpo/ppo/grpo) —
reading the model dims from the checkpoint itself — and generates with the chat template
(instruction models) or as raw continuation (the base model):

```bash
# instruction-tuned models (chat template applied automatically)
PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/sft.pt  --prompt "What is 13 + 29?"
PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/grpo.pt --prompt "..." --greedy
# base model continuation
PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/base_pretrained.pt --raw --prompt "Once upon a time"
# interactive REPL (omit --prompt); sampling via --temperature/--top_p/--top_k or --greedy
PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/sft.pt
```

Generation reuses the same tested core as training/eval
([src/post_training/rollout.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/rollout.py),
[src/post_training/inference.py](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/inference.py)).

## 8. The across-stages results table

```bash
for s in base_pretrained sft dpo ppo grpo; do
  PYTHONPATH=. python scripts/eval_post_training.py --ckpt /ephemeral/ckpts/$s.pt \
    --label $s --limit 200 --append /ephemeral/logs/stage_table.jsonl
done
PYTHONPATH=. python scripts/eval_post_training.py --table /ephemeral/logs/stage_table.jsonl
```

Every trainer also writes a JSONL metrics file under `/ephemeral/logs/` (plottable without
any external service); pass `--use_wandb true` to also mirror to Weights & Biases.

---

## Design notes (why it's built this way)

- **Wrap, don't rewrite.** The educational `Transformer`/`Block`/`Head`/`MLP` are unchanged
  except one additive method, `forward_hidden` (returns post-final-LN hidden states the
  heads consume). Value head, reward head, and all RL log-prob math compose around it.
- **Causal attention ⇒ right-padding is safe.** The last real token never attends to
  padding after it, so RM (last-token reward) and DPO (masked response) need no attention
  mask; the response mask zeros padded positions in the loss.
- **fp32 log-probs.** PPO/GRPO/DPO subtract log-probs, so they're always computed in fp32
  even under bf16 autocast.
- **Context cap.** Learned absolute positions cap any sequence at `context_length`; rollouts
  enforce `prompt + generation ≤ context_length`.
- **Reward hacking / KL control.** Verifier rewards are correctness-dominant with a small,
  bounded format bonus; a KL-to-reference penalty anchors RL to the SFT policy.

## Tests

```bash
PYTHONPATH=. python tests/test_post_training_smoke.py   # core math: log-probs, heads, parsing, masking
```

Each trainer also runs end-to-end on a tiny model in seconds (see the smoke commands used
during development), and the PPO/GRPO math has standalone unit checks (GAE, clipped losses,
group advantages, k3 KL).

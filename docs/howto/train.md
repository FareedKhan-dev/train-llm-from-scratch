# Train (UI & CLI)

You can run every stage two ways: from the **command line** (full control, best for long jobs) or from
the **Streamlit control panel** (forms, one-click launch, live logs, charts — see [The UI](ui.md)).

## Install

```bash
pip install -e ".[train]"     # editable install — no more PYTHONPATH=.
export HF_HOME=/ephemeral/hf_cache
```

## The pipeline, end to end (CLI)

Each command reads its stage JSON from `configs/` (override anything with `--field`, or point at another
file with `--config`). Use `python` for one GPU and `torchrun` for many.

=== "1. Data"

    ```bash
    python scripts/prepare_pretrain_data.py --split val   --out /ephemeral/data/pile_dev.h5
    python scripts/prepare_pretrain_data.py --split train --num_shards 1 --out /ephemeral/data/pile_train.h5
    python scripts/prepare_sft_data.py
    python scripts/prepare_preference_data.py --source both
    python scripts/prepare_rl_prompts.py
    ```

=== "2. Pretrain"

    ```bash
    # single GPU
    python scripts/pretrain_base.py --config configs/pretrain.json
    # both GPUs (effective batch = batch_size * grad_accum * num_gpus)
    torchrun --standalone --nproc_per_node=2 scripts/pretrain_base.py --config configs/pretrain.json
    ```

=== "3. Align"

    ```bash
    torchrun --standalone --nproc_per_node=2 scripts/train_sft.py
    torchrun --standalone --nproc_per_node=2 scripts/train_reward.py
    torchrun --standalone --nproc_per_node=2 scripts/train_dpo.py --loss_type dpo
    torchrun --standalone --nproc_per_node=2 scripts/train_ppo.py --reward_source verifier
    torchrun --standalone --nproc_per_node=2 scripts/train_grpo.py
    ```

The whole alignment chain in one shot:

```bash
bash scripts/run_posttraining.sh        # SFT → RM → DPO → PPO → GRPO → eval table
```

## Multi-GPU notes

- `torchrun --standalone --nproc_per_node=N` launches N data-parallel ranks (DDP + bf16). Only rank 0
  logs and checkpoints.
- On the dev box (2× H100, no NVLink) the educational attention materializes a `(B, n_head, T, T)` tensor
  per block, so memory scales with sequence length — at context 1024 use `--batch_size 8 --grad_accum 12`
  and recover the effective batch via accumulation. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Where outputs go

- Checkpoints → `/ephemeral/ckpts/<stage>.pt` (each carries its own resolved `cfg`).
- Metrics → `/ephemeral/logs/<stage>_<timestamp>.jsonl` (one JSON per logged step). The UI plots these
  live; you can also `--use_wandb true` to mirror to Weights & Biases.

Then [evaluate](../08_evaluation.md) on GSM8K and [chat](../09_inference.md) with any checkpoint.

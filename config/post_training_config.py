"""
Configuration for the from-scratch post-training suite.

Kept entirely separate from ``config/config.py`` (which import-executes for the original
pretraining path and must stay untouched). Each stage is a frozen-ish dataclass that
inherits the shared :class:`BaseModelConfig` model/runtime fields and adds its own
hyperparameters. Construct with overrides, e.g. ``SFTConfig(lr=2e-5, batch_size=16)``.

The default base model is ~400M parameters (n_embed=1024, n_head=16, n_blocks=24,
context_length=1024) -- the "mid" size chosen so real datasets (Alpaca, HH-RLHF, GSM8K)
give meaningful results while still fitting comfortably on one H100 and training in a
reasonable time on 2x H100. A tiny ``SMOKE`` variant is provided for fast CPU/1-GPU tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


# Shared paths (all heavy artifacts live on the 1.5TB /ephemeral disk).
EPHEMERAL = "/ephemeral"
CKPT_DIR = f"{EPHEMERAL}/ckpts"
DATA_DIR = f"{EPHEMERAL}/data"
LOG_DIR = f"{EPHEMERAL}/logs"


@dataclass
class BaseModelConfig:
    # --- model architecture (must match across all stages + the pretrained ckpt) ---
    vocab_size: int = 50304
    context_length: int = 1024
    n_embed: int = 1024
    n_head: int = 16
    n_blocks: int = 24

    # --- runtime ---
    device: str = "cuda"
    amp_dtype: str | None = "bf16"      # None | "bf16"; bf16 needs no GradScaler on H100
    seed: int = 1337
    compile: bool = False               # torch.compile the model (big speedup, slow 1st step)
    ckpt_dir: str = CKPT_DIR
    log_dir: str = LOG_DIR
    use_wandb: bool = False
    wandb_project: str = "train-llm-from-scratch-posttrain"


@dataclass
class PretrainConfig(BaseModelConfig):
    """Pretrain the mid base model from scratch on the Pile HDF5 (mix in task text late)."""
    train_path: str = "data/train/pile_train.h5"
    dev_path: str = "data/val/pile_dev.h5"
    batch_size: int = 24                # per-GPU micro-batch
    grad_accum: int = 8                 # effective batch = batch_size * grad_accum * world
    train_steps: int = 200_000
    eval_steps: int = 1_000
    eval_iters: int = 100
    warmup_steps: int = 2_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    out_ckpt: str = f"{CKPT_DIR}/base_pretrained.pt"
    save_every: int = 2_000


@dataclass
class SFTConfig(BaseModelConfig):
    pretrained_ckpt: str = f"{CKPT_DIR}/base_pretrained.pt"
    data_path: str = f"{DATA_DIR}/sft_packed.h5"
    out_ckpt: str = f"{CKPT_DIR}/sft.pt"
    batch_size: int = 16
    grad_accum: int = 2
    epochs: int = 3
    max_steps: int = -1                 # -1 = run full epochs
    eval_steps: int = 200
    warmup_steps: int = 100
    lr: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    save_every: int = 500


@dataclass
class RewardConfig(BaseModelConfig):
    sft_ckpt: str = f"{CKPT_DIR}/sft.pt"
    pref_path: str = f"{DATA_DIR}/preferences.jsonl"
    out_ckpt: str = f"{CKPT_DIR}/reward.pt"
    batch_size: int = 8                 # pairs per step (2x sequences through the model)
    epochs: int = 1
    eval_steps: int = 200
    warmup_steps: int = 50
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_len: int = 768
    save_every: int = 500


@dataclass
class DPOConfig(BaseModelConfig):
    sft_ckpt: str = f"{CKPT_DIR}/sft.pt"        # init policy + frozen reference
    pref_path: str = f"{DATA_DIR}/preferences.jsonl"
    out_ckpt: str = f"{CKPT_DIR}/dpo.pt"
    loss_type: str = "dpo"              # "dpo" | "orpo" | "kto"
    beta: float = 0.1
    orpo_lambda: float = 1.0           # ORPO odds-ratio weight (loss_type="orpo")
    batch_size: int = 8
    epochs: int = 1
    eval_steps: int = 200
    warmup_steps: int = 50
    lr: float = 5e-7
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_len: int = 768
    save_every: int = 500


@dataclass
class PPOConfig(BaseModelConfig):
    sft_ckpt: str = f"{CKPT_DIR}/sft.pt"
    reward_ckpt: str = f"{CKPT_DIR}/reward.pt"   # used when reward_source="rm"
    prompt_path: str = f"{DATA_DIR}/rl_prompts_train.jsonl"
    eval_prompt_path: str = f"{DATA_DIR}/rl_prompts_test.jsonl"
    out_ckpt: str = f"{CKPT_DIR}/ppo.pt"
    reward_source: str = "verifier"    # "verifier" (GSM8K checker) | "rm" (reward model)
    iterations: int = 1_000
    prompts_per_iter: int = 32         # prompts sampled per PPO iteration (per rank)
    rollout_len: int = 300
    temperature: float = 1.0
    top_p: float = 1.0
    ppo_epochs: int = 4
    minibatch_size: int = 16
    clip: float = 0.2
    vf_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    kl_coef: float = 0.05              # penalty on KL(policy || ref) added to reward
    lr: float = 1e-6
    grad_clip: float = 1.0
    eval_every: int = 50
    save_every: int = 100


@dataclass
class GRPOConfig(BaseModelConfig):
    sft_ckpt: str = f"{CKPT_DIR}/sft.pt"
    prompt_path: str = f"{DATA_DIR}/rl_prompts_train.jsonl"
    eval_prompt_path: str = f"{DATA_DIR}/rl_prompts_test.jsonl"
    curriculum_path: str = f"{DATA_DIR}/arithmetic_prompts.jsonl"  # warm-up before GSM8K
    curriculum_iters: int = 100        # iterations on the arithmetic warm-up before GSM8K
    out_ckpt: str = f"{CKPT_DIR}/grpo.pt"
    iterations: int = 1_000
    prompts_per_iter: int = 8          # distinct prompts per iter (per rank)
    group_size: int = 8               # samples per prompt (group)
    rollout_len: int = 300
    temperature: float = 1.0
    top_p: float = 1.0
    grpo_epochs: int = 1
    clip: float = 0.2
    kl_coef: float = 0.04             # KL(policy || ref) penalty term in the loss
    lr: float = 1e-6
    grad_clip: float = 1.0
    eval_every: int = 50
    save_every: int = 100


# Tiny config for fast smoke tests (CPU or a single GPU, seconds not hours).
SMOKE = dict(
    vocab_size=256, context_length=64, n_embed=64, n_head=4, n_blocks=2, device="cpu", amp_dtype=None
)


def smoke(cfg_cls):
    """Return an instance of ``cfg_cls`` shrunk to the tiny SMOKE model dims."""
    return replace(cfg_cls(), **SMOKE)

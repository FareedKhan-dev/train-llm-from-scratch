"""
Configuration for the post-training suite (and the scalable pretraining script).

Kept separate from ``config/config.py`` (the plain constants of the original pretraining
path). Each stage is a dataclass that inherits the shared :class:`BaseModelConfig`
model/runtime fields and adds its own hyperparameters. Construct with overrides, e.g.
``SFTConfig(lr=2e-5, batch_size=16)``, or load from JSON with :func:`config.loader.load_config`.

The fields are typed (``Literal`` for every choice), and the JSON loader and the CLI check
values against these types, so a typo such as ``"loss_type": "dop"`` fails at startup with a
clear message instead of halfway through a run. ``__post_init__`` adds the checks a type
cannot express (positive sizes, ``n_embed`` divisible by ``n_head``, and so on).

The default base model is ~400M parameters (n_embed=1024, n_head=16, n_blocks=24,
context_length=1024), the "mid" size chosen so real datasets (Alpaca, HH-RLHF, GSM8K) give
meaningful results while still fitting on one 80GB GPU. A tiny ``SMOKE`` variant is provided
for fast CPU tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypeVar

from config.paths import CKPT_DIR, DATA_DIR, LOG_DIR

Arch = Literal["classic", "modern"]
AttentionKind = Literal["gqa", "mla"]
AmpDtype = Literal["bf16", "fp16"]
OptimizerName = Literal["adamw", "muon"]
Schedule = Literal["cosine", "wsd", "linear"]
PreferenceLoss = Literal["dpo", "orpo", "kto"]
RewardSource = Literal["verifier", "rm"]


class ConfigError(ValueError):
    """A configuration value is missing, has the wrong type, or is out of range."""


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


@dataclass
class BaseModelConfig:
    # --- model architecture (must match across all stages + the pretrained ckpt) ---
    arch: Arch = "classic"  # "classic" = the original Transformer, "modern" = Llama-style
    vocab_size: int = 50304
    context_length: int = 1024
    n_embed: int = 1024
    n_head: int = 16
    n_blocks: int = 24

    # --- modern architecture only (ignored when arch="classic"), see src/models/modern ---
    n_kv_head: int | None = None  # grouped-query attention; None = one KV head per query head
    attention: AttentionKind = "gqa"  # "mla" = DeepSeek multi-head latent attention
    kv_latent_dim: int | None = None  # MLA latent size; None = n_embed // 4
    rope_theta: float = 10_000.0
    qk_norm: bool = True
    attn_gate: bool = False
    sliding_window: int | None = None
    tie_embeddings: bool = True
    n_experts: int = 0  # > 0 = Mixture of Experts
    moe_top_k: int = 2
    n_shared_experts: int = 0

    # --- runtime ---
    device: str = "auto"  # auto | cuda | mps | cpu
    amp_dtype: AmpDtype | None = "bf16"  # None | "bf16" | "fp16"; ignored on CPU
    seed: int = 1337
    compile: bool = False  # torch.compile the model (big speedup, slow 1st step)
    ckpt_dir: str = CKPT_DIR
    log_dir: str = LOG_DIR
    use_wandb: bool = False
    wandb_project: str = "train-llm-from-scratch-posttrain"

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "n_embed", "n_head", "n_blocks"):
            _check(getattr(self, name) > 0, f"{name} must be positive, got {getattr(self, name)}")
        _check(self.n_embed % self.n_head == 0,
               f"n_embed ({self.n_embed}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_head is not None:
            _check(self.n_kv_head > 0 and self.n_head % self.n_kv_head == 0,
                   f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        _check(self.n_experts >= 0, "n_experts must be >= 0")
        if self.n_experts:
            _check(1 <= self.moe_top_k <= self.n_experts, "moe_top_k must be between 1 and n_experts")


@dataclass
class PretrainConfig(BaseModelConfig):
    """Pretrain the mid base model from scratch on the Pile HDF5 (mix in task text late)."""
    train_path: str = f"{DATA_DIR}/pile_train.h5"
    dev_path: str = f"{DATA_DIR}/pile_dev.h5"
    batch_size: int = 24                # per-GPU micro-batch
    grad_accum: int = 8                 # effective batch = batch_size * grad_accum * world
    train_steps: int = 200_000
    eval_steps: int = 1_000
    eval_iters: int = 100
    warmup_steps: int = 2_000
    optimizer: OptimizerName = "adamw"  # "muon" = Muon for weight matrices, AdamW for the rest
    lr_schedule: Schedule = "cosine"    # cosine | wsd (warmup-stable-decay) | linear (to min_lr)
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    out_ckpt: str = f"{CKPT_DIR}/base_pretrained.pt"
    save_every: int = 2_000

    def __post_init__(self) -> None:
        super().__post_init__()
        _check(self.batch_size > 0 and self.grad_accum > 0, "batch_size and grad_accum must be positive")
        _check(0 <= self.min_lr <= self.lr, "need 0 <= min_lr <= lr")
        _check(self.warmup_steps < self.train_steps, "warmup_steps must be smaller than train_steps")


@dataclass
class SFTConfig(BaseModelConfig):
    pretrained_ckpt: str = f"{CKPT_DIR}/base_pretrained.pt"
    data_path: str = f"{DATA_DIR}/sft_packed.h5"
    dev_path: str = f"{DATA_DIR}/sft_dev_packed.h5"
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
    test_path: str = f"{DATA_DIR}/preferences_test.jsonl"
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

    def __post_init__(self) -> None:
        super().__post_init__()
        _check(16 <= self.max_len <= self.context_length, "max_len must be between 16 and context_length")


@dataclass
class DPOConfig(BaseModelConfig):
    sft_ckpt: str = f"{CKPT_DIR}/sft.pt"        # init policy + frozen reference
    pref_path: str = f"{DATA_DIR}/preferences.jsonl"
    test_path: str = f"{DATA_DIR}/preferences_test.jsonl"
    out_ckpt: str = f"{CKPT_DIR}/dpo.pt"
    loss_type: PreferenceLoss = "dpo"   # dpo | orpo | kto
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

    def __post_init__(self) -> None:
        super().__post_init__()
        _check(self.beta > 0, "beta must be positive")
        _check(16 <= self.max_len <= self.context_length, "max_len must be between 16 and context_length")


@dataclass
class PPOConfig(BaseModelConfig):
    sft_ckpt: str = f"{CKPT_DIR}/sft.pt"
    reward_ckpt: str = f"{CKPT_DIR}/reward.pt"   # used when reward_source="rm"
    prompt_path: str = f"{DATA_DIR}/rl_prompts_train.jsonl"
    eval_prompt_path: str = f"{DATA_DIR}/rl_prompts_test.jsonl"
    out_ckpt: str = f"{CKPT_DIR}/ppo.pt"
    reward_source: RewardSource = "verifier"  # "verifier" (GSM8K checker) | "rm" (reward model)
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

    def __post_init__(self) -> None:
        super().__post_init__()
        _check(self.group_size >= 2, "group_size must be at least 2 (the group is the baseline)")
        _check(self.clip > 0, "clip must be positive")


# Tiny config for fast smoke tests (CPU or a single GPU, seconds not hours).
SMOKE = dict(
    vocab_size=256, context_length=64, n_embed=64, n_head=4, n_blocks=2, device="cpu", amp_dtype=None
)

C = TypeVar("C", bound=BaseModelConfig)


def smoke(cfg_cls: type[C]) -> C:
    """Return an instance of ``cfg_cls`` shrunk to the tiny SMOKE model dims."""
    overrides: dict[str, object] = dict(SMOKE)
    if "max_len" in cfg_cls.__dataclass_fields__:
        overrides["max_len"] = SMOKE["context_length"]
    return replace(cfg_cls(), **overrides)  # type: ignore[arg-type]

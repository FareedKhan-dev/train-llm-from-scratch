"""
Shared helpers for the post-training stack: model construction from configs,
frozen reference/old-policy copies, checkpoint I/O (keeping the repo's existing
checkpoint shape), masked reductions, and seeding.
"""

from __future__ import annotations

import contextlib
import copy
import random
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.checkpoint import (
    atomic_save,
    load_checkpoint,
    load_model_weights,
    model_state_from_checkpoint,
    strip_wrapper_prefixes,
    unwrap_model,
)
from src.models.transformer import Transformer


def amp_autocast(amp_dtype: str | None, device: str):
    """Return a bf16 autocast context on CUDA when requested, else a no-op context.

    bf16 needs no GradScaler (unlike fp16), which keeps the training loops clean.
    """
    if amp_dtype == "bf16" and str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --- Reproducibility ---------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (incl. CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Model construction ------------------------------------------------------

def _cfg_get(cfg: Any, key: str) -> Any:
    """Read a field from either a dataclass/object (attr) or a dict."""
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def build_model_from_config(cfg: Any) -> Transformer:
    """
    Construct a fresh :class:`Transformer` from a config carrying the standard keys
    ``n_head, n_embed, context_length, vocab_size, n_blocks``. Works with the new
    post-training dataclasses and with the legacy ``default_config`` dict.
    """
    return Transformer(
        n_head=_cfg_get(cfg, "n_head"),
        n_embed=_cfg_get(cfg, "n_embed"),
        context_length=_cfg_get(cfg, "context_length"),
        vocab_size=_cfg_get(cfg, "vocab_size"),
        N_BLOCKS=_cfg_get(cfg, "n_blocks"),
    )


# Kept for code that imported the old private name; it now also strips torch.compile's
# ``_orig_mod.`` prefix (issue #36).
_strip_ddp_prefix = strip_wrapper_prefixes


def load_backbone_from_ckpt(cfg: Any, ckpt_path: str, device: str) -> Transformer:
    """
    Build a Transformer from ``cfg`` and load backbone weights from a checkpoint saved
    by the pretraining script or any post-training stage (``model_state_dict`` key).

    DDP (``module.``) and torch.compile (``_orig_mod.``) prefixes are stripped. Auxiliary
    head weights (value/reward), if present, are ignored here because wrappers add their own
    fresh heads. A checkpoint that does not cover every backbone parameter raises an error
    instead of silently leaving random weights in place.
    """
    model = build_model_from_config(cfg)
    state = model_state_from_checkpoint(load_checkpoint(ckpt_path, map_location="cpu"))
    load_model_weights(model, state, source=ckpt_path)
    return model.to(device)


def unwrap(model: nn.Module) -> nn.Module:
    """Return the plain model behind DDP and torch.compile wrappers (or the model itself)."""
    return unwrap_model(model)


def make_frozen_copy(model: nn.Module, device: str | None = None) -> nn.Module:
    """
    Deep-copy a model, put it in eval mode, and disable all gradients. Used for the
    DPO/PPO/GRPO reference model and the PPO old-policy snapshot. At the ~300M-1B
    scale this is cheap on an 80GB H100 and is the clearest possible implementation.
    """
    ref = copy.deepcopy(unwrap(model))
    if device is not None:
        ref = ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# --- Masked reductions -------------------------------------------------------

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over positions where ``mask`` is truthy (safe if mask empty)."""
    mask = mask.to(values.dtype)
    total = (values * mask).sum()
    count = mask.sum().clamp(min=1.0)
    return total / count


def masked_mean_per_row(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row (B,) mean of ``values`` (B,T) over masked positions."""
    mask = mask.to(values.dtype)
    total = (values * mask).sum(dim=-1)
    count = mask.sum(dim=-1).clamp(min=1.0)
    return total / count


def gather_last(values: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    """
    Given per-token ``values`` (B, T) and ``seq_lengths`` (B,), return the value at the
    last real token of each row, i.e. ``values[i, seq_lengths[i]-1]``. This is the
    InstructGPT convention for reading a scalar reward off a sequence model.
    """
    idx = (seq_lengths - 1).clamp(min=0).long()
    return values[torch.arange(values.size(0), device=values.device), idx]


# --- Checkpoint I/O ----------------------------------------------------------

def _cfg_to_dict(cfg: Any) -> Any:
    return asdict(cfg) if is_dataclass(cfg) else cfg


def save_stage_ckpt(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    stage: str,
    cfg: Any,
    step: int,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> None:
    """
    Save a checkpoint in the repo's existing shape (``model_state_dict`` /
    ``optimizer_state_dict``) plus post-training metadata (``stage``, ``cfg``, ``step``,
    ``metrics``). DDP and torch.compile wrappers are unwrapped first so the keys are clean
    and the file loads into a bare model on any device. The write is atomic.
    """
    payload = {
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "stage": stage,
        "cfg": _cfg_to_dict(cfg),
        "step": step,
        "metrics": metrics or {},
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if extra:
        payload.update(extra)
    atomic_save(payload, path)

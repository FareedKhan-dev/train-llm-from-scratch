"""
Shared helpers for the post-training stack: model construction from configs,
frozen reference/old-policy copies, checkpoint I/O (keeping the repo's existing
checkpoint shape), masked reductions, and seeding.
"""

from __future__ import annotations

import contextlib
import copy
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

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


def _strip_ddp_prefix(state_dict: dict) -> dict:
    """Remove a leading ``module.`` from keys saved by DistributedDataParallel."""
    if any(k.startswith("module.") for k in state_dict):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def load_backbone_from_ckpt(cfg: Any, ckpt_path: str, device: str) -> Transformer:
    """
    Build a Transformer from ``cfg`` and load backbone weights from a checkpoint saved
    by the pretraining script or any post-training stage (``model_state_dict`` key).
    DDP ``module.`` prefixes are stripped. Auxiliary head weights (value/reward), if
    present, are ignored here -- wrappers add their own fresh heads.
    """
    model = build_model_from_config(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state = _strip_ddp_prefix(state)
    # Keep only keys that belong to the bare Transformer backbone.
    backbone_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in backbone_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[load_backbone] {len(missing)} missing keys (e.g. {missing[:3]})")
    return model.to(device)


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module behind a DDP wrapper (or the model itself)."""
    return model.module if hasattr(model, "module") else model


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
    ``metrics``). DDP wrappers are unwrapped first so checkpoints load cleanly on a
    single GPU.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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
    torch.save(payload, path)

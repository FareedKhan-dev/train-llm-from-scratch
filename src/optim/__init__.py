"""Optimizers and learning-rate schedules: AdamW (with the GPT weight-decay split) and Muon."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.optim.muon import Muon, muon_param_groups, newton_schulz
from src.optim.schedules import cosine_lr, linear_lr, lr_at, set_lr, wsd_lr


def adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay the >=2D weight matrices, not biases / norms / 1D params (the standard GPT recipe)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """``"adamw"`` or ``"muon"`` (Muon on hidden matrices, AdamW on everything else)."""
    if name == "adamw":
        return torch.optim.AdamW(adamw_param_groups(model, weight_decay), lr=lr, betas=betas)
    if name == "muon":
        return Muon(muon_param_groups(model, weight_decay), lr=lr, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"unknown optimizer {name!r}; expected 'adamw' or 'muon'")


__all__ = [
    "Muon",
    "adamw_param_groups",
    "build_optimizer",
    "cosine_lr",
    "linear_lr",
    "lr_at",
    "muon_param_groups",
    "newton_schulz",
    "set_lr",
    "wsd_lr",
]

"""
Learning-rate schedules. Each one is a pure function ``step -> learning rate``.

- ``cosine``: linear warmup, then a cosine curve down to ``min_lr`` (GPT-3, Llama 2).
- ``wsd``: warmup, then a long *stable* phase at the peak rate, then a short linear *decay*
  over the last ``decay_frac`` of training (MiniCPM, DeepSeek, OLMo 2 annealing). The big
  practical win: a checkpoint from the stable phase can be annealed at any time, so you do
  not need to fix the training length in advance.
- ``linear``: warmup, then a straight line to ``min_lr``. Decaying all the way to zero this
  way matches or beats cosine in recent studies ("Straight to Zero", Bergsma et al. 2025).

Every schedule warms up linearly for ``warmup_steps``: Adam's second-moment estimates are
poor in the first steps, and a full-size step then can wreck a fresh model.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

Schedule = Literal["cosine", "wsd", "linear"]


def _warmup(step: int, warmup_steps: int, lr: float) -> float | None:
    return lr * (step + 1) / max(1, warmup_steps) if step < warmup_steps else None


def cosine_lr(step: int, *, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup to ``lr`` over ``warmup_steps``, then cosine decay to ``min_lr`` by
    ``max_steps`` (constant ``min_lr`` afterwards)."""
    warm = _warmup(step, warmup_steps, lr)
    if warm is not None:
        return warm
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def wsd_lr(
    step: int, *, warmup_steps: int, max_steps: int, lr: float, min_lr: float, decay_frac: float = 0.2
) -> float:
    """Warmup, stay at ``lr``, then decay linearly to ``min_lr`` over the last ``decay_frac``."""
    warm = _warmup(step, warmup_steps, lr)
    if warm is not None:
        return warm
    decay_start = max(warmup_steps, int(max_steps * (1.0 - decay_frac)))
    if step < decay_start:
        return lr
    if step >= max_steps:
        return min_lr
    progress = (step - decay_start) / max(1, max_steps - decay_start)
    return lr + progress * (min_lr - lr)


def linear_lr(step: int, *, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Warmup, then a straight line from ``lr`` to ``min_lr`` at ``max_steps``."""
    warm = _warmup(step, warmup_steps, lr)
    if warm is not None:
        return warm
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr + progress * (min_lr - lr)


def lr_at(schedule: Schedule, step: int, *, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Dispatch to the schedule called ``schedule``."""
    fn = {"cosine": cosine_lr, "wsd": wsd_lr, "linear": linear_lr}.get(schedule)
    if fn is None:
        raise ValueError(f"unknown lr schedule {schedule!r}")
    return fn(step, warmup_steps=warmup_steps, max_steps=max_steps, lr=lr, min_lr=min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set the learning rate of every parameter group (respecting an optional ``lr_scale``)."""
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)

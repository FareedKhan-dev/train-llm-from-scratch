"""
Muon: momentum + orthogonalized updates (Jordan et al. 2024), written from scratch.

Adam treats every number of a weight matrix independently. Muon treats a weight matrix as a
matrix. It keeps an ordinary momentum buffer ``M`` of the gradient, but instead of stepping
along ``M`` it steps along the *orthogonalized* ``M``: the closest matrix whose singular
values are all 1 (``U V^T`` if ``M = U S V^T``). Every direction the gradient points in gets
the same step size, so rare but useful directions are not drowned out by a few dominant ones.

Computing an SVD every step would be slow, so Muon uses a few Newton-Schulz iterations,
cheap matrix multiplications that push all singular values toward 1:

    X <- a X + b (X X^T) X + c (X X^T)^2 X          (a, b, c) = (3.4445, -4.7750, 2.0315)

Muon trained nanoGPT speedruns faster than AdamW and was scaled to a 1T parameter model
(Kimi K2, with "MuonClip"). It only makes sense for 2D hidden matrices; embeddings, the
output layer, norms and biases keep using AdamW. This class does both, so it drops into any
training loop that expects one optimizer.

Learning-rate scale: with ``adjust_lr="match_rms_adamw"`` (Liu et al. 2025, "Muon is
Scalable for LLM Training") the Muon update is rescaled to the size of a typical AdamW
update, so the *same* learning rate and weight decay work for both parameter groups. That is
the default here, because it lets the existing LR schedules drive Muon unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
AdjustLR = Literal["match_rms_adamw", "original"]


def newton_schulz(
    G: Float[Tensor, "rows cols"],
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: tuple[float, float, float] = NS_COEFFICIENTS,
    dtype: torch.dtype = torch.bfloat16,
) -> Float[Tensor, "rows cols"]:
    """Approximately orthogonalize ``G`` (all singular values pushed to about 1)."""
    a, b, c = coefficients
    X = G.to(dtype)
    transposed = G.size(0) > G.size(1)
    if transposed:  # iterate on the wide orientation, so X @ X.T is the small Gram matrix
        X = X.T
    X = X / X.norm().clamp(min=eps)  # the Frobenius norm bounds the spectral norm: start <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


def adjusted_lr(lr: float, shape: torch.Size, mode: AdjustLR) -> float:
    rows, cols = shape[0], shape[1]
    if mode == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(rows, cols))
    return lr * math.sqrt(max(1.0, rows / cols))


class Muon(torch.optim.Optimizer):
    """
    Muon for parameter groups with ``use_muon=True``, AdamW for the others.

    Build the groups with :func:`muon_param_groups`, or pass your own list of dicts.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr: AdjustLR = "match_rms_adamw",
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, adjust_lr=adjust_lr, betas=betas, eps=eps, use_muon=True)
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["use_muon"] and any(p.ndim != 2 for p in group["params"]):
                raise ValueError("Muon groups may only hold 2D weight matrices; put the rest in an AdamW group")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - group["momentum"])  # buf = momentum * buf + (1 - momentum) * grad
            update = p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
            update = newton_schulz(update, group["ns_steps"]).to(p.dtype)
            p.mul_(1 - group["lr"] * group["weight_decay"])  # decoupled weight decay, as in AdamW
            p.add_(update, alpha=-adjusted_lr(group["lr"], p.shape, group["adjust_lr"]))

    def _adamw_step(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            m, v = state["exp_avg"], state["exp_avg_sq"]
            m.lerp_(p.grad, 1 - beta1)
            v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            p.mul_(1 - group["lr"] * group["weight_decay"])
            denom = (v / bias2).sqrt_().add_(group["eps"])
            p.addcdiv_(m, denom, value=-group["lr"] / bias1)


def muon_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """
    Split ``model`` into a Muon group (2D hidden weight matrices) and an AdamW group
    (embeddings, the output layer, norms, biases). Matches the usual Muon recipe.
    """
    skip: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            skip.update(id(p) for p in module.parameters())
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Module):
        skip.update(id(p) for p in lm_head.parameters())

    muon, adam_decay, adam_no_decay = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and id(p) not in skip:
            muon.append(p)
        elif p.ndim >= 2:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)
    return [
        {"params": muon, "use_muon": True, "weight_decay": weight_decay},
        {"params": adam_decay, "use_muon": False, "weight_decay": weight_decay},
        {"params": adam_no_decay, "use_muon": False, "weight_decay": 0.0},
    ]

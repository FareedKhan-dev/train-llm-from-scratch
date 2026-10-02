"""
LoRA: low-rank adaptation (Hu et al. 2021), written from scratch.

Full fine-tuning updates every weight matrix ``W`` (``d_out x d_in`` numbers each). LoRA
freezes ``W`` and learns a low-rank correction instead:

    y = x W^T + (alpha / r) * x A^T B^T          A: (r x d_in),  B: (d_out x r)

With ``r = 8`` and ``d = 1024`` that is 16k trainable numbers per matrix instead of 1M.
``B`` starts at zero, so training starts exactly from the pretrained model, and after
training ``W + (alpha / r) B A`` can be folded back into one matrix (:func:`merge_lora`),
so inference costs nothing extra.

Why it works: fine-tuning changes turn out to be low-rank in practice, and the optimizer
state (Adam keeps two numbers per trainable weight) shrinks with the trainable count, so
LoRA fits on much smaller GPUs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

# Linear layers adapted by default: the attention projections of both architectures, and the
# MLP output projection of the classic model (it is also named "proj").
DEFAULT_TARGETS = frozenset({"query", "key", "value", "proj", "q_proj", "k_proj", "v_proj", "o_proj"})


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update ``(alpha / r) * B @ A``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.rank, self.scaling = rank, alpha / rank
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # same init as nn.Linear
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Float[Tensor, "*batch d_in"]) -> Float[Tensor, "*batch d_out"]:
        update = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return self.base(x) + update * self.scaling

    def merged(self) -> nn.Linear:
        """A plain ``nn.Linear`` with the update folded in: ``W + scaling * B @ A``."""
        out = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None,
                        device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            out.weight.copy_(self.base.weight + self.scaling * (self.lora_B @ self.lora_A))
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> list[str]:
    """
    Freeze ``model`` and wrap every ``nn.Linear`` whose attribute name is in ``targets`` with a
    :class:`LoRALinear`. Only the LoRA matrices stay trainable. Returns the wrapped names.
    """
    targets = set(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                wrapped.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    if not wrapped:
        raise ValueError(f"no nn.Linear layer named one of {sorted(targets)} was found")
    return wrapped


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every LoRA update into its base layer (in place) and make all weights trainable again."""
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, child_name, child.merged())
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def lora_parameter_count(model: nn.Module) -> tuple[int, int]:
    """``(trainable, total)`` parameter counts, to report how small the update is."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

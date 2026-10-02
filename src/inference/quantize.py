"""
Int8 weight-only quantization, written from scratch.

A float32 weight takes 4 bytes; an int8 takes 1. Weight-only quantization stores every
``nn.Linear`` weight matrix as int8 plus one float scale per output row:

    scale_r = max(|W_r|) / 127            (per output channel, "absmax")
    Q_r     = round(W_r / scale_r)        (integers in [-127, 127])
    W_r    ~= Q_r * scale_r               (dequantized on the fly in forward)

The matmul still runs in floating point, so this saves memory (about 4x for the linear
layers) rather than compute. That is the main win for inference on small devices, where
memory bandwidth, not arithmetic, limits token speed. Per-channel scales keep the error
small because each row gets its own range.

The output layer is left alone by default: when its weight is tied to the input embedding,
quantizing it would silently untie them, and the vocabulary projection is the most
sensitive layer anyway.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class Int8Linear(nn.Module):
    """A frozen ``nn.Linear`` whose weight is stored as int8 with one scale per output row."""

    weight_int8: Tensor
    scale: Tensor

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("weight_int8", torch.round(w / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def dequantized_weight(self) -> Float[Tensor, "out in"]:
        return self.weight_int8.float() * self.scale

    def forward(self, x: Float[Tensor, "*batch in_features"]) -> Float[Tensor, "*batch out_features"]:
        weight = self.dequantized_weight().to(x.dtype)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, weight, bias)


def quantize_int8(model: nn.Module, skip: Iterable[str] = ("lm_head",)) -> nn.Module:
    """Replace every ``nn.Linear`` (except names in ``skip``) by :class:`Int8Linear`, in place."""
    skip = set(skip)
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(parent, name, Int8Linear(child))
    return model


def model_size_bytes(model: nn.Module) -> int:
    """Bytes of all parameters and buffers (shared tensors counted once)."""
    seen: set[int] = set()
    total = 0
    for t in [*model.parameters(), *model.buffers()]:
        if id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()
    return total

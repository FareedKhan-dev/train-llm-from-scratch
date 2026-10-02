"""
Rotary position embeddings (RoPE, Su et al. 2021), used by nearly every modern LLM.

The classic model *adds* a learned position vector to each token. RoPE instead *rotates* the
query and key vectors by an angle that grows with the position. Split a head vector into
pairs of numbers; pair ``i`` at position ``m`` is rotated by the angle ``m * theta_i`` with

    theta_i = rope_theta ** (-2i / head_dim)

so early pairs spin fast (they track nearby words) and late pairs spin slowly (they track
long-range structure). The useful property: the dot product of a query at position ``m`` and
a key at position ``n`` only depends on ``m - n``. Attention therefore sees *relative*
positions, with no position parameters to learn.

We use the "rotate half" layout (pair ``i`` is ``(x[i], x[i + d/2])``), as in most open code.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


def rope_frequencies(dim: int, theta: float = 10_000.0) -> Float[Tensor, " half"]:
    """The rotation speed ``theta_i`` of each of the ``dim / 2`` pairs."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def rope_cache(
    dim: int, max_len: int, theta: float = 10_000.0
) -> tuple[Float[Tensor, "seq half"], Float[Tensor, "seq half"]]:
    """Precompute ``cos`` and ``sin`` of every angle ``position * theta_i``."""
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, rope_frequencies(dim, theta))
    return angles.cos(), angles.sin()


def apply_rope(
    x: Float[Tensor, "*batch seq dim"], cos: Float[Tensor, "seq half"], sin: Float[Tensor, "seq half"]
) -> Float[Tensor, "*batch seq dim"]:
    """Rotate every pair ``(x1, x2)`` of ``x`` by its angle: a 2D rotation per pair."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

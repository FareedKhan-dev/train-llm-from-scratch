"""
SwiGLU feed-forward block (Shazeer, 2020), used by Llama, Qwen, Mistral, DeepSeek and Gemma.

The classic MLP is ``proj(relu(hidden(x)))``. A gated linear unit computes two projections
and lets one gate the other:

    SwiGLU(x) = down( silu(gate(x)) * up(x) )

The gate decides, feature by feature, how much of ``up(x)`` passes through. It has three
matrices instead of two, so the hidden width is ``8/3 * n_embed`` instead of ``4 * n_embed``
to keep the parameter count the same. It consistently trains to a lower loss than ReLU or GELU.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class SwiGLU(nn.Module):
    def __init__(self, n_embed: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(n_embed, hidden, bias=False)
        self.up = nn.Linear(n_embed, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embed, bias=False)

    def forward(self, x: Float[Tensor, "*batch embed"]) -> Float[Tensor, "*batch embed"]:
        return self.down(F.silu(self.gate(x)) * self.up(x))

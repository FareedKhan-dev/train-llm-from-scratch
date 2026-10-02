"""
One block of the modern model: the same pre-norm residual shape as the classic block,

    x = x + attention(norm(x))
    x = x + mlp(norm(x))

with RMSNorm instead of LayerNorm, rotary attention (GQA or MLA) instead of the classic
heads, and a SwiGLU or Mixture-of-Experts layer instead of the ReLU MLP.
"""

from __future__ import annotations

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from src.models.modern.attention import GroupedQueryAttention, MultiHeadLatentAttention
from src.models.modern.config import ModernConfig
from src.models.modern.kv_cache import KVCache
from src.models.modern.mlp import SwiGLU
from src.models.modern.moe import MoE
from src.models.modern.norm import RMSNorm


class ModernBlock(nn.Module):
    def __init__(self, cfg: ModernConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embed, cfg.norm_eps)
        self.attn: nn.Module = (
            MultiHeadLatentAttention(cfg, layer_idx) if cfg.attention == "mla" else GroupedQueryAttention(cfg, layer_idx)
        )
        self.mlp_norm = RMSNorm(cfg.n_embed, cfg.norm_eps)
        self.mlp: nn.Module = (
            MoE(cfg.n_embed, cfg.hidden_dim, cfg.n_experts, cfg.moe_top_k, cfg.n_shared_experts)
            if cfg.n_experts > 0
            else SwiGLU(cfg.n_embed, cfg.hidden_dim)
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq embed"],
        cos: Float[Tensor, "seq half"],
        sin: Float[Tensor, "seq half"],
        cache: KVCache | None = None,
    ) -> Float[Tensor, "batch seq embed"]:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x

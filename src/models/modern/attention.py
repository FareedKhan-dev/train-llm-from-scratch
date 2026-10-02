"""
Attention for the modern model: grouped-query attention and multi-head latent attention.

Compared with the classic ``MultiHeadAttention`` (one small module per head, a stored
triangular mask, and the softmax written out by hand), this file adds five ideas:

1. **Fused projections.** One matrix multiply produces the queries of every head.
2. **Grouped-query attention** (GQA, Ainslie et al. 2023). Several query heads share one
   key/value head. With ``n_kv_head = n_head`` it is the usual multi-head attention, with
   ``n_kv_head = 1`` it is multi-query attention. Fewer key/value heads means a smaller KV
   cache, which is what limits batch size and context length at inference time.
3. **Rotary embeddings and QK-norm.** Positions enter by rotating queries and keys (see
   ``rope.py``), and RMSNorm on queries and keys keeps the attention logits from blowing up.
4. **Fused attention kernel.** ``F.scaled_dot_product_attention`` computes
   ``softmax(Q K^T / sqrt(d)) V`` without storing the (T x T) score matrix when it can
   (FlashAttention on GPUs), which is a big memory saving for long sequences.
5. **Multi-head latent attention** (MLA, DeepSeek-V2). Keys and values are rebuilt from one
   small latent vector per token, and only that latent is cached.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from src.models.modern.config import ModernConfig
from src.models.modern.kv_cache import KVCache
from src.models.modern.norm import RMSNorm
from src.models.modern.rope import apply_rope


def attention_mask(
    q_len: int, k_len: int, start: int, window: int | None, device: torch.device
) -> tuple[Bool[Tensor, "q_len k_len"] | None, bool]:
    """
    Decide how to mask attention. Returns ``(mask, is_causal)`` for SDPA.

    ``start`` is the position of the first query (the number of cached tokens before it).
    Query ``i`` (absolute position ``start + i``) may look at key ``j`` when ``j <= start + i``,
    and, with a sliding window, only when ``j > start + i - window``.

    The common cases need no mask tensor at all: a fresh causal pass uses SDPA's built-in
    ``is_causal`` flag, and a single new token with a cache may see every cached key.
    """
    if window is None and start == 0 and q_len == k_len:
        return None, q_len > 1
    if window is None and q_len == 1:
        return None, False
    q_pos = torch.arange(start, start + q_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    allowed = k_pos <= q_pos
    if window is not None:
        allowed &= k_pos > q_pos - window
    return allowed, False


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModernConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.kv_heads, cfg.head_dim
        self.window, self.dropout = cfg.sliding_window, cfg.dropout
        C, H, Hkv, D = cfg.n_embed, self.n_head, self.n_kv_head, self.head_dim
        self.q_proj = nn.Linear(C, H * D, bias=False)
        self.k_proj = nn.Linear(C, Hkv * D, bias=False)
        self.v_proj = nn.Linear(C, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, C, bias=False)
        self.q_norm: nn.Module = RMSNorm(D, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm: nn.Module = RMSNorm(D, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.gate = nn.Linear(C, H * D, bias=False) if cfg.attn_gate else None

    def forward(
        self,
        x: Float[Tensor, "batch seq embed"],
        cos: Float[Tensor, "seq half"],
        sin: Float[Tensor, "seq half"],
        cache: KVCache | None = None,
    ) -> Float[Tensor, "batch seq embed"]:
        B, T, _ = x.shape
        H, Hkv, D = self.n_head, self.n_kv_head, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)  # (B, Hkv, T, D)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        start = 0
        if cache is not None:
            start = cache.length
            k, v = cache.update(self.layer_idx, k, v)  # all keys/values so far: (B, Hkv, start+T, D)
        mask, is_causal = attention_mask(T, k.size(2), start, self.window, x.device)

        if Hkv != H:  # every group of H // Hkv query heads reads the same key/value head
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(B, T, H * D)
        if self.gate is not None:  # gated attention: each head decides how much of its output to pass on
            y = y * torch.sigmoid(self.gate(x))
        return self.o_proj(y)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head latent attention, simplified from DeepSeek-V2/V3.

    Each token is compressed into a latent ``c`` of ``kv_latent_dim`` numbers; every head's key
    and value are linear read-outs of that latent. Rotary embeddings would break this (a
    rotated key can no longer be rebuilt from the cached latent), so position travels in a
    separate, small ``rope_dim`` part of each query and in one shared rotated key. The cache
    therefore holds only ``kv_latent_dim + rope_dim`` numbers per token per layer, instead of
    ``2 * n_kv_head * head_dim`` for GQA.

    (Production implementations also fold the key read-out into the query projection at
    inference time; we keep the read-out explicit because it is easier to follow.)
    """

    def __init__(self, cfg: ModernConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.latent_dim, self.rope_dim = cfg.latent_dim, cfg.mla_rope_dim
        self.window, self.dropout = cfg.sliding_window, cfg.dropout
        C, H, D, R, Dr = cfg.n_embed, cfg.n_head, cfg.head_dim, self.latent_dim, self.rope_dim
        self.q_proj = nn.Linear(C, H * (D + Dr), bias=False)  # per head: content part + position part
        self.kv_down = nn.Linear(C, R + Dr, bias=False)  # the latent, plus one shared position key
        self.kv_norm = RMSNorm(R, cfg.norm_eps)
        self.k_up = nn.Linear(R, H * D, bias=False)  # latent -> content keys of every head
        self.v_up = nn.Linear(R, H * D, bias=False)  # latent -> values of every head
        self.o_proj = nn.Linear(H * D, C, bias=False)

    def forward(
        self,
        x: Float[Tensor, "batch seq embed"],
        cos: Float[Tensor, "seq half"],
        sin: Float[Tensor, "seq half"],
        cache: KVCache | None = None,
    ) -> Float[Tensor, "batch seq embed"]:
        B, T, _ = x.shape
        H, D, R, Dr = self.n_head, self.head_dim, self.latent_dim, self.rope_dim
        q = self.q_proj(x).view(B, T, H, D + Dr).transpose(1, 2)
        q_content, q_pos = q.split([D, Dr], dim=-1)
        q_pos = apply_rope(q_pos, cos, sin)

        latent, k_pos = self.kv_down(x).split([R, Dr], dim=-1)
        latent = self.kv_norm(latent).unsqueeze(1)  # (B, 1, T, R)
        k_pos = apply_rope(k_pos.unsqueeze(1), cos, sin)  # (B, 1, T, Dr), shared by all heads

        start = 0
        if cache is not None:  # only the latent and the position key are stored
            start = cache.length
            latent, k_pos = cache.update(self.layer_idx, latent, k_pos)
        S = latent.size(2)
        k_content = self.k_up(latent.squeeze(1)).view(B, S, H, D).transpose(1, 2)
        v = self.v_up(latent.squeeze(1)).view(B, S, H, D).transpose(1, 2)
        k = torch.cat([k_content, k_pos.expand(B, H, S, Dr)], dim=-1)
        q = torch.cat([q_content, q_pos], dim=-1)

        mask, is_causal = attention_mask(T, S, start, self.window, x.device)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0, scale=1.0 / math.sqrt(D + Dr),
        )
        return self.o_proj(y.transpose(1, 2).reshape(B, T, H * D))

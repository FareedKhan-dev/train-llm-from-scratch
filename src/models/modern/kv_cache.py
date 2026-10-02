"""
A key/value cache for fast generation.

Without a cache, generating token ``t`` re-runs the whole prefix through every layer, so a
reply of ``n`` tokens costs O(n^2) work. But the keys and values of old tokens never change
(the attention is causal), so we can keep them. Each new token then only computes its own
query, key and value and attends to the stored ones: generation becomes O(n).

The cache is the main memory cost of serving an LLM, which is exactly why grouped-query
attention and multi-head latent attention exist: they shrink what has to be stored per token.
:meth:`KVCache.bytes_per_token` makes that cost visible.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


class KVCache:
    """Preallocated keys and values for every layer. ``length`` counts the tokens stored."""

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_len: int,
        n_kv_heads: int,
        k_dim: int,
        v_dim: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_len = max_len
        self.length = 0
        shape = (n_layers, batch_size, n_kv_heads, max_len)
        self.k = torch.zeros(*shape, k_dim, device=device, dtype=dtype)
        self.v = torch.zeros(*shape, v_dim, device=device, dtype=dtype)

    def update(
        self, layer: int, k_new: Float[Tensor, "batch heads seq k_dim"], v_new: Float[Tensor, "batch heads seq v_dim"]
    ) -> tuple[Float[Tensor, "batch heads total k_dim"], Float[Tensor, "batch heads total v_dim"]]:
        """Store this step's keys/values for ``layer`` and return everything cached so far."""
        end = self.length + k_new.size(2)
        if end > self.max_len:
            raise ValueError(f"KV cache is full ({self.max_len} tokens); reset it or use a longer one")
        self.k[layer, :, :, self.length : end] = k_new
        self.v[layer, :, :, self.length : end] = v_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n_tokens: int) -> None:
        """Called once per forward pass, after every layer has stored its new entries."""
        self.length += n_tokens

    def reset(self) -> None:
        self.length = 0

    def bytes_per_token(self) -> int:
        """Memory one token costs across all layers (keys + values)."""
        n_layers, batch, heads = self.k.shape[:3]
        per_token = heads * (self.k.shape[-1] + self.v.shape[-1]) * self.k.element_size()
        return n_layers * per_token

"""
The modern decoder: the same job as the classic ``Transformer``, built the way 2024-2026
open models (Llama 3/4, Qwen 3, Gemma 3, DeepSeek-V3, OLMo 2) are built.

    token ids -> embedding -> N x ModernBlock -> RMSNorm -> lm_head (tied to the embedding)

What changed compared with the classic model, and why:

- No learned position table: positions come from rotary embeddings inside attention, so
  there are no position parameters and attention sees relative distances.
- RMSNorm instead of LayerNorm, SwiGLU instead of ReLU, no biases in linear layers.
- Grouped-query attention or multi-head latent attention, with a KV cache for generation.
- Optional Mixture of Experts, sliding-window attention and gated attention.
- The output layer shares its weight with the input embedding ("tied embeddings"), which
  saves ``vocab_size * n_embed`` parameters, a large fraction of a small model.
- Small, depth-scaled initialization: residual output projections start at
  ``init_std / sqrt(2 * n_blocks)`` so the residual stream does not grow with depth (GPT-2).

It has the same interface as the classic model (``forward``, ``forward_hidden``, ``lm_head``,
``context_length``, ``generate``), so every post-training stage works with it unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from jaxtyping import Float, Int
from torch import Tensor

from src.inference.sampling import sample_next_token
from src.models.modern.block import ModernBlock
from src.models.modern.config import ModernConfig
from src.models.modern.kv_cache import KVCache
from src.models.modern.moe import MoE
from src.models.modern.norm import RMSNorm
from src.models.modern.rope import rope_cache


class ModernTransformer(nn.Module):
    def __init__(self, config: ModernConfig) -> None:
        super().__init__()
        self.config = config
        self.context_length = config.context_length
        self.gradient_checkpointing = False
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([ModernBlock(config, i) for i in range(config.n_blocks)])
        self.final_norm = RMSNorm(config.n_embed, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        cos, sin = rope_cache(config.rope_table_dim, config.context_length, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)  # rebuilt on load, never saved
        self.register_buffer("rope_sin", sin, persistent=False)
        self.aux_loss: Tensor | None = None

        self.apply(self._init_weights)
        residual_std = config.init_std / math.sqrt(2 * config.n_blocks)
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    # ------------------------------------------------------------------ forward
    def forward_hidden(
        self, idx: Int[Tensor, "batch seq"], cache: KVCache | None = None
    ) -> Float[Tensor, "batch seq embed"]:
        """Hidden states after the final norm (what ``lm_head`` reads; reward/value heads reuse it)."""
        T = idx.size(1)
        start = cache.length if cache is not None else 0
        if start + T > self.context_length:
            raise ValueError(f"sequence of {start + T} tokens exceeds context_length {self.context_length}")
        cos, sin = self.rope_cos[start : start + T], self.rope_sin[start : start + T]
        x = self.drop(self.token_embed(idx))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and cache is None:
                x = checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin, cache)
        if cache is not None:
            cache.advance(T)
        self.aux_loss = self._moe_aux_loss()
        return self.final_norm(x)

    def forward(
        self, idx: Int[Tensor, "batch seq"], targets: Int[Tensor, "batch seq"] | None = None
    ) -> tuple[Float[Tensor, "batch seq vocab"], Float[Tensor, ""] | None]:
        logits = self.lm_head(self.forward_hidden(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1).long())
            if self.aux_loss is not None:
                loss = loss + self.config.moe_aux_loss_coef * self.aux_loss
        return logits, loss

    def _moe_aux_loss(self) -> Tensor | None:
        losses = [b.mlp.aux_loss for b in self.blocks if isinstance(b.mlp, MoE) and b.mlp.aux_loss is not None]
        return torch.stack(losses).mean() if losses else None

    # ------------------------------------------------------------------ generation
    def new_cache(self, batch_size: int) -> KVCache:
        """An empty KV cache sized for this model (it stores the latent for MLA)."""
        cfg = self.config
        ref = self.token_embed.weight
        if cfg.attention == "mla":
            heads, k_dim, v_dim = 1, cfg.latent_dim, cfg.mla_rope_dim
        else:
            heads, k_dim, v_dim = cfg.kv_heads, cfg.head_dim, cfg.head_dim
        return KVCache(cfg.n_blocks, batch_size, cfg.context_length, heads, k_dim, v_dim,
                       device=ref.device, dtype=ref.dtype)

    @torch.no_grad()
    def generate(
        self,
        idx: Int[Tensor, "batch seq"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        use_cache: bool = True,
        context_window: int | None = None,
    ) -> Int[Tensor, "batch total"]:
        """
        Sample ``max_new_tokens`` tokens after ``idx``.

        With the cache, the prompt is processed once ("prefill") and every later step feeds only
        the newest token ("decode"). When the window fills up, the most recent half is
        re-encoded into a fresh cache, so generation can run longer than ``context_length``.
        """
        window = min(context_window or self.context_length, self.context_length)
        if not use_cache:
            for _ in range(max_new_tokens):
                logits = self(idx[:, -window:])[0][:, -1, :]
                idx = torch.cat([idx, sample_next_token(logits, temperature, top_k, top_p, min_p)], dim=1)
            return idx

        cache = self.new_cache(idx.size(0))
        hidden = self.forward_hidden(idx[:, -window:], cache)  # prefill
        for _ in range(max_new_tokens):
            logits = self.lm_head(hidden[:, -1, :])
            next_token = sample_next_token(logits, temperature, top_k, top_p, min_p)
            idx = torch.cat([idx, next_token], dim=1)
            if cache.length >= window:  # out of room: keep the latest half and continue
                cache.reset()
                hidden = self.forward_hidden(idx[:, -(window // 2) :], cache)
            else:
                hidden = self.forward_hidden(next_token, cache)  # decode one token
        return idx

    # ------------------------------------------------------------------ accounting
    def num_params(self, exclude_embeddings: bool = False) -> int:
        """Parameter count (tied weights counted once)."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.token_embed.weight.numel() if exclude_embeddings else n

    def active_params(self) -> int:
        """Parameters used per token: with MoE only ``moe_top_k`` of the routed experts run."""
        n = self.num_params()
        for block in self.blocks:
            if isinstance(block.mlp, MoE):
                per_expert = sum(p.numel() for p in block.mlp.experts[0].parameters())
                n -= per_expert * (block.mlp.n_experts - block.mlp.top_k)
        return n

    def flops_per_token(self, seq_len: int | None = None) -> float:
        """
        Training FLOPs per token (forward + backward), following PaLM appendix B:

        - ``6`` FLOPs per weight that multiplies the token (2 forward, 4 backward). The input
          embedding is a lookup and costs nothing; the output layer is a real matmul.
        - the attention scores and the weighted sum of values, which grow with ``seq_len``.
        """
        cfg = self.config
        T = seq_len or cfg.context_length
        body = self.active_params() - self.token_embed.weight.numel()
        if not cfg.tie_embeddings:
            body -= self.lm_head.weight.numel()
        qk_dim = cfg.head_dim + (cfg.mla_rope_dim if cfg.attention == "mla" else 0)
        attention = 6 * cfg.n_blocks * cfg.n_head * T * (qk_dim + cfg.head_dim)
        return 6 * body + 6 * cfg.n_embed * cfg.vocab_size + attention

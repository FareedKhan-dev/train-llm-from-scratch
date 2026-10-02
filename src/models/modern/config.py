"""
Settings for :class:`~src.models.modern.model.ModernTransformer`.

The defaults describe a small Llama / Qwen style decoder. Every switch maps to one idea from
the last few years of LLM research, and each one can be turned off to see what it buys:

==================  =====================================================================
field               what it does
==================  =====================================================================
``n_kv_head``       grouped-query attention: several query heads share one key/value head
``attention``       ``"gqa"`` (standard) or ``"mla"`` (DeepSeek's multi-head latent attention)
``rope_theta``      base of the rotary position embedding frequencies
``qk_norm``         RMSNorm on queries and keys before attention (OLMo 2, Qwen 3, Gemma 3)
``attn_gate``       sigmoid gate on the attention output (Qwen3-Next, gated attention)
``sliding_window``  each token only attends to the last N tokens (Mistral, Gemma 3)
``tie_embeddings``  the output layer reuses the input embedding matrix
``n_experts``       > 0 swaps the dense MLP for a Mixture of Experts (Mixtral, DeepSeek, Qwen 3)
==================  =====================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

AttentionKind = Literal["gqa", "mla"]


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class ModernConfig:
    # size
    vocab_size: int = 50304
    context_length: int = 1024
    n_embed: int = 768
    n_head: int = 12
    n_blocks: int = 12
    ffn_hidden: int | None = None  # SwiGLU width; None = 8/3 * n_embed rounded up to 64
    # attention
    attention: AttentionKind = "gqa"
    n_kv_head: int | None = None  # None = n_head (plain multi-head); 1 = multi-query
    kv_latent_dim: int | None = None  # MLA: size of the cached key/value latent; None = n_embed // 4
    rope_dim: int | None = None  # MLA: per-head dims that carry position; None = head_dim // 2
    rope_theta: float = 10_000.0
    qk_norm: bool = True
    attn_gate: bool = False
    sliding_window: int | None = None
    # mixture of experts
    n_experts: int = 0  # 0 = dense SwiGLU MLP
    moe_top_k: int = 2
    n_shared_experts: int = 0
    moe_aux_loss_coef: float = 0.01
    # misc
    tie_embeddings: bool = True
    norm_eps: float = 1e-6
    init_std: float = 0.02
    dropout: float = 0.0

    def __post_init__(self) -> None:
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"ModernConfig: {msg}")

        for name in ("vocab_size", "context_length", "n_embed", "n_head", "n_blocks"):
            need(getattr(self, name) > 0, f"{name} must be positive")
        need(self.n_embed % self.n_head == 0, "n_embed must be divisible by n_head")
        need(self.head_dim % 2 == 0, "head_dim (n_embed / n_head) must be even for rotary embeddings")
        need(self.attention in ("gqa", "mla"), "attention must be 'gqa' or 'mla'")
        need(self.n_head % self.kv_heads == 0, "n_head must be divisible by n_kv_head")
        need(self.mla_rope_dim % 2 == 0, "rope_dim must be even")
        need(self.sliding_window is None or self.sliding_window > 0, "sliding_window must be positive")
        need(self.n_experts >= 0, "n_experts must be >= 0")
        if self.n_experts:
            need(1 <= self.moe_top_k <= self.n_experts, "moe_top_k must be between 1 and n_experts")
        need(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")

    # ------------------------------------------------------------------ derived sizes
    @property
    def head_dim(self) -> int:
        return self.n_embed // self.n_head

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def hidden_dim(self) -> int:
        return self.ffn_hidden or round_up(int(8 * self.n_embed / 3), 64)

    @property
    def latent_dim(self) -> int:
        return self.kv_latent_dim or max(16, self.n_embed // 4)

    @property
    def mla_rope_dim(self) -> int:
        return self.rope_dim or max(2, self.head_dim // 2)

    @property
    def rope_table_dim(self) -> int:
        """Width of the cos/sin tables: the full head for GQA, the decoupled part for MLA."""
        return self.mla_rope_dim if self.attention == "mla" else self.head_dim

    # ------------------------------------------------------------------ conversion
    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ModernConfig:
        """Build from any mapping (a JSON config, a checkpoint ``cfg``), ignoring unrelated keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

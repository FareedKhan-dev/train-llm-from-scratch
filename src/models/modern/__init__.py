"""The modern (2024-2026 style) decoder. See ``model.py`` for what changed and why."""

from src.models.modern.attention import GroupedQueryAttention, MultiHeadLatentAttention, attention_mask
from src.models.modern.block import ModernBlock
from src.models.modern.config import ModernConfig
from src.models.modern.kv_cache import KVCache
from src.models.modern.mlp import SwiGLU
from src.models.modern.model import ModernTransformer
from src.models.modern.moe import MoE
from src.models.modern.norm import RMSNorm
from src.models.modern.rope import apply_rope, rope_cache, rope_frequencies

__all__ = [
    "GroupedQueryAttention",
    "KVCache",
    "MoE",
    "ModernBlock",
    "ModernConfig",
    "ModernTransformer",
    "MultiHeadLatentAttention",
    "RMSNorm",
    "SwiGLU",
    "apply_rope",
    "attention_mask",
    "rope_cache",
    "rope_frequencies",
]

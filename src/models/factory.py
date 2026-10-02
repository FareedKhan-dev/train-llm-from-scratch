"""
Build a model from any config: a post-training dataclass, the legacy ``default_config`` dict,
or the ``cfg`` stored inside a checkpoint.

``arch`` picks the architecture:

- ``"classic"`` (default): the original 2017-style Transformer in ``src/models/transformer.py``.
- ``"modern"``: the Llama/Qwen-style decoder in ``src/models/modern`` (RoPE, RMSNorm, SwiGLU,
  GQA or MLA, optional MoE). Its extra settings (``n_kv_head``, ``n_experts``, ...) are read
  from the same config when present.

Checkpoints written before ``arch`` existed have no such key, so they load as ``"classic"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import torch.nn as nn

from src.models.modern import ModernConfig, ModernTransformer
from src.models.transformer import Transformer

ARCHITECTURES = ("classic", "modern")


def config_as_dict(cfg: Any) -> dict[str, Any]:
    """A plain dict view of a dataclass instance, a mapping, or a namespace."""
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    return dict(vars(cfg))


def build_model(cfg: Any) -> nn.Module:
    """Construct a fresh, randomly initialized model described by ``cfg``."""
    values = config_as_dict(cfg)
    arch = values.get("arch") or "classic"
    if arch == "classic":
        return Transformer(
            n_head=values["n_head"],
            n_embed=values["n_embed"],
            context_length=values["context_length"],
            vocab_size=values["vocab_size"],
            N_BLOCKS=values["n_blocks"],
        )
    if arch == "modern":
        return ModernTransformer(ModernConfig.from_mapping(values))
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHITECTURES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

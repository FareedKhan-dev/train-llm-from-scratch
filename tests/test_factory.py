"""One factory builds either architecture, and checkpoints remember which one they hold."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from config.loader import load_config
from config.post_training_config import SFTConfig, smoke
from src.models.factory import build_model
from src.models.modern import ModernTransformer
from src.models.transformer import Transformer
from src.post_training.inference import load_model_from_ckpt
from src.post_training.utils import load_backbone_from_ckpt, save_stage_ckpt

DIMS = {"n_head": 4, "n_embed": 32, "context_length": 16, "vocab_size": 64, "n_blocks": 2}


def test_build_model_picks_the_architecture() -> None:
    assert isinstance(build_model(DIMS), Transformer)  # no arch key: classic, like old checkpoints
    assert isinstance(build_model({**DIMS, "arch": "classic"}), Transformer)
    modern = build_model({**DIMS, "arch": "modern", "n_kv_head": 2, "lr": 1e-3})  # unrelated keys ignored
    assert isinstance(modern, ModernTransformer) and modern.config.kv_heads == 2
    with pytest.raises(ValueError, match="unknown arch"):
        build_model({**DIMS, "arch": "mamba"})


def test_cli_style_overrides_build_a_modern_model() -> None:
    cfg = load_config(SFTConfig, "configs/smoke/sft.json", {"arch": "modern", "n_kv_head": "2", "n_experts": "4"})
    model = build_model(cfg)
    assert isinstance(model, ModernTransformer)
    assert model.config.kv_heads == 2 and model.config.n_experts == 4


def test_modern_checkpoints_round_trip(tmp_path: str) -> None:
    cfg = replace(smoke(SFTConfig), arch="modern", n_kv_head=2)
    torch.manual_seed(0)
    model = build_model(cfg)
    path = os.path.join(tmp_path, "modern.pt")
    save_stage_ckpt(path, model, None, stage="sft", cfg=cfg, step=1)
    for loaded in (load_model_from_ckpt(path, "cpu"), load_backbone_from_ckpt(cfg, path, "cpu")):
        assert isinstance(loaded, ModernTransformer)
        for (k, a), b in zip(model.state_dict().items(), loaded.state_dict().values()):
            assert torch.equal(a, b), k

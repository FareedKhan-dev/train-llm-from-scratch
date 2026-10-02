"""scripts/model_report.py: its counts and FLOPs agree with the real models."""

from __future__ import annotations

import pytest
import torch

from config.config import default_config
from config.presets import apply_preset
from scripts.model_report import estimate
from src.models.factory import build_model
from src.models.modern import ModernConfig, ModernTransformer


def test_parameter_count_of_the_readme_13m_model() -> None:
    assert estimate(apply_preset(dict(default_config), "13m"))["total"] == 13_142_656


@pytest.mark.parametrize("extra", [{}, {"n_kv_head": 2}, {"attention": "mla"}, {"n_experts": 4}, {"tie_embeddings": False}])
def test_flops_match_the_model_formula(extra: dict) -> None:
    cfg = {"arch": "modern", "vocab_size": 512, "context_length": 64, "n_embed": 64, "n_head": 4, "n_blocks": 2, **extra}
    model = ModernTransformer(ModernConfig.from_mapping(cfg))
    e = estimate(cfg)
    assert e["total"] == model.num_params()
    assert e["flops_per_token"] == pytest.approx(model.flops_per_token(64))


def test_the_meta_device_builds_big_models_without_memory() -> None:
    e = estimate(apply_preset(dict(default_config), "3b"))
    assert e["total"] > 3e9
    with torch.device("meta"):
        assert build_model({"n_head": 16, "n_embed": 2048, "context_length": 512,
                            "vocab_size": 50304, "n_blocks": 64}).lm_head.weight.is_meta

"""Typed configs: every shipped JSON file loads, and bad values fail early with a clear message."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from config.loader import coerce, load_config
from config.post_training_config import (
    ConfigError,
    DPOConfig,
    GRPOConfig,
    PPOConfig,
    PretrainConfig,
    RewardConfig,
    SFTConfig,
    smoke,
)
from src.post_training.cli import parse_config_with_json

STAGES = {
    "pretrain": PretrainConfig, "sft": SFTConfig, "reward": RewardConfig,
    "dpo": DPOConfig, "ppo": PPOConfig, "grpo": GRPOConfig,
}


@pytest.mark.parametrize("folder", ["configs", "configs/smoke"])
@pytest.mark.parametrize("name", list(STAGES))
def test_every_shipped_config_loads(folder: str, name: str) -> None:
    cfg = load_config(STAGES[name], f"{folder}/{name}.json")
    assert isinstance(cfg, STAGES[name])
    for value in vars(cfg).values():  # no machine-specific absolute paths in the defaults
        assert not (isinstance(value, str) and value.startswith("/ephemeral"))


def test_smoke_configs_never_overwrite_real_checkpoints() -> None:
    for name, cls in STAGES.items():
        assert load_config(cls, f"configs/smoke/{name}.json").out_ckpt.startswith("models/smoke/")


def _json(tmp_path: Path, data: dict[str, Any]) -> str:
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_unknown_keys_fail_with_a_suggestion(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="did you mean 'batch_size'"):
        load_config(SFTConfig, _json(tmp_path, {"batchsize": 4}))


def test_wrong_values_name_the_field_and_the_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"loss_type must be one of \["):
        load_config(DPOConfig, _json(tmp_path, {"loss_type": "dop"}))
    with pytest.raises(ConfigError, match="stage.json: lr must be a number"):
        load_config(SFTConfig, _json(tmp_path, {"lr": "fast"}))
    with pytest.raises(ConfigError, match="divisible"):
        load_config(SFTConfig, _json(tmp_path, {"n_embed": 30, "n_head": 4}))


def test_comment_keys_are_ignored(tmp_path: Path) -> None:
    cfg = load_config(SFTConfig, _json(tmp_path, {"_comment": "half the default lr", "lr": 5e-6}))
    assert cfg.lr == 5e-6


@pytest.mark.parametrize(
    ("value", "annotation", "expected"),
    [
        ("4", int, 4),
        (2000.0, int, 2000),
        ("50_000", int, 50_000),
        ("1e-5", float, 1e-5),
        (3, float, 3.0),
        ("true", bool, True),
        ("off", bool, False),
        ("none", int | None, None),
        ("8", int | None, 8),
        (None, str | None, None),
    ],
)
def test_coerce_accepts_sensible_spellings(value: Any, annotation: Any, expected: Any) -> None:
    assert coerce(value, annotation) == expected


@pytest.mark.parametrize(("value", "annotation"), [(True, int), (1, bool), ("1.5", int), ("abc", float), (3, str)])
def test_coerce_rejects_the_rest(value: Any, annotation: Any) -> None:
    with pytest.raises(ConfigError):
        coerce(value, annotation)


def test_environment_variables_are_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIG_DISK", "/mnt/big")
    cfg = load_config(SFTConfig, _json(tmp_path, {"out_ckpt": "$BIG_DISK/models/sft.pt"}))
    assert cfg.out_ckpt == "/mnt/big/models/sft.pt"


def test_smoke_shrinks_every_stage() -> None:
    for cls in STAGES.values():
        cfg = smoke(cls)
        assert cfg.context_length == 64 and cfg.device == "cpu"
    assert smoke(DPOConfig).max_len == 64


def test_cli_values_are_converted_like_json(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = ["train_sft.py", "--config", "configs/smoke/sft.json",
            "--lr", "2e-5", "--compile", "true", "--amp_dtype", "none", "--max_steps", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    cfg, _ = parse_config_with_json(SFTConfig, "configs/sft.json")
    assert (cfg.lr, cfg.compile, cfg.amp_dtype, cfg.max_steps) == (2e-5, True, None, 3)
    monkeypatch.setattr(sys, "argv", ["train_dpo.py", "--loss_type", "dop"])
    with pytest.raises(SystemExit):
        parse_config_with_json(DPOConfig, "configs/dpo.json")

"""
The laptop (CPU) student track end to end: tiny data -> train with a preset -> generate.

Uses the real scripts: prepare_tiny_data's encoder and writer, train_transformer's CLI, and
generate_text's checkpoint loader, on a few hundred tokens so it runs in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from config.presets import PRESETS
from scripts import generate_text, prepare_tiny_data, train_transformer
from src.tokenizer import BPETokenizer, tokenizer_spec

STORIES = [
    "Once upon a time there was a little cat. The cat liked to play with a red ball.",
    "One day a girl named Lily went to the park. She saw a big dog and smiled.",
    "Tom had a small boat. He took the boat to the lake and played all day.",
] * 30


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("tiny")
    tok = BPETokenizer.train(STORIES, vocab_size=300)
    spec = tokenizer_spec(tok)
    paths = {}
    for split, docs in (("train", STORIES), ("val", STORIES[:15])):
        path = root / f"{split}.h5"
        prepare_tiny_data.write_h5(path, prepare_tiny_data.encode_documents(tok, docs, split), spec, "test", len(docs))
        paths[split] = str(path)
    paths["root"] = str(root)
    return paths


def _train(monkeypatch: pytest.MonkeyPatch, tiny_data: dict[str, str], *extra: str) -> Path:
    out = Path(tiny_data["root"]) / f"model_{len(extra)}.pt"
    argv = ["train_transformer.py", "--preset", "tiny", "--device", "cpu", "--steps", "4", "--eval-every", "2",
            "--train-path", tiny_data["train"], "--dev-path", tiny_data["val"], "--out-path", str(out),
            "--seed", "0", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    train_transformer.main()
    return out


@pytest.mark.parametrize("arch", [[], ["--arch", "modern"]])
def test_train_then_generate(monkeypatch: pytest.MonkeyPatch, tiny_data: dict[str, str], arch: list[str]) -> None:
    out = _train(monkeypatch, tiny_data, *arch)
    checkpoint = torch.load(out, weights_only=False)
    cfg = checkpoint["config"]
    assert cfg["vocab_size"] == 320  # 300 BPE ids, padded to a multiple of 64
    assert cfg["tokenizer"]["type"] == "bpe" and cfg["t_context_length"] == cfg["context_length"]
    assert cfg.get("arch", "classic") == ("modern" if arch else "classic")

    text = generate_text.generate_text(str(out), "Once upon a time", max_new_tokens=8, device="cpu")
    assert text.startswith("Once upon a time") and len(text) > len("Once upon a time")


def test_missing_data_explains_how_to_make_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["train_transformer.py", "--preset", "tiny", "--train-path", str(tmp_path / "nope.h5")])
    with pytest.raises(SystemExit, match="prepare_tiny_data"):
        train_transformer.main()


def test_presets_are_consistent() -> None:
    for name, preset in PRESETS.items():
        assert preset["n_embed"] % preset["n_head"] == 0, name
        assert preset["t_lr_decay_step"] < preset["t_train_steps"], name
        if name != "3b":  # the original config trains on 16-token windows; the presets do not
            assert preset["t_context_length"] == preset["context_length"], name


def test_round_up_pads_to_64() -> None:
    assert [train_transformer.round_up(v) for v in (1, 64, 300, 4096, 50257)] == [64, 64, 320, 4096, 50304]

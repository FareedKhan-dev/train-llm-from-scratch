"""Device selection: "auto" picks what exists, a missing backend falls back to the CPU."""

from __future__ import annotations

import contextlib

import pytest
import torch

from src.device import autocast, describe_device, device_type, resolve_device


def test_auto_returns_a_usable_device() -> None:
    device = resolve_device("auto")
    assert device in ("cuda", "mps", "cpu")
    torch.zeros(2, device=device)  # actually usable


def test_missing_backend_falls_back_to_cpu_with_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="CUDA"):
        assert resolve_device("cuda") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu-please")


def test_autocast_is_a_no_op_on_cpu() -> None:
    assert isinstance(autocast("cpu", "bf16"), contextlib.nullcontext)
    assert isinstance(autocast("cuda", None), contextlib.nullcontext)
    assert device_type("cuda:1") == "cuda"
    assert "PyTorch" in describe_device("cpu")

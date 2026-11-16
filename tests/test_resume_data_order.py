"""
The data order is reproducible, a resumed run continues it, and DDP ranks never overlap.

Before 1.0.0 a resumed run re-seeded numpy and opened a fresh loader, so it trained on the
batches of steps 0, 1, 2... a second time and never reached the data it had not seen yet.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch

import data_loader.data_loader as dl
from data_loader.data_loader import get_batch_iterator

REPO = Path(__file__).resolve().parents[1]
CTX = 8


def _windows_file(path: Path, n_windows: int, ctx: int = CTX) -> Path:
    """An h5 whose window i starts with token i, so a batch's first column names its windows."""
    tokens = np.random.RandomState(1).randint(1000, 4000, size=n_windows * ctx + 1).astype(np.int32)
    tokens[np.arange(n_windows) * ctx] = np.arange(n_windows)
    with h5py.File(path, "w") as f:
        f.create_dataset("tokens", data=tokens)
    return path


def _ids(it: Any, n: int) -> list[list[int]]:
    return [next(it)[0][:, 0].tolist() for _ in range(n)]


def test_a_seed_fixes_the_order_whatever_else_uses_numpy(tmp_path: Path) -> None:
    path = _windows_file(tmp_path / "d.h5", 20)
    with contextlib.redirect_stdout(io.StringIO()):
        a = _ids(get_batch_iterator(str(path), 3, CTX, seed=7), 12)
        it = get_batch_iterator(str(path), 3, CTX, seed=7)
        b = []
        for _ in range(12):
            np.random.shuffle(np.arange(50))  # an eval loader or anything else drawing from numpy
            b += _ids(it, 1)
    assert a == b


def test_skipping_continues_the_order_across_epochs(tmp_path: Path) -> None:
    path = _windows_file(tmp_path / "d.h5", 10)  # 3 batches of 3 per epoch, so 12 batches is 4 epochs
    with contextlib.redirect_stdout(io.StringIO()):
        full = _ids(get_batch_iterator(str(path), 3, CTX, seed=3), 12)
        for k in (1, 3, 5, 9):
            assert _ids(get_batch_iterator(str(path), 3, CTX, seed=3, skip_batches=k), 12 - k) == full[k:]


def test_ranks_read_disjoint_slices_of_one_shuffle(tmp_path: Path) -> None:
    path = _windows_file(tmp_path / "d.h5", 24)
    with contextlib.redirect_stdout(io.StringIO()):
        r0 = _ids(get_batch_iterator(str(path), 3, CTX, seed=5, rank=0, world_size=2), 4)
        r1 = _ids(get_batch_iterator(str(path), 3, CTX, seed=5, rank=1, world_size=2), 4)
        single = _ids(get_batch_iterator(str(path), 6, CTX, seed=5), 4)
    seen = [w for batch in r0 + r1 for w in batch]
    assert len(seen) == len(set(seen)) == 24  # one epoch, every window exactly once
    assert [a + b for a, b in zip(r0, r1)] == single  # the same global batches as one process


def test_bad_rank_or_skip_is_an_error(tmp_path: Path) -> None:
    path = _windows_file(tmp_path / "d.h5", 10)
    with pytest.raises(ValueError, match="rank"):
        next(get_batch_iterator(str(path), 2, CTX, rank=2, world_size=2))
    with pytest.raises(ValueError, match="skip_batches"):
        next(get_batch_iterator(str(path), 2, CTX, skip_batches=-1))


def _record_training_batches(monkeypatch: pytest.MonkeyPatch, module: Any, attr: str) -> list[list[int]]:
    """Record the first column of every batch the training loader yields (the one main() reads)."""
    seen: list[list[int]] = []
    original = dl.get_batch_iterator

    def recording(*args: Any, **kwargs: Any) -> Any:
        it = original(*args, **kwargs)
        if sys._getframe(1).f_code.co_name != "main":  # an eval loader, read by estimate_loss
            yield from it
            return
        for xb, yb in it:
            seen.append(xb[:, 0].tolist())
            yield xb, yb

    monkeypatch.setattr(module, attr, recording)
    return seen


def test_train_transformer_resume_continues_the_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import train_transformer
    from src.tokenizer import BPETokenizer, tokenizer_spec

    ctx = 128  # the tiny preset's window
    spec = tokenizer_spec(BPETokenizer.train(["Once upon a time there was a cat."] * 5, vocab_size=260))
    for name in ("train", "val"):
        path = _windows_file(tmp_path / f"{name}.h5", 64, ctx)
        with h5py.File(path, "a") as f:
            f["tokens"].attrs["tokenizer"] = json.dumps(spec)
            f["tokens"].attrs["vocab_size"] = 4096
    seen = _record_training_batches(monkeypatch, dl, "get_batch_iterator")
    ckdir = tmp_path / "ck"

    def run(*extra: str) -> list[list[int]]:
        seen.clear()
        argv = ["train_transformer.py", "--preset", "tiny", "--device", "cpu", "--steps", "6",
                "--batch-size", "4", "--eval-every", "1000",
                "--train-path", str(tmp_path / "train.h5"), "--dev-path", str(tmp_path / "val.h5"),
                "--out-path", str(tmp_path / "m.pt"), "--checkpoint-every", "3",
                "--checkpoint-dir", str(ckdir), "--keep-last", "0", *extra]
        monkeypatch.setattr(sys, "argv", argv)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train_transformer.main()
        return list(seen)

    full = run()  # no --seed: the drawn data seed is still saved in the checkpoint
    assert len(full) == 6
    first = sorted(ckdir.iterdir())[0]  # after steps 0..2
    assert run("--resume", str(first)) == full[3:6]


def test_pretrain_base_resume_continues_and_extends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import pretrain_base

    for name in ("train", "dev"):
        _windows_file(tmp_path / f"{name}.h5", 64, 32)
    ck = tmp_path / "base.pt"
    seen = _record_training_batches(monkeypatch, pretrain_base, "get_batch_iterator")
    monkeypatch.chdir(REPO)

    def run(steps: int, *extra: str) -> tuple[list[list[int]], str]:
        seen.clear()
        argv = ["pretrain_base.py", "--config", "configs/smoke/pretrain.json", "--device", "cpu",
                "--n_embed", "32", "--n_head", "2", "--n_blocks", "1", "--context_length", "32",
                "--vocab_size", "4096", "--batch_size", "4", "--grad_accum", "1", "--train_steps", str(steps),
                "--warmup_steps", "1", "--save_every", "100000", "--eval_steps", "100000",
                "--train_path", str(tmp_path / "train.h5"), "--dev_path", str(tmp_path / "dev.h5"),
                "--out_ckpt", str(ck), "--log_dir", str(tmp_path / "logs"), *extra]
        monkeypatch.setattr(sys, "argv", argv)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pretrain_base.main()
        return list(seen), out.getvalue()

    full, _ = run(8)
    assert len(full) == 8
    run(4)
    assert torch.load(ck, map_location="cpu", weights_only=False)["step"] == 3  # the last completed step
    resumed, _ = run(8, "--resume", str(ck))
    assert resumed == full[4:8]  # steps 4..7 run, on the batches an uninterrupted run would use
    assert torch.load(ck, map_location="cpu", weights_only=False)["step"] == 7
    batches, out = run(8, "--resume", str(ck))  # nothing left to do: says so instead of crashing
    assert batches == [] and "Nothing to train" in out

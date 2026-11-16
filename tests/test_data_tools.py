"""The pretraining data tools stop at the token budget and never keep a broken download."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import pytest
import requests
import zstandard as zstd

from scripts import data_download
from scripts import prepare_pretrain_data as ppd


class _TenTokens:
    """Stands in for tiktoken (no download): every document is 10 tokens."""

    def encode_ordinary_batch(self, docs: list[str]) -> list[list[int]]:
        return [[1] * 10 for _ in docs]


def _shard(path: Path, n_docs: int) -> Path:
    with open(path, "wb") as fh, zstd.ZstdCompressor().stream_writer(fh) as w:
        for i in range(n_docs):
            w.write((json.dumps({"text": f"doc {i}"}) + "\n").encode())
    return path


@pytest.mark.parametrize("max_tokens", [1, 1_000, 12_345, 54_999])
def test_max_tokens_is_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_tokens: int) -> None:
    monkeypatch.setattr(ppd.tiktoken, "get_encoding", lambda name: _TenTokens())
    monkeypatch.setattr(ppd, "WRITE_CHUNK", 50_000)  # scaled down from 8M so the test is small
    shard = _shard(tmp_path / "00.jsonl.zst", 10_000)  # 110,000 tokens with the end-of-text ids
    out = tmp_path / "out.h5"
    assert ppd.tokenize_to_h5([str(shard)], str(out), max_tokens=max_tokens) == max_tokens
    with h5py.File(out) as f:
        assert f["tokens"].shape[0] == max_tokens  # it used to write at least one 8M-token chunk


def test_no_budget_keeps_every_token_and_a_bad_budget_is_an_error(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ppd.tiktoken, "get_encoding", lambda name: _TenTokens())
    monkeypatch.setattr(ppd, "WRITE_CHUNK", 50_000)
    shard = _shard(tmp_path / "00.jsonl.zst", 3_000)
    assert ppd.tokenize_to_h5([str(shard), str(shard)], str(tmp_path / "out.h5"), max_tokens=None) == 66_000
    with pytest.raises(ValueError, match="positive"):
        ppd.tokenize_to_h5([str(shard)], str(tmp_path / "out.h5"), max_tokens=0)


class _Response:
    def __init__(self, body: bytes, status: int = 200, length: int | None = None) -> None:
        self.body, self.status_code = body, status
        self.headers = {"content-length": str(len(body) if length is None else length)}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def iter_content(self, size: int) -> Any:
        for i in range(0, len(self.body), size):
            yield self.body[i:i + size]

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_a_failed_or_short_download_leaves_nothing_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "00.jsonl.zst"
    monkeypatch.setattr(data_download.requests, "get", lambda *a, **k: _Response(b"<html>503</html>", 503))
    with pytest.raises(requests.HTTPError):
        data_download.download_file("https://example.invalid/00.jsonl.zst", str(dest))
    monkeypatch.setattr(data_download.requests, "get", lambda *a, **k: _Response(b"x" * 100, length=5_000))
    with pytest.raises(OSError, match="100 of 5,000"):
        data_download.download_file("https://example.invalid/00.jsonl.zst", str(dest))
    assert list(tmp_path.iterdir()) == []  # so the next run downloads again instead of skipping

    monkeypatch.setattr(data_download.requests, "get", lambda *a, **k: _Response(b"y" * 3_000_000))
    data_download.download_file("https://example.invalid/00.jsonl.zst", str(dest))
    assert dest.read_bytes() == b"y" * 3_000_000 and list(tmp_path.iterdir()) == [dest]

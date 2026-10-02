"""
Tokenizers: OpenAI's tiktoken encodings, or a byte-level BPE trained from scratch.

Every tokenizer here is described by a small JSON-friendly dict, so it can travel inside an
HDF5 data file and inside a checkpoint. That keeps a trained model self-contained: the
generation script reads the tokenizer from the checkpoint instead of guessing.

    {"type": "tiktoken", "name": "r50k_base"}          # GPT-2/3 tokenizer, the repo default
    {"type": "bpe", "merges": [...], ...}                # one you trained with BPETokenizer
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Protocol

import tiktoken

from src.tokenizer.bpe import EOT, BPETokenizer

DEFAULT_TOKENIZER = {"type": "tiktoken", "name": "r50k_base"}


class Tokenizer(Protocol):
    """The tokenizer interface the scripts rely on (tiktoken.Encoding and BPETokenizer both fit)."""

    @property
    def n_vocab(self) -> int: ...

    @property
    def eot_token(self) -> int: ...

    def encode_ordinary(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


@lru_cache(maxsize=8)
def _tiktoken(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def tokenizer_from_spec(spec: dict[str, Any] | str | None) -> Any:
    """Build a tokenizer from a spec dict (or its JSON string, or a tiktoken name)."""
    if spec is None:
        spec = DEFAULT_TOKENIZER
    if isinstance(spec, str):
        spec = json.loads(spec) if spec.lstrip().startswith("{") else {"type": "tiktoken", "name": spec}
    if spec.get("type") == "tiktoken":
        return _tiktoken(str(spec.get("name", "r50k_base")))
    if spec.get("type") == "bpe":
        return BPETokenizer.from_dict(spec)
    raise ValueError(f"unknown tokenizer spec: {spec!r}")


def tokenizer_spec(tokenizer: Any) -> dict[str, Any]:
    """The JSON-friendly description of ``tokenizer`` (the inverse of :func:`tokenizer_from_spec`)."""
    if isinstance(tokenizer, BPETokenizer):
        return tokenizer.to_dict()
    if isinstance(tokenizer, tiktoken.Encoding):
        return {"type": "tiktoken", "name": tokenizer.name}
    raise TypeError(f"cannot describe tokenizer of type {type(tokenizer).__name__}")


def safe_decode(tokenizer: Any, ids: list[int]) -> str:
    """Decode while skipping ids the tokenizer does not know.

    Models pad their vocabulary (50257 -> 50304 for r50k_base) so the embedding table is a
    multiple of 64. An under-trained model can sample one of those padding ids, and tiktoken
    raises on them. Dropping them keeps generation from crashing.
    """
    return tokenizer.decode([int(t) for t in ids if 0 <= int(t) < tokenizer.n_vocab])


__all__ = [
    "BPETokenizer",
    "DEFAULT_TOKENIZER",
    "EOT",
    "Tokenizer",
    "safe_decode",
    "tokenizer_from_spec",
    "tokenizer_spec",
]

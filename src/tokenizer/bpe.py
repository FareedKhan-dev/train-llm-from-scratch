"""
A byte-level BPE tokenizer, written from scratch.

This is the same algorithm behind GPT-2, GPT-4 and most modern LLM tokenizers:

1. Split text into chunks with a regex (words, numbers, punctuation, spaces), so a merge
   never glues the end of one word to the start of the next.
2. Write every chunk as its UTF-8 bytes. The starting vocabulary is the 256 byte values,
   so any text at all can be encoded and nothing is ever "unknown".
3. Training repeatedly finds the most frequent adjacent pair of ids and merges it into a new
   id. ``vocab_size - 256`` merges later, the frequent words and word pieces have their
   own ids.
4. Encoding replays the merges in the order they were learned; decoding concatenates the
   bytes of every id.

The class mirrors the parts of ``tiktoken.Encoding`` this repo uses (``encode_ordinary``,
``encode``, ``decode``, ``n_vocab``, ``eot_token``), so the data scripts and the generation
script accept either one.

Training uses the standard incremental algorithm: pair counts are kept up to date as words
change, and a heap finds the most frequent pair, so a 4k vocabulary trains on tens of MB of
text in seconds rather than hours.
"""

from __future__ import annotations

import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

import regex

# The split patterns used by OpenAI's r50k_base (GPT-2/3) and cl100k_base (GPT-4) tokenizers.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)
EOT = "<|endoftext|>"

Pair = tuple[int, int]


def _merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE. Build one with :meth:`train` or :meth:`load`."""

    def __init__(
        self,
        merges: Sequence[Pair],
        special_tokens: Sequence[str] = (EOT,),
        pattern: str = GPT4_SPLIT_PATTERN,
        name: str = "bpe",
    ) -> None:
        self.name = name
        self.pattern = pattern
        self._split = regex.compile(pattern)
        self.merges: list[Pair] = [(int(a), int(b)) for a, b in merges]
        self._ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(self.merges)}

        self._bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])
        first_special = 256 + len(self.merges)
        self.special_tokens: dict[str, int] = {
            tok: first_special + i for i, tok in enumerate(special_tokens)
        }
        self._special_by_id = {i: tok for tok, i in self.special_tokens.items()}
        self._special_split = (
            regex.compile("(" + "|".join(regex.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ properties
    @property
    def n_vocab(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_token(self) -> int:
        if EOT not in self.special_tokens:
            raise AttributeError("this tokenizer has no <|endoftext|> token")
        return self.special_tokens[EOT]

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: str | Iterable[str],
        vocab_size: int,
        *,
        special_tokens: Sequence[str] = (EOT,),
        pattern: str = GPT4_SPLIT_PATTERN,
        verbose: bool = False,
    ) -> BPETokenizer:
        """Learn ``vocab_size - 256 - len(special_tokens)`` merges from ``texts``."""
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")
        split = regex.compile(pattern)
        special_re = (
            regex.compile("|".join(regex.escape(t) for t in special_tokens)) if special_tokens else None
        )

        # 1. Count every regex chunk once. Special tokens are cut out first so they never merge.
        counts: Counter[str] = Counter()
        for text in [texts] if isinstance(texts, str) else texts:
            for part in special_re.split(text) if special_re else [text]:
                counts.update(split.findall(part))
        words = [list(chunk.encode("utf-8")) for chunk in counts]
        freqs = list(counts.values())

        # 2. Pair counts (weighted by how often each chunk occurs) and which chunks hold a pair.
        pair_counts: defaultdict[Pair, int] = defaultdict(int)
        where: defaultdict[Pair, set[int]] = defaultdict(set)
        for wi, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        # 3. Merge the most frequent pair, update only the chunks that contained it, repeat.
        merges: list[Pair] = []
        while len(merges) < n_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current <= 0:
                continue
            if -neg_count != current:  # stale heap entry, put it back with the real count
                heapq.heappush(heap, (-current, pair))
                continue
            new_id = 256 + len(merges)
            merges.append(pair)
            changed: set[Pair] = set()
            for wi in where.pop(pair, ()):
                old = words[wi]
                new = _merge(old, pair, new_id)
                if len(new) == len(old):
                    continue
                f = freqs[wi]
                for p in zip(old, old[1:]):
                    pair_counts[p] -= f
                for p in zip(new, new[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    changed.add(p)
                words[wi] = new
            pair_counts.pop(pair, None)
            for p in changed:
                if pair_counts.get(p, 0) > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
            if verbose and (len(merges) % 500 == 0 or len(merges) == n_merges):
                print(f"  merge {len(merges)}/{n_merges}: {pair} -> {new_id} ({current:,} times)")

        return cls(merges, special_tokens=special_tokens, pattern=pattern)

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge present in the chunk, exactly as training did.
            best = min(zip(ids, ids[1:]), key=lambda p: self._ranks.get(p, len(self._ranks)))
            rank = self._ranks.get(best)
            if rank is None:
                break
            ids = _merge(ids, best, 256 + rank)
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text with no special-token handling (``<|endoftext|>`` is just characters)."""
        out: list[int] = []
        for chunk in self._split.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allowed_special: Literal["all"] | set[str] = "all") -> list[int]:
        """Encode text; special tokens in ``allowed_special`` become their single id."""
        allowed = set(self.special_tokens) if allowed_special == "all" else set(allowed_special)
        if not allowed or self._special_split is None:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_split.split(text):
            if part in allowed:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """Turn ids back into text. Bytes that are not valid UTF-8 on their own become U+FFFD."""
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if i in self._special_by_id:
                parts.append(self._special_by_id[i].encode("utf-8"))
            elif 0 <= i < len(self._bytes):
                parts.append(self._bytes[i])
            # ids beyond the vocabulary (padding rows of the embedding table) are skipped
        return b"".join(parts).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        """The raw bytes behind one id (handy for inspecting what was learned)."""
        return self._bytes[token_id]

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, object]:
        return {
            "type": "bpe",
            "version": 1,
            "pattern": self.pattern,
            "merges": [list(p) for p in self.merges],
            "special_tokens": list(self.special_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BPETokenizer:
        if data.get("type") != "bpe":
            raise ValueError("not a BPE tokenizer description")
        merges = [(int(a), int(b)) for a, b in data["merges"]]  # type: ignore[union-attr]
        specials = [str(t) for t in data.get("special_tokens", [EOT])]  # type: ignore[union-attr]
        return cls(merges, special_tokens=specials, pattern=str(data["pattern"]))

    def save(self, path: str | os.PathLike[str]) -> None:
        os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BPETokenizer:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

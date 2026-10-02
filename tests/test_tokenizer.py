"""The byte-level BPE tokenizer written from scratch, and the tokenizer specs."""

from __future__ import annotations

from collections import Counter

import pytest

from src.tokenizer import BPETokenizer, safe_decode, tokenizer_from_spec, tokenizer_spec

CORPUS = (
    "the cat sat on the mat. the cat ate the rat. a dog saw the cat and the rat. " * 40
    + "Once upon a time, there was a little girl named Lily. She liked to play outside. " * 40
)


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=400)


@pytest.mark.parametrize(
    "text",
    ["the cat sat", "Hello, World!  Spaces   and\nnew lines\n", "naïve café, 日本語, emoji 🙂", "", "zxq 123456789"],
)
def test_round_trip_is_lossless(tok: BPETokenizer, text: str) -> None:
    assert tok.decode(tok.encode_ordinary(text)) == text


def test_training_learns_frequent_pieces_first(tok: BPETokenizer) -> None:
    # The first merge must be the most frequent adjacent byte pair inside the regex chunks.
    import regex

    pairs: Counter[tuple[int, int]] = Counter()
    for chunk in regex.findall(tok.pattern, CORPUS):
        b = chunk.encode("utf-8")
        pairs.update(zip(b, b[1:]))
    assert tok.merges[0] == pairs.most_common(1)[0][0]
    assert len(tok.encode_ordinary(CORPUS)) < len(CORPUS.encode("utf-8")) / 3  # real compression
    assert tok.n_vocab <= 400


def test_training_is_deterministic() -> None:
    a = BPETokenizer.train(CORPUS, vocab_size=320)
    b = BPETokenizer.train(CORPUS, vocab_size=320)
    assert a.merges == b.merges


def _textbook_bpe(text: str, n_merges: int, pattern: str) -> list[tuple[int, int]]:
    """The slow, obvious algorithm: recount every pair in every chunk after every merge."""
    import regex

    from src.tokenizer.bpe import _merge

    chunks = [list(c.encode("utf-8")) for c in regex.findall(pattern, text)]
    merges: list[tuple[int, int]] = []
    for k in range(n_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for ids in chunks:
            counts.update(zip(ids, ids[1:]))
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))  # most frequent; ties go to the smaller pair
        merges.append(best)
        chunks = [_merge(ids, best, 256 + k) for ids in chunks]
    return merges


def test_fast_training_learns_the_same_merges_as_the_textbook_algorithm() -> None:
    text = CORPUS + "naïve café, 日本語 and emoji 🙂🙂 " * 5 + "aaabdaaabac " * 7
    fast = BPETokenizer.train(text, vocab_size=256 + 150 + 1)
    assert fast.merges == _textbook_bpe(text, 150, fast.pattern)


def test_special_tokens(tok: BPETokenizer) -> None:
    ids = tok.encode("story one<|endoftext|>story two")
    assert ids.count(tok.eot_token) == 1
    assert tok.decode(ids) == "story one<|endoftext|>story two"
    assert tok.eot_token not in tok.encode_ordinary("story one<|endoftext|>story two")


def test_specs_round_trip_and_safe_decode(tok: BPETokenizer, tmp_path) -> None:
    again = tokenizer_from_spec(tokenizer_spec(tok))
    assert again.encode_ordinary(CORPUS[:500]) == tok.encode_ordinary(CORPUS[:500])
    tok.save(tmp_path / "tok.json")
    assert BPETokenizer.load(tmp_path / "tok.json").merges == tok.merges

    r50k = tokenizer_from_spec(None)  # the repo default
    assert tokenizer_spec(r50k) == {"type": "tiktoken", "name": "r50k_base"}
    ids = r50k.encode_ordinary("The cat")
    assert safe_decode(r50k, ids + [50300]) == "The cat"  # padding ids are skipped, not a crash


def test_vocab_must_hold_the_bytes() -> None:
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", vocab_size=100)

"""
The benchmark scorer's arithmetic, on a toy model whose probabilities are known exactly.

The scorer was also checked against lm-evaluation-harness itself on GPT-2 124M (the numbers are
in the README); these tests keep the rules that check depends on from drifting.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.benchmarks import TASKS, Example, Request, encode_pair, evaluate_task, score_requests

V = 8


def encode(text: str) -> list[int]:
    """Characters as tokens, except that "ab" merges into one token (id 7)."""
    out: list[int] = []
    i = 0
    while i < len(text):
        if text[i:i + 2] == "ab":
            out.append(7)
            i += 2
        else:
            out.append(ord(text[i]) % 7)
            i += 1
    return out


def bigram_logits(ids: torch.Tensor) -> torch.Tensor:
    """A fixed bigram model: p(next = j | current = i) is proportional to exp(table[i, j])."""
    table = torch.arange(V * V, dtype=torch.float32).reshape(V, V).sin() * 3
    return table[ids]


def bigram_logp(prev: int, nxt: int) -> float:
    table = torch.arange(V * V, dtype=torch.float32).reshape(V, V).sin() * 3
    return float(torch.log_softmax(table[prev], -1)[nxt])


def test_encode_pair_moves_trailing_spaces_and_scores_boundary_merges() -> None:
    assert encode_pair(encode, "x ", "y") == encode_pair(encode, "x", " y")
    # "abc" -> [ab, c]; "a" alone is one token, so the merged token counts as context
    assert encode_pair(encode, "a", "bc") == Request(context=(7,), continuation=(ord("c") % 7,))
    with pytest.raises(ValueError):
        encode_pair(encode, "abc", "")


def test_loglik_is_the_sum_over_the_continuation_tokens() -> None:
    req = Request((1, 2), (3, 4, 5))
    [s] = score_requests(bigram_logits, [req], max_length=16)
    assert math.isclose(s.loglik, bigram_logp(2, 3) + bigram_logp(3, 4) + bigram_logp(4, 5), rel_tol=1e-6)
    greedy = [int(torch.argmax(bigram_logits(torch.tensor([t]))[0])) for t in (2,)]
    [g] = score_requests(bigram_logits, [Request((1, 2), (greedy[0],))], max_length=16)
    assert g.greedy and not s.greedy


def test_batching_and_padding_change_nothing() -> None:
    reqs = [Request(tuple(range(1, 2 + k % 5)), (k % V, (k + 3) % V)) for k in range(13)]
    one = score_requests(bigram_logits, reqs, max_length=16, batch_size=1)
    many = score_requests(bigram_logits, reqs, max_length=16, batch_size=8)
    assert [s.loglik for s in one] == pytest.approx([s.loglik for s in many], rel=1e-6)


def test_an_empty_context_is_refused() -> None:
    with pytest.raises(ValueError, match="context token"):
        score_requests(bigram_logits, [Request((), (1, 2))], max_length=8)


def test_long_inputs_keep_their_last_tokens() -> None:
    long = Request(tuple([0] * 50 + [1, 2]), (3,))
    short = Request((2,), (3,))  # a bigram only sees the previous token, so these must agree
    a, b = score_requests(bigram_logits, [long, short], max_length=4)
    assert math.isclose(a.loglik, b.loglik, rel_tol=1e-6)


def test_choice_tasks_report_acc_and_length_normalized_acc_norm() -> None:
    # Per character, the long answer is likelier; in total, the short one is.
    logp = {"a": -0.1, "b": -3.0}
    ex = Example("q", (" a", " bbbb"), label=1)

    def logits(ids: torch.Tensor) -> torch.Tensor:  # p(token) independent of context
        out = torch.full((*ids.shape, V), -50.0)
        out[..., ord("a") % 7] = 0.0
        out[..., ord("b") % 7] = math.log(math.exp(logp["b"]) / math.exp(logp["a"]))
        out[..., ord(" ") % 7] = 0.0
        return out

    r = evaluate_task(TASKS["arc_easy"], [ex], logits, encode, max_length=32)
    assert r["acc"] == 0.0 and r["n"] == 1
    total = [s.loglik for s in score_requests(logits, [encode_pair(encode, "q", c) for c in ex.choices],
                                              max_length=32)]
    assert (total[1] / 4 > total[0] / 1) == (r["acc_norm"] == 1.0)


def test_lambada_scores_greedy_accuracy_and_perplexity() -> None:
    greedy_next = int(torch.argmax(bigram_logits(torch.tensor([3]))[0]))
    ex_right = Example("x" * 2 + chr(3 + 7 * 10), (chr(greedy_next + 7 * 10),), 0)
    r = evaluate_task(TASKS["lambada"], [ex_right], bigram_logits, encode, max_length=16)
    assert r["acc"] == 1.0
    assert math.isclose(r["perplexity"], math.exp(-bigram_logp(3, greedy_next)), rel_tol=1e-5)


def test_the_tasks_are_pinned_and_hellaswag_is_not_downloaded() -> None:
    assert set(TASKS) == {"arc_easy", "arc_challenge", "lambada"}
    assert all(len(t.revision) == 40 for t in TASKS.values())

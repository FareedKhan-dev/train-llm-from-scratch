"""
Log-likelihood scoring, the way base models are compared.

A base model cannot answer a question in a chat format yet, but it can tell how likely a
piece of text is. So a benchmark gives it a context and a few candidate continuations and
asks which one the model finds most likely:

    ll(continuation | context) = sum over the continuation's tokens of log p(token | everything before it)

The rules below follow EleutherAI's lm-evaluation-harness exactly, so a number from this
module means the same thing as the numbers people publish: trailing spaces of the context
move to the continuation, the continuation's tokens are those of the full string after as
many tokens as the context alone encodes to, and an input longer than the model's window
keeps its last tokens. ``tests/test_benchmarks.py`` checks the arithmetic, and the README records the
check against the harness itself on GPT-2.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

#: ``logits_fn(ids)`` maps a ``(batch, seq)`` tensor of token ids to ``(batch, seq, vocab)`` logits.
LogitsFn = Callable[[torch.Tensor], torch.Tensor]
Encode = Callable[[str], list[int]]


@dataclass(frozen=True)
class Request:
    """One continuation to score: its context tokens and its own tokens."""

    context: tuple[int, ...]
    continuation: tuple[int, ...]


@dataclass(frozen=True)
class Score:
    """``loglik`` is the summed log-probability of the continuation; ``greedy`` says whether
    greedy decoding from the context would have produced exactly the continuation."""

    loglik: float
    greedy: bool


def encode_pair(encode: Encode, context: str, continuation: str) -> Request:
    """Tokenize a (context, continuation) pair the way lm-evaluation-harness does.

    Trailing whitespace moves from the context to the continuation, and the continuation's
    tokens are those of ``context + continuation`` after the first ``len(encode(context))`` of
    them. A token that merges across the boundary therefore counts as context.
    """
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = encode(context + continuation)
    n_ctx = len(encode(context))
    if not whole[n_ctx:]:
        raise ValueError(f"the continuation {continuation!r} encodes to no tokens")
    return Request(tuple(whole[:n_ctx]), tuple(whole[n_ctx:]))


@torch.no_grad()
def score_requests(
    logits_fn: LogitsFn,
    requests: Sequence[Request],
    *,
    max_length: int,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
) -> list[Score]:
    """Score every request, ``batch_size`` at a time, inside a window of ``max_length`` tokens.

    Each input is the last ``max_length + 1`` tokens of context plus continuation, minus the
    final one (the model predicts every token from the ones before it). Rows are padded on the
    right, which a causal model never looks at.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}")
    for r in requests:
        if not r.context:
            # Nothing would predict the first token. The harness prepends end-of-text here;
            # no task in this repo has an empty context, so refuse instead of guessing.
            raise ValueError("a request needs at least one context token")
    scores: list[Score] = []
    for start in range(0, len(requests), batch_size):
        chunk = requests[start:start + batch_size]
        rows = [list(r.context + r.continuation)[-(max_length + 1):][:-1] for r in chunk]
        width = max(len(row) for row in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        logp = F.log_softmax(logits_fn(ids.to(device)).float(), dim=-1).cpu()
        for i, (row, req) in enumerate(zip(rows, chunk)):
            n = len(req.continuation)
            positions = torch.arange(len(row) - n, len(row))
            target = torch.tensor(req.continuation, dtype=torch.long)
            picked = logp[i, positions].gather(-1, target[:, None]).squeeze(-1)
            greedy = bool((logp[i, positions].argmax(-1) == target).all())
            scores.append(Score(float(picked.sum()), greedy))
    return scores

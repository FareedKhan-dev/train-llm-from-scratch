"""
The base-model benchmarks: ARC-Easy, ARC-Challenge and LAMBADA.

Each task is the same as in EleutherAI's lm-evaluation-harness (same prompt, same split,
same metrics), so the numbers compare with published ones:

* **ARC** (AI2 Reasoning Challenge, CC BY-SA 4.0): grade-school science questions with 3 to 5
  answers. The model scores ``"Question: <q>\\nAnswer:"`` followed by each answer; ``acc``
  picks the most likely answer, ``acc_norm`` the most likely per character, which removes the
  bias toward short answers. Random guessing is about 25%.
* **LAMBADA** (the OpenAI version GPT-2 was evaluated on, MIT): passages whose last word needs
  the whole passage to guess. ``acc`` is how often greedy decoding produces exactly that word,
  ``perplexity`` the exponential of its mean negative log-likelihood.

HellaSwag, the usual third, is left out on purpose: its source repository was taken down by a
DMCA notice in September 2026 (part of its text is scraped from wikiHow), so this repo does
not download it.

The datasets are pinned to a revision so a number measured today reproduces later.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.benchmarks.scoring import Encode, LogitsFn, Request, encode_pair, score_requests


@dataclass(frozen=True)
class Example:
    """A context, the continuations to score after it, and the index of the right one."""

    context: str
    choices: tuple[str, ...]
    label: int


@dataclass(frozen=True)
class Task:
    name: str
    kind: str  # "choice" (pick the likeliest continuation) or "lambada" (greedy last word)
    hf_path: str
    hf_name: str | None
    split: str
    revision: str
    to_example: Callable[[dict[str, Any]], Example]


def _arc(doc: dict[str, Any]) -> Example:
    labels = list(doc["choices"]["label"])
    texts = tuple(doc["choices"]["text"])
    return Example(f"Question: {doc['question']}\nAnswer:", tuple(" " + t for t in texts),
                   labels.index(doc["answerKey"]))


def _lambada(doc: dict[str, Any]) -> Example:
    words = doc["text"].split(" ")
    return Example(" ".join(words[:-1]), (" " + words[-1],), 0)


_ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
_LAMBADA_REVISION = "900124bf3b8235c6daf21033af9948b3f07346c4"

TASKS: dict[str, Task] = {
    "arc_easy": Task("arc_easy", "choice", "allenai/ai2_arc", "ARC-Easy", "test", _ARC_REVISION, _arc),
    "arc_challenge": Task("arc_challenge", "choice", "allenai/ai2_arc", "ARC-Challenge", "test",
                          _ARC_REVISION, _arc),
    "lambada": Task("lambada", "lambada", "EleutherAI/lambada_openai", "default", "test",
                    _LAMBADA_REVISION, _lambada),
}


def load_examples(task: Task, limit: int | None = None) -> list[Example]:
    """Download (once, into the Hugging Face cache) and convert a task's examples."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the install
        raise SystemExit('The benchmarks need the datasets package: pip install -e ".[train]"') from e
    rows = load_dataset(task.hf_path, task.hf_name, split=task.split, revision=task.revision)
    n = len(rows) if limit is None else min(limit, len(rows))
    return [task.to_example(dict(rows[i])) for i in range(n)]


def _stderr(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / max(n - 1, 1))


def evaluate_task(
    task: Task,
    examples: list[Example],
    logits_fn: LogitsFn,
    encode: Encode,
    *,
    max_length: int,
    device: str = "cpu",
    batch_size: int = 8,
) -> dict[str, float]:
    """Score ``examples`` and return the task's metrics, each with its standard error.

    ``choice`` tasks report ``acc`` and ``acc_norm``; ``lambada`` reports ``acc`` and
    ``perplexity``. ``n`` is the number of examples.
    """
    requests: list[Request] = []
    owner: list[int] = []
    for i, ex in enumerate(examples):
        for choice in ex.choices:
            requests.append(encode_pair(encode, ex.context, choice))
            owner.append(i)
    scores = score_requests(logits_fn, requests, max_length=max_length, device=device, batch_size=batch_size)
    per_example: list[list[float]] = [[] for _ in examples]
    greedy: list[bool] = [False] * len(examples)
    for i, s in zip(owner, scores):
        per_example[i].append(s.loglik)
        greedy[i] = s.greedy
    n = len(examples)
    if task.kind == "lambada":
        acc = sum(greedy) / n
        mean_ll = sum(lls[0] for lls in per_example) / n
        return {"acc": acc, "acc_stderr": _stderr(acc, n), "perplexity": math.exp(-mean_ll), "n": n}
    hits = hits_norm = 0
    for ex, lls in zip(examples, per_example):
        # Lengths of the answers themselves, without the separating space, as the harness does.
        norm = [ll / len(c[1:] if c.startswith(" ") else c) for ll, c in zip(lls, ex.choices)]
        hits += int(max(range(len(lls)), key=lambda j: (lls[j], -j)) == ex.label)
        hits_norm += int(max(range(len(norm)), key=lambda j: (norm[j], -j)) == ex.label)
    acc, acc_norm = hits / n, hits_norm / n
    return {"acc": acc, "acc_stderr": _stderr(acc, n), "acc_norm": acc_norm,
            "acc_norm_stderr": _stderr(acc_norm, n), "n": n}

"""
Tolerant parsing of model outputs and gold answers for verifiable rewards.

The model is trained (via SFT) to answer in the form::

    <think> step by step reasoning ... </think><answer>42</answer>

but a small model is inconsistent, so parsing falls back gracefully: prefer an
``<answer>`` tag, then a GSM8K-style ``#### N``, then the last number in the text.
"""

from __future__ import annotations

import re

from src.post_training.chat_template import (
    ANSWER_OPEN, ANSWER_CLOSE, THINK_OPEN, THINK_CLOSE,
)

# A number: optional sign, digits with optional thousands commas, optional decimal.
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_ANSWER_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.DOTALL)
_THINK_RE = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.DOTALL)
_HASH_RE = re.compile(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)")


def parse_number(s: str | None) -> float | None:
    """Parse the first number out of a string, tolerating ``$`` and thousands commas."""
    if s is None:
        return None
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    raw = m.group(0).replace(",", "").replace("$", "")
    try:
        return float(raw)
    except ValueError:
        return None


def extract_think(text: str) -> str | None:
    """Return the content of the first ``<think>...</think>`` block, if any."""
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> float | None:
    """
    Extract the model's final numeric answer with graceful fallback:
    1) inside an ``<answer>...</answer>`` tag, else
    2) after a GSM8K-style ``#### N``, else
    3) the last number appearing anywhere in the text.
    """
    m = _ANSWER_RE.search(text)
    if m:
        n = parse_number(m.group(1))
        if n is not None:
            return n
    m = _HASH_RE.search(text)
    if m:
        n = parse_number(m.group(1))
        if n is not None:
            return n
    nums = _NUMBER_RE.findall(text)
    if nums:
        return parse_number(nums[-1])
    return None


def gsm8k_gold_answer(answer_field: str) -> float | None:
    """Extract the numeric gold answer from a GSM8K ``answer`` field (ends in ``#### N``)."""
    m = _HASH_RE.search(answer_field)
    if m:
        return parse_number(m.group(1))
    return parse_number(answer_field)


def has_well_formed_answer(text: str) -> bool:
    """True iff there is exactly one well-formed ``<answer>...</answer>`` block."""
    return len(_ANSWER_RE.findall(text)) == 1

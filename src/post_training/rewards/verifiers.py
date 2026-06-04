"""
Verifiable reward functions (RLVR). Each maps a decoded model response (+ a gold answer)
to a scalar reward. The design is deliberately **correctness-dominant** with only a small,
bounded format bonus, to limit reward hacking (a tiny model will happily emit empty
``<answer></answer>`` tags or repeat tokens if the format bonus is too large).

Reward decomposition (clipped to [0, 1.2]):
    +1.0  if the parsed final answer matches the gold answer (exact, with float tolerance)
    +0.2  if there is exactly one well-formed <answer>...</answer> block (format shaping)
"""

from __future__ import annotations

import math

from src.post_training.rewards.parsing import extract_answer, has_well_formed_answer

CORRECT_BONUS = 1.0
FORMAT_BONUS = 0.2
REWARD_CLIP = 1.2
FLOAT_TOL = 1e-4


def _answers_match(pred: float | None, gold: float | None) -> bool:
    if pred is None or gold is None:
        return False
    return math.isclose(pred, gold, rel_tol=0.0, abs_tol=FLOAT_TOL)


def is_correct(text: str, gold: float | None) -> bool:
    """Whether the model's parsed answer matches the gold answer (used for accuracy eval)."""
    return _answers_match(extract_answer(text), gold)


def reward_gsm8k(text: str, gold: float | None) -> float:
    """Reward for a GSM8K response: correctness (dominant) + small format bonus, clipped."""
    r = 0.0
    if _answers_match(extract_answer(text), gold):
        r += CORRECT_BONUS
    if has_well_formed_answer(text):
        r += FORMAT_BONUS
    return min(r, REWARD_CLIP)


# Arithmetic warm-up uses the same logic (gold is a single number).
reward_arithmetic = reward_gsm8k


def reward_format(text: str) -> float:
    """Pure format reward (no gold): 1.0 if exactly one well-formed answer block else 0."""
    return 1.0 if has_well_formed_answer(text) else 0.0

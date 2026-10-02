"""Sampling filters: temperature, top-k, top-p (nucleus), min-p and greedy."""

from __future__ import annotations

import torch

from src.inference.sampling import filter_logits, sample_next_token

PROBS = torch.tensor([[0.50, 0.25, 0.15, 0.06, 0.04]])
LOGITS = PROBS.log()


def _kept(logits: torch.Tensor) -> list[int]:
    return torch.isfinite(logits[0]).nonzero().flatten().tolist()


def test_top_k_keeps_the_k_most_likely() -> None:
    assert _kept(filter_logits(LOGITS, top_k=2)) == [0, 1]


def test_top_p_keeps_the_smallest_set_reaching_p() -> None:
    assert _kept(filter_logits(LOGITS, top_p=0.7)) == [0, 1]   # 0.50 + 0.25 crosses 0.7
    assert _kept(filter_logits(LOGITS, top_p=0.1)) == [0]      # always keeps the top token


def test_min_p_scales_with_the_top_probability() -> None:
    assert _kept(filter_logits(LOGITS, min_p=0.4)) == [0, 1]   # keep p >= 0.4 * 0.50 = 0.20
    flat = torch.zeros(1, 5)                                   # an uncertain step keeps everything
    assert _kept(filter_logits(flat, min_p=0.4)) == [0, 1, 2, 3, 4]


def test_temperature_and_greedy() -> None:
    sharp = filter_logits(LOGITS, temperature=0.5).softmax(-1)
    assert sharp[0, 0] > PROBS[0, 0]
    assert sample_next_token(LOGITS, greedy=True).item() == 0
    torch.manual_seed(0)
    draws = torch.cat([sample_next_token(LOGITS, top_k=1) for _ in range(20)])
    assert (draws == 0).all()

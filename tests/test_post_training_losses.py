"""The preference-optimization family: DPO, IPO, SimPO and conservative DPO."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.post_training.dpo import dpo_loss, implicit_accuracy, ipo_loss, simpo_loss


def _pairs():
    torch.manual_seed(0)
    pc, pr, rc, rr = (torch.randn(6) * 5 - 20 for _ in range(4))
    n = torch.full((6,), 10.0)
    return pc, pr, rc, rr, n


def test_dpo_matches_the_formula_and_label_smoothing_hedges() -> None:
    pc, pr, rc, rr, _ = _pairs()
    loss, cr, rj = dpo_loss(pc, pr, rc, rr, beta=0.1)
    logits = 0.1 * ((pc - pr) - (rc - rr))
    assert torch.allclose(loss, -F.logsigmoid(logits).mean())
    assert torch.equal(implicit_accuracy(cr, rj), ((pc - rc) > (pr - rr)).float().mean())
    # conservative DPO: with eps = 0.5 the two directions cancel and the loss stops depending on the order
    smoothed, _, _ = dpo_loss(pc, pr, rc, rr, beta=0.1, label_smoothing=0.3)
    flipped, _, _ = dpo_loss(pr, pc, rr, rc, beta=0.1, label_smoothing=0.3)
    plain_flipped, _, _ = dpo_loss(pr, pc, rr, rc, beta=0.1)
    assert abs(smoothed - flipped) < abs(loss - plain_flipped)


def test_ipo_is_zero_at_its_target_gap() -> None:
    beta = 0.25
    n = torch.ones(1)
    target = 1.0 / (2 * beta)  # IPO wants exactly this gap, not an infinite one
    loss, _, _ = ipo_loss(torch.tensor([target]), torch.zeros(1), torch.zeros(1), torch.zeros(1), n, n, beta)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    bigger, _, _ = ipo_loss(torch.tensor([target + 3]), torch.zeros(1), torch.zeros(1), torch.zeros(1), n, n, beta)
    assert bigger.item() > 1.0  # overshooting is penalized too, unlike DPO


def test_simpo_is_reference_free_and_length_normalized() -> None:
    pc, pr, _, _, n = _pairs()
    loss, cr, rj = simpo_loss(pc, pr, n, n, beta=2.0, gamma=0.5)
    expected = -F.logsigmoid(2.0 * pc / 10 - 2.0 * pr / 10 - 0.5).mean()
    assert torch.allclose(loss, expected)
    # twice as long answers with twice the total log-prob get the same reward
    loss_long, _, _ = simpo_loss(2 * pc, 2 * pr, 2 * n, 2 * n, beta=2.0, gamma=0.5)
    assert torch.allclose(loss, loss_long)


"""The preference-optimization family (DPO, IPO, SimPO, conservative DPO) and the GRPO variants."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.post_training.dpo import dpo_loss, implicit_accuracy, ipo_loss, simpo_loss
from src.post_training.grpo import aggregate_token_loss, group_advantages, grpo_loss


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


def test_group_advantages_std_and_dr_grpo() -> None:
    r = torch.tensor([1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 2.0, 2.0])
    std_adv = group_advantages(r, 4)
    centered = group_advantages(r, 4, scale="none")
    assert torch.allclose(centered, torch.tensor([0.75, -0.25, -0.25, -0.25, 1.0, -1.0, 0.0, 0.0]))
    assert torch.allclose(std_adv[:4] * r[:4].std(), centered[:4], atol=1e-3)


def test_loss_aggregation_modes() -> None:
    per_token = torch.tensor([[1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    assert aggregate_token_loss(per_token, mask, "token-mean").item() == pytest.approx(8 / 5)
    assert aggregate_token_loss(per_token, mask, "seq-mean-token-mean").item() == pytest.approx((1 + 4) / 2)
    assert aggregate_token_loss(per_token, mask, "seq-mean-token-sum-norm", max_len=8).item() == pytest.approx((4 / 8 + 4 / 8) / 2)


def test_skipped_groups_do_not_shrink_the_update() -> None:
    """--filter_groups masks whole answers; every averaging mode must ignore them, not count zeros."""
    torch.manual_seed(0)
    old = torch.randn(4, 5) - 2
    new = old + 0.1 * torch.randn(4, 5)
    adv = torch.tensor([1.0, -0.5, 0.0, 0.0])
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[0, 3:] = False
    skipped = mask.clone()
    skipped[2:] = False  # the last two answers belong to a group with no reward spread
    for mode in ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum-norm"):
        full = aggregate_token_loss(new, skipped, mode, max_len=5)
        kept = aggregate_token_loss(new[:2], skipped[:2], mode, max_len=5)
        assert full.item() == pytest.approx(kept.item()), mode
    for level in ("token", "sequence"):
        full, _ = grpo_loss(new, old, old, adv, skipped, kl_coef=0.04, ratio_level=level, loss_agg="seq-mean-token-mean")
        kept, _ = grpo_loss(new[:2], old[:2], old[:2], adv[:2], skipped[:2], kl_coef=0.04, ratio_level=level,
                            loss_agg="seq-mean-token-mean")
        assert full.item() == pytest.approx(kept.item()), level
    empty = torch.zeros(4, 5, dtype=torch.bool)  # a micro-batch where every group was skipped
    assert aggregate_token_loss(new, empty, "seq-mean-token-mean").item() == 0.0


def test_grpo_default_is_unchanged_and_clip_higher_widens_the_range() -> None:
    torch.manual_seed(0)
    B, L = 4, 6
    old = torch.randn(B, L) - 2
    new = old + 0.25  # ratio = e^0.25 = 1.28 everywhere
    ref = old.clone()
    adv = torch.tensor([1.0, -1.0, 2.0, 0.5])
    mask = torch.ones(B, L, dtype=torch.bool)
    loss, stats = grpo_loss(new, old, ref, adv, mask, clip=0.2, kl_coef=0.0)
    ratio = math.exp(0.25)
    clipped = torch.tensor([min(ratio * a, min(max(ratio, 0.8), 1.2) * a) for a in adv.tolist()])
    assert loss.item() == pytest.approx(-clipped.mean().item(), rel=1e-5)
    assert stats["clipfrac"] == pytest.approx(1.0)
    _, wide = grpo_loss(new, old, ref, adv, mask, clip=0.2, clip_high=0.3, kl_coef=0.0)
    assert wide["clipfrac"] == pytest.approx(0.0)  # 1.28 < 1.3: inside the clip-higher range


def test_gspo_uses_one_ratio_per_answer() -> None:
    old = torch.zeros(2, 4)
    new = torch.tensor([[0.1, -0.1, 0.2, -0.2], [0.0, 0.0, 0.0, 0.0]])  # mean log-ratio 0 for both rows
    mask = torch.ones(2, 4, dtype=torch.bool)
    adv = torch.tensor([1.0, -1.0])
    loss, stats = grpo_loss(new, old, old, adv, mask, clip=1e-3, kl_coef=0.0, ratio_level="sequence")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)  # both sequence ratios are exactly 1
    assert stats["clipfrac"] == 0.0
    new.requires_grad_(True)
    loss, _ = grpo_loss(new, old, old, adv, mask, clip=1e-3, kl_coef=0.0, ratio_level="sequence")
    loss.backward()
    assert new.grad is not None and torch.allclose(new.grad[0], torch.full((4,), -0.125))

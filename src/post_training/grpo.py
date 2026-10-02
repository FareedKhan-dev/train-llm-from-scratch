"""
GRPO (Group Relative Policy Optimization) core -- the DeepSeekMath / DeepSeek-R1 RL algorithm,
plus the switches that turn it into the 2025 follow-ups.

GRPO drops PPO's value network. For each prompt it samples a GROUP of G completions,
scores them with a verifiable reward, and uses the group's own mean/std to compute a
relative advantage -- so the baseline is the group, not a learned critic. The update is a
clipped surrogate plus a per-token KL penalty to the reference policy.

The variants differ in a few precise places, each one a keyword here:

=================  ==========================================================================
variant            what changes (keyword)
=================  ==========================================================================
GRPO               ``adv_norm="std"``; loss averaged per answer, then over answers
                   (``loss_agg="seq-mean-token-mean"``, the paper's formula)
DAPO               all tokens of the batch averaged together (``loss_agg="token-mean"``), a
                   larger upper clip ``clip_high`` ("clip-higher", keeps exploration alive),
                   groups with no reward spread skipped (``filter_groups`` in the trainer),
                   usually ``kl_coef=0``
Dr. GRPO           no division by the group std (``adv_norm="none"``) and a constant length
                   normalizer (``loss_agg="seq-mean-token-sum-norm"``); both remove biases
                   toward long wrong answers and toward easy/hard prompts
GSPO               one importance ratio per answer, the length-normalized sequence likelihood
                   ratio (``ratio_level="sequence"``), clipped with a tiny range (~3e-4)
=================  ==========================================================================

The trainer default (``token-mean``, symmetric clip, std-normalized advantages) is exactly
what this repo has always run.
"""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from src.post_training.utils import masked_mean, masked_mean_per_row

AdvantageNorm = Literal["std", "none"]
LossAggregation = Literal["token-mean", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
RatioLevel = Literal["token", "sequence"]


def group_advantages(
    rewards: Float[Tensor, " n"], group_size: int, eps: float = 1e-4, scale: AdvantageNorm = "std"
) -> Float[Tensor, " n"]:
    """
    Group-relative advantage: ``(r - group_mean) / (group_std + eps)``.

    ``rewards`` is (num_prompts * group_size,) laid out group-contiguously (all G samples
    of prompt 0, then prompt 1, ...). Returns advantages of the same shape. With
    ``scale="none"`` (Dr. GRPO) the advantage is just ``r - group_mean``: dividing by the std
    gives prompts that are almost always solved (or failed) a disproportionately large weight.
    """
    r = rewards.view(-1, group_size)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale == "std":
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.reshape(-1)


def k3_kl(new_logp: Float[Tensor, "*shape"], ref_logp: Float[Tensor, "*shape"]) -> Float[Tensor, "*shape"]:
    """Per-token unbiased, non-negative KL estimator (Schulman's k3) for KL(policy||ref)."""
    diff = ref_logp - new_logp
    return torch.exp(diff) - diff - 1.0


def aggregate_token_loss(
    per_token: Float[Tensor, "batch steps"],
    mask: Bool[Tensor, "batch steps"],
    mode: LossAggregation = "token-mean",
    max_len: int | None = None,
) -> Float[Tensor, ""]:
    """
    Reduce a per-token loss to one number. The choice matters more than it looks:

    - ``token-mean``: every token in the batch counts the same. Long answers get more say,
      which is what DAPO wants (long wrong answers are penalized in full).
    - ``seq-mean-token-mean``: average inside each answer, then across answers. Every answer
      counts the same, so each token of a long answer counts less. This is the GRPO paper.
    - ``seq-mean-token-sum-norm``: sum inside each answer, divide by a constant (the generation
      budget ``max_len``), then average. Dr. GRPO: unbiased and still length-independent.
    """
    m = mask.to(per_token.dtype)
    if mode == "token-mean":
        return masked_mean(per_token, m)
    if mode == "seq-mean-token-mean":
        return masked_mean_per_row(per_token, m).mean()
    if mode == "seq-mean-token-sum-norm":
        return ((per_token * m).sum(dim=-1) / float(max_len or per_token.size(-1))).mean()
    raise ValueError(f"unknown loss aggregation {mode!r}")


def grpo_loss(
    new_logp: Float[Tensor, "batch steps"],
    old_logp: Float[Tensor, "batch steps"],
    ref_logp: Float[Tensor, "batch steps"],
    advantages: Float[Tensor, " batch"],
    resp_mask: Bool[Tensor, "batch steps"],
    clip: float = 0.2,
    kl_coef: float = 0.04,
    *,
    clip_high: float | None = None,
    loss_agg: LossAggregation = "token-mean",
    ratio_level: RatioLevel = "token",
    max_len: int | None = None,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """
    Clipped surrogate + KL penalty, with the variant switches described in the module docstring.

    Args:
        new_logp/old_logp/ref_logp: (B, L) per-token log-probs (policy / sampling / ref).
        advantages: (B,) one scalar per completion, broadcast over its tokens.
        resp_mask:  (B, L) bool over response tokens.
        clip, clip_high: the ratio may move in ``[1 - clip, 1 + clip_high]`` (``clip_high``
            defaults to ``clip``).
        loss_agg: how per-token terms are averaged (token-level ratio only).
        ratio_level: ``"token"`` (GRPO/DAPO) or ``"sequence"`` (GSPO).
        max_len: the constant normalizer for ``seq-mean-token-sum-norm``.

    Returns:
        (loss, stats) with mean KL and clip fraction for logging.
    """
    high = clip if clip_high is None else clip_high
    adv = advantages[:, None]
    kl = k3_kl(new_logp, ref_logp)

    if ratio_level == "sequence":
        # GSPO: s_i = exp(mean_t log(pi/pi_old)), one ratio for the whole answer.
        log_ratio = masked_mean_per_row(new_logp - old_logp, resp_mask)
        ratio = torch.exp(log_ratio)
        surrogate = torch.min(ratio * advantages, torch.clamp(ratio, 1.0 - clip, 1.0 + high) * advantages)
        loss = -(surrogate - kl_coef * masked_mean_per_row(kl, resp_mask)).mean()
        clipped = ((ratio < 1.0 - clip) | (ratio > 1.0 + high)).float().mean()
    else:
        ratio = torch.exp(new_logp - old_logp)
        surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1.0 - clip, 1.0 + high) * adv)
        loss = -aggregate_token_loss(surrogate - kl_coef * kl, resp_mask, loss_agg, max_len)
        clipped = masked_mean(((ratio < 1.0 - clip) | (ratio > 1.0 + high)).float(), resp_mask)

    stats = {"kl": masked_mean(kl, resp_mask).item(), "clipfrac": clipped.item()}
    return loss, stats

"""
GRPO (Group Relative Policy Optimization) core -- the DeepSeek-R1 RL algorithm.

GRPO drops PPO's value network. For each prompt it samples a GROUP of G completions,
scores them with a verifiable reward, and uses the group's own mean/std to compute a
relative advantage -- so the baseline is the group, not a learned critic. The update is a
token-level clipped surrogate plus a per-token KL penalty to the reference policy.
"""

from __future__ import annotations

import torch

from src.post_training.utils import masked_mean


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    """
    Group-relative advantage: ``(r - group_mean) / (group_std + eps)``.

    ``rewards`` is (num_prompts * group_size,) laid out group-contiguously (all G samples
    of prompt 0, then prompt 1, ...). Returns advantages of the same shape.
    """
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True)
    adv = (r - mean) / (std + eps)
    return adv.reshape(-1)


def k3_kl(new_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token unbiased, non-negative KL estimator (Schulman's k3) for KL(policy||ref)."""
    diff = ref_logp - new_logp
    return torch.exp(diff) - diff - 1.0


def grpo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    resp_mask: torch.Tensor,
    clip: float = 0.2,
    kl_coef: float = 0.04,
) -> tuple[torch.Tensor, dict]:
    """
    Token-level clipped surrogate + KL penalty.

    Args:
        new_logp/old_logp/ref_logp: (B, L) per-token log-probs (policy / sampling / ref).
        advantages: (B,) one scalar per completion, broadcast over its tokens.
        resp_mask:  (B, L) bool over response tokens.

    Returns:
        (loss, stats) with mean KL and clip fraction for logging.
    """
    adv = advantages[:, None]
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    surrogate = torch.min(surr1, surr2)
    kl = k3_kl(new_logp, ref_logp)

    per_token = surrogate - kl_coef * kl
    loss = -masked_mean(per_token, resp_mask)
    stats = {
        "kl": masked_mean(kl, resp_mask).item(),
        "clipfrac": masked_mean(((ratio - 1.0).abs() > clip).float(), resp_mask).item(),
    }
    return loss, stats

"""
PPO core (the classic InstructGPT RLHF update), written from scratch.

The orchestration (rollout -> score -> GAE -> clipped epochs) lives in
``scripts/train_ppo.py``; this module holds the reusable, testable pieces:

- ``compute_gae``      : Generalized Advantage Estimation over response tokens.
- ``whiten``           : advantage normalization (masked).
- ``ppo_policy_loss``  : clipped surrogate objective.
- ``ppo_value_loss``   : clipped value-function loss.

All tensors are in the "action frame" of length ``L = T-1``: index ``t`` corresponds to
producing token ``t+1``, with state-value ``V(s_t)``, log-prob of the produced token, and
the reward received for that action. ``resp_mask`` marks which actions are response tokens.
"""

from __future__ import annotations

import torch

from src.post_training.utils import masked_mean


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    values_next: torch.Tensor,
    resp_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation over the response.

    Args (all (B, L) in the action frame; ``resp_mask`` bool):
        rewards:     per-action reward (KL penalty per token + task reward at the last
                     response token).
        values:      V(s_t).
        values_next: V(s_{t+1}).
        resp_mask:   True where action t is a response token. The episode is terminal
                     after the last response token (no bootstrap past it).

    Returns:
        (advantages, returns) each (B, L), zeroed outside the response.
    """
    B, L = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgae = torch.zeros(B, device=rewards.device)
    m = resp_mask.float()
    for t in reversed(range(L)):
        # Bootstrap only if the NEXT action is still a response action.
        nonterminal = m[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * values_next[:, t] * nonterminal - values[:, t]
        lastgae = delta + gamma * lam * nonterminal * lastgae
        adv[:, t] = lastgae
    returns = adv + values
    return adv * m, returns * m


def whiten(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize advantages to zero mean / unit std over masked (response) positions."""
    m = mask.float()
    mean = masked_mean(advantages, m)
    var = masked_mean((advantages - mean) ** 2, m)
    return ((advantages - mean) / (var.sqrt() + 1e-8)) * m


def ppo_policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clipped surrogate policy loss. Returns (loss, clip_fraction)."""
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    loss = -masked_mean(torch.min(surr1, surr2), mask)
    clipped = ((ratio - 1.0).abs() > clip).float()
    return loss, masked_mean(clipped, mask)


def ppo_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    vf_clip: float = 0.2,
) -> torch.Tensor:
    """Clipped value-function loss (0.5 * max of clipped/unclipped squared error)."""
    v_clipped = old_values + torch.clamp(new_values - old_values, -vf_clip, vf_clip)
    loss_unclipped = (new_values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    return 0.5 * masked_mean(torch.max(loss_unclipped, loss_clipped), mask)


def approx_kl(new_logp: torch.Tensor, old_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean approximate KL(old || new) over response tokens (a PPO health metric)."""
    return masked_mean(old_logp - new_logp, mask)

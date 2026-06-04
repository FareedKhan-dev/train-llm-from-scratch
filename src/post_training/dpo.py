"""
Direct Preference Optimization and two reference-free / unpaired variants.

All operate on sequence-level log-probabilities of the chosen/rejected responses (summed
over response tokens via :func:`src.post_training.rollout.sequence_logprobs`).

- ``dpo_loss``  : the standard DPO objective; aligns the policy to preferences using a
                  frozen reference model and temperature ``beta``.
- ``orpo_loss`` : ORPO -- reference-FREE; combines the SFT NLL on the chosen response with
                  an odds-ratio preference term (folds SFT + alignment into one stage).
- ``kto_loss``  : KTO -- works from a per-example desirable/undesirable signal (here read
                  off the chosen/rejected pair) with a reference KL baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard DPO loss. Inputs are summed response log-probs (B,).

    Returns ``(loss, chosen_reward, rejected_reward)`` where the implicit rewards
    ``beta * (policy_logp - ref_logp)`` are detached diagnostics.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_reward, rejected_reward


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x < 0."""
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def orpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_n_tokens: torch.Tensor,
    rejected_n_tokens: torch.Tensor,
    orpo_lambda: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ORPO (reference-free). Uses per-token MEAN log-probs.

    ``L = NLL(chosen) + lambda * -log sigmoid(log_odds_chosen - log_odds_rejected)``
    where ``log_odds = mean_logp - log(1 - exp(mean_logp))``.
    """
    chosen_mean = policy_chosen_logps / chosen_n_tokens.clamp(min=1)
    rejected_mean = policy_rejected_logps / rejected_n_tokens.clamp(min=1)
    log_odds = (chosen_mean - _log1mexp(chosen_mean)) - (rejected_mean - _log1mexp(rejected_mean))
    or_loss = -F.logsigmoid(log_odds).mean()
    nll = -chosen_mean.mean()
    loss = nll + orpo_lambda * or_loss
    # Implicit rewards for logging: the mean log-probs themselves.
    return loss, chosen_mean.detach(), rejected_mean.detach()


def kto_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    KTO from paired data: chosen = desirable, rejected = undesirable, with a reference-KL
    baseline estimated (detached) from the batch's mean log-ratio.
    """
    chosen_logratio = policy_chosen_logps - ref_chosen_logps
    rejected_logratio = policy_rejected_logps - ref_rejected_logps
    kl = torch.cat([chosen_logratio, rejected_logratio]).mean().clamp(min=0).detach()
    chosen_losses = 1.0 - torch.sigmoid(beta * (chosen_logratio - kl))
    rejected_losses = 1.0 - torch.sigmoid(beta * (kl - rejected_logratio))
    loss = (desirable_weight * chosen_losses).mean() + (undesirable_weight * rejected_losses).mean()
    return loss, (beta * chosen_logratio).detach(), (beta * rejected_logratio).detach()


def implicit_accuracy(chosen_reward: torch.Tensor, rejected_reward: torch.Tensor) -> torch.Tensor:
    """Fraction of pairs where the implicit (DPO/KTO/ORPO) reward prefers chosen."""
    return (chosen_reward > rejected_reward).float().mean()

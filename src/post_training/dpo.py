"""
Direct Preference Optimization and its main variants.

All operate on sequence-level log-probabilities of the chosen/rejected responses (summed
over response tokens via :func:`src.post_training.rollout.sequence_logprobs`), so they share
one trainer (``scripts/train_dpo.py --loss_type ...``).

- ``dpo_loss``   : the standard DPO objective (Rafailov et al. 2023); aligns the policy to
                   preferences using a frozen reference model and temperature ``beta``.
                   ``label_smoothing > 0`` gives conservative DPO for noisy labels.
- ``ipo_loss``   : IPO (Azar et al. 2023); regresses the preference gap to a fixed target
                   instead of pushing it to infinity, so it does not overfit easy pairs.
- ``simpo_loss`` : SimPO (Meng et al. 2024); reference-FREE, uses the *average* log-prob per
                   token as the reward (no bias toward long answers) plus a target margin.
- ``orpo_loss``  : ORPO -- reference-FREE; combines the SFT NLL on the chosen response with
                   an odds-ratio preference term (folds SFT + alignment into one stage).
- ``kto_loss``   : KTO -- works from a per-example desirable/undesirable signal (here read
                   off the chosen/rejected pair) with a reference KL baseline.

Every function returns ``(loss, chosen_reward, rejected_reward)``; the rewards are detached
diagnostics, and "implicit accuracy" is how often chosen_reward > rejected_reward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

Batch = Float[Tensor, " batch"]
LossAndRewards = tuple[Float[Tensor, ""], Batch, Batch]


def _preference_nll(logits: Batch, label_smoothing: float = 0.0) -> Float[Tensor, ""]:
    """``-log sigmoid(logits)``, optionally hedged against flipped labels (conservative DPO)."""
    if label_smoothing == 0.0:
        return -F.logsigmoid(logits).mean()
    return (-(1 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)).mean()


def dpo_loss(
    policy_chosen_logps: Batch,
    policy_rejected_logps: Batch,
    ref_chosen_logps: Batch,
    ref_rejected_logps: Batch,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> LossAndRewards:
    """
    Standard DPO loss. Inputs are summed response log-probs (B,).

    ``L = -log sigmoid(beta * [(pi_c - pi_r) - (ref_c - ref_r)])``. With
    ``label_smoothing = eps`` the loss assumes each label is flipped with probability eps
    (conservative DPO), which keeps the model from becoming overconfident on noisy pairs.

    Returns ``(loss, chosen_reward, rejected_reward)`` where the implicit rewards
    ``beta * (policy_logp - ref_logp)`` are detached diagnostics.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = _preference_nll(beta * logits, label_smoothing)
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_reward, rejected_reward


def ipo_loss(
    policy_chosen_logps: Batch,
    policy_rejected_logps: Batch,
    ref_chosen_logps: Batch,
    ref_rejected_logps: Batch,
    chosen_n_tokens: Batch,
    rejected_n_tokens: Batch,
    beta: float = 0.1,
) -> LossAndRewards:
    """
    IPO: ``L = (h - 1 / (2 beta))^2`` with ``h`` the DPO log-ratio gap on per-token averages.

    DPO keeps pushing ``h`` up even after a pair is clearly separated (the sigmoid never
    saturates to zero loss), which can overfit when preferences are near-deterministic. IPO
    asks for a fixed gap of ``1 / (2 beta)`` and stops there.
    """
    nc, nr = chosen_n_tokens.clamp(min=1), rejected_n_tokens.clamp(min=1)
    h = (policy_chosen_logps / nc - policy_rejected_logps / nr) - (ref_chosen_logps / nc - ref_rejected_logps / nr)
    loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_reward, rejected_reward


def simpo_loss(
    policy_chosen_logps: Batch,
    policy_rejected_logps: Batch,
    chosen_n_tokens: Batch,
    rejected_n_tokens: Batch,
    beta: float = 2.0,
    gamma: float = 0.5,
    label_smoothing: float = 0.0,
) -> LossAndRewards:
    """
    SimPO: ``L = -log sigmoid(beta * avg_logp(chosen) - beta * avg_logp(rejected) - gamma)``.

    Two differences from DPO: no reference model (cheaper, one model in memory), and the
    reward is the *average* log-prob per token, which removes DPO's pull toward longer
    answers. ``gamma`` asks for a minimum reward margin. SimPO uses a larger ``beta``
    (about 2.0 to 2.5) than DPO because the average log-prob is a much smaller number.
    """
    chosen_reward = beta * policy_chosen_logps / chosen_n_tokens.clamp(min=1)
    rejected_reward = beta * policy_rejected_logps / rejected_n_tokens.clamp(min=1)
    loss = _preference_nll(chosen_reward - rejected_reward - gamma, label_smoothing)
    return loss, chosen_reward.detach(), rejected_reward.detach()


def _log1mexp(x: Batch) -> Batch:
    """Numerically stable log(1 - exp(x)) for x < 0."""
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def orpo_loss(
    policy_chosen_logps: Batch,
    policy_rejected_logps: Batch,
    chosen_n_tokens: Batch,
    rejected_n_tokens: Batch,
    orpo_lambda: float = 1.0,
) -> LossAndRewards:
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
    policy_chosen_logps: Batch,
    policy_rejected_logps: Batch,
    ref_chosen_logps: Batch,
    ref_rejected_logps: Batch,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> LossAndRewards:
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


def implicit_accuracy(chosen_reward: Batch, rejected_reward: Batch) -> Float[Tensor, ""]:
    """Fraction of pairs where the implicit (DPO/IPO/SimPO/KTO/ORPO) reward prefers chosen."""
    return (chosen_reward > rejected_reward).float().mean()

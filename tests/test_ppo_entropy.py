"""PPO's ent_coef adds the entropy bonus it is documented to add (it used to do nothing)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from scripts.train_ppo import ppo_minibatch_loss
from src.models.transformer import Transformer
from src.post_training.utils import masked_mean
from src.post_training.value_head import TransformerWithValueHead


def _actor() -> TransformerWithValueHead:
    torch.manual_seed(0)
    return TransformerWithValueHead(Transformer(n_head=2, n_embed=16, context_length=12, vocab_size=40, N_BLOCKS=1))


def _batch() -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(2)
    seqs = torch.randint(0, 40, (3, 12), generator=gen)
    m = torch.zeros(3, 11, dtype=torch.bool)
    m[:, 5:] = True
    old_logp = -torch.rand(3, 11, generator=gen) - 3.0
    old_values = torch.randn(3, 11, generator=gen)
    adv, returns = torch.randn(3, 11, generator=gen), torch.randn(3, 11, generator=gen)
    return seqs, m, old_logp, old_values, adv, returns


def _cfg(ent_coef: float) -> SimpleNamespace:
    return SimpleNamespace(amp_dtype="fp32", temperature=1.0, clip=0.2, vf_clip=0.2, vf_coef=0.5, ent_coef=ent_coef)


def test_ent_coef_subtracts_the_mean_policy_entropy() -> None:
    seqs, m, *rest = _batch()
    actor = _actor()
    base = ppo_minibatch_loss(actor, seqs, m, *rest, _cfg(0.0), "cpu")[0]
    bonus = ppo_minibatch_loss(actor, seqs, m, *rest, _cfg(0.1), "cpu")[0]

    logits, _ = actor(seqs)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    entropy = masked_mean(-(logp.exp() * logp).sum(-1), m)
    assert float(entropy) > 0
    assert torch.allclose(bonus, base - 0.1 * entropy, atol=1e-6)


def test_the_entropy_bonus_reaches_the_gradient() -> None:
    seqs, m, *rest = _batch()
    grads = []
    for ent_coef in (0.0, 0.1):
        actor = _actor()
        ppo_minibatch_loss(actor, seqs, m, *rest, _cfg(ent_coef), "cpu")[0].backward()
        grads.append(actor.transformer.lm_head.weight.grad.clone())
    assert not torch.allclose(grads[0], grads[1])

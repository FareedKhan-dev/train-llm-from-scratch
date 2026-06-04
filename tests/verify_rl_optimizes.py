"""
Prove the RL machinery actually OPTIMIZES its reward (not just runs without error).

Uses the real GRPO and PPO code paths (rollout -> reward -> advantage -> clipped update)
on a tiny model with a learnable synthetic reward: "emit the target token". If the loops
are correct, the policy raises the probability of the target token and the mean reward
climbs. Runs on CPU in well under a minute (tiny vocab).

    PYTHONPATH=. python tests/verify_rl_optimizes.py
"""

import torch

from src.models.transformer import Transformer
from src.post_training.grpo import group_advantages, grpo_loss
from src.post_training.ppo import compute_gae, whiten, ppo_policy_loss, ppo_value_loss
from src.post_training.rollout import generate_with_logprobs, compute_logprobs
from src.post_training.utils import make_frozen_copy, set_seed
from src.post_training.value_head import TransformerWithValueHead

VOCAB, TARGET = 64, 7
PROMPT_LEN, GEN = 4, 8


def dense_reward(seq, prompt_len, target=TARGET):
    """Fraction of generated tokens equal to the target token (dense, easy to learn)."""
    gen = seq[prompt_len:]
    return (gen == target).float().mean().item()


def tiny_model():
    set_seed(0)
    return Transformer(n_head=4, n_embed=64, context_length=PROMPT_LEN + GEN + 2, vocab_size=VOCAB, N_BLOCKS=2)


def verify_grpo_optimizes():
    print("\n== GRPO optimizes reward (real grpo_loss path) ==")
    policy = tiny_model()
    ref = make_frozen_copy(policy)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    G = 8
    prompt = torch.zeros(1, PROMPT_LEN, dtype=torch.long)
    history = []
    for it in range(60):
        batch = prompt.repeat(G, 1)
        rb = generate_with_logprobs(policy, batch, GEN, temperature=1.0)
        seqs, rmask = rb.sequences, rb.response_mask
        rewards = torch.tensor([dense_reward(seqs[i], PROMPT_LEN) for i in range(G)])
        adv = group_advantages(rewards, G)
        old_lp, _ = compute_logprobs(policy, seqs, rmask, requires_grad=False)
        ref_lp, _ = compute_logprobs(ref, seqs, rmask, requires_grad=False)
        new_lp, _ = compute_logprobs(policy, seqs, rmask, requires_grad=True)
        loss, _ = grpo_loss(new_lp, old_lp, ref_lp, adv, rmask[:, 1:], clip=0.2, kl_coef=0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(rewards.mean().item())
    start, end = sum(history[:5]) / 5, sum(history[-5:]) / 5
    print(f"  mean reward: start {start:.3f} -> end {end:.3f}")
    assert end > start + 0.2, f"GRPO did not improve reward ({start:.3f} -> {end:.3f})"
    print(f"  [PASS] GRPO raised P(target token); reward climbed {start:.2f} -> {end:.2f}")


def verify_ppo_optimizes():
    print("\n== PPO optimizes reward (real GAE + clipped losses path) ==")
    backbone = tiny_model()
    ref = make_frozen_copy(backbone)
    actor = TransformerWithValueHead(backbone)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    B = 8
    prompt = torch.zeros(B, PROMPT_LEN, dtype=torch.long)
    history = []
    for it in range(60):
        rb = generate_with_logprobs(actor, prompt, GEN, temperature=1.0)
        seqs, rmask = rb.sequences, rb.response_mask
        resp = rmask[:, 1:]
        task_r = torch.tensor([dense_reward(seqs[i], PROMPT_LEN) for i in range(B)])

        with torch.no_grad():
            logits, values = actor(seqs)
            old_lp, _ = compute_logprobs(actor, seqs, rmask, requires_grad=False)
        old_v = values[:, :-1]
        # sparse terminal reward at last response token
        rewards = torch.zeros_like(resp, dtype=torch.float32)
        last = resp.float().cumsum(1).argmax(1)
        rewards[torch.arange(B), last] = task_r
        vnext = torch.cat([old_v[:, 1:], torch.zeros_like(old_v[:, :1])], 1)
        adv, ret = compute_gae(rewards, old_v, vnext, resp, gamma=1.0, lam=0.95)
        adv = whiten(adv, resp)
        for _ in range(4):
            logits, values = actor(seqs)
            new_lp, _ = compute_logprobs(actor, seqs, rmask, requires_grad=True)
            pl, _ = ppo_policy_loss(new_lp, old_lp, adv, resp, clip=0.2)
            vl = ppo_value_loss(values[:, :-1], old_v, ret, resp, vf_clip=0.2)
            loss = pl + 0.5 * vl
            opt.zero_grad(); loss.backward(); opt.step()
        history.append(task_r.mean().item())
    start, end = sum(history[:5]) / 5, sum(history[-5:]) / 5
    print(f"  mean reward: start {start:.3f} -> end {end:.3f}")
    assert end > start + 0.15, f"PPO did not improve reward ({start:.3f} -> {end:.3f})"
    print(f"  [PASS] PPO raised reward {start:.2f} -> {end:.2f}")


if __name__ == "__main__":
    verify_grpo_optimizes()
    verify_ppo_optimizes()
    print("\nRL OPTIMIZATION VERIFIED: both PPO and GRPO increase their reward.")

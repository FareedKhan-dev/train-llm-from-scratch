"""
Unit tests for the from-scratch RL math (PPO + GRPO). Pure-CPU, runs in milliseconds.

    PYTHONPATH=. python tests/test_rl_math.py
"""

import torch

from src.post_training.ppo import compute_gae, whiten, ppo_policy_loss, ppo_value_loss, approx_kl
from src.post_training.grpo import group_advantages, grpo_loss, k3_kl


def test_gae_reward_to_go():
    # lambda=1, zero values -> advantage is plain reward-to-go (=1 everywhere here).
    r = torch.tensor([[0.0, 0.0, 1.0]])
    v = torch.zeros(1, 3)
    m = torch.ones(1, 3, dtype=torch.bool)
    adv, ret = compute_gae(r, v, v, m, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.ones(1, 3)), adv
    assert torch.allclose(ret, torch.ones(1, 3)), ret
    # lambda<1 discounts earlier advantages.
    adv2, _ = compute_gae(r, v, v, m, gamma=1.0, lam=0.95)
    assert adv2[0, 0] < adv2[0, 1] < adv2[0, 2]
    print("ok  GAE reward-to-go + lambda discounting")


def test_gae_masks_outside_response():
    r = torch.tensor([[0.0, 1.0, 0.0]])
    v = torch.zeros(1, 3)
    m = torch.tensor([[0, 1, 0]], dtype=torch.bool)  # only middle token is a response token
    adv, ret = compute_gae(r, v, v, m, gamma=1.0, lam=1.0)
    assert adv[0, 0] == 0 and adv[0, 2] == 0
    print("ok  GAE zeros advantages outside the response")


def test_whiten():
    a = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
    m = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    w = whiten(a, m)
    assert abs(w[0, :3].mean().item()) < 1e-5
    assert w[0, 3] == 0
    print("ok  whiten: zero-mean over mask, zero outside")


def test_ppo_losses():
    m = torch.ones(1, 3, dtype=torch.bool)
    # ratio == 1 -> policy loss = -mean(advantage)
    loss, clipf = ppo_policy_loss(torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1, 3), m, clip=0.2)
    assert abs(loss.item() + 1.0) < 1e-5 and clipf.item() == 0.0
    # value loss = 0.5 * MSE
    vl = ppo_value_loss(torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1, 3), m, vf_clip=0.2)
    assert abs(vl.item() - 0.5) < 1e-5
    # clipping engages for a large ratio
    _, cf = ppo_policy_loss(torch.full((1, 3), 1.0), torch.zeros(1, 3), torch.ones(1, 3), m, clip=0.2)
    assert cf.item() == 1.0
    print("ok  ppo policy/value loss + clip fraction")


def test_group_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    adv = group_advantages(r, 4)
    assert abs(adv[:4].mean().item()) < 1e-5
    assert adv[4:].abs().max().item() < 1e-3  # zero-variance group -> ~zero advantage
    assert adv[0] > 0 and adv[1] < 0
    print("ok  group_advantages: group-relative, zero-variance group ~0")


def test_grpo_loss_and_kl():
    B, L = 2, 3
    z = torch.zeros(B, L)
    m = torch.ones(B, L, dtype=torch.bool)
    loss, st = grpo_loss(z, z, z, torch.ones(B), m, clip=0.2, kl_coef=0.04)
    assert abs(loss.item() + 1.0) < 1e-5      # ratio=1, adv=1, kl=0 -> -1
    assert abs(st["kl"]) < 1e-6
    kl = k3_kl(torch.tensor([-1.0, 0.5, 0.0]), torch.zeros(3))
    assert (kl >= 0).all()                    # k3 KL is non-negative
    print("ok  grpo_loss + non-negative k3 KL")


if __name__ == "__main__":
    test_gae_reward_to_go()
    test_gae_masks_outside_response()
    test_whiten()
    test_ppo_losses()
    test_group_advantages()
    test_grpo_loss_and_kl()
    print("\nALL RL MATH TESTS PASSED")

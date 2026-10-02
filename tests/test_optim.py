"""Muon, the AdamW fallback inside it, and the learning-rate schedules."""

from __future__ import annotations

import pytest
import torch

from src.models.modern import ModernConfig, ModernTransformer
from src.models.transformer import Transformer
from src.optim import Muon, build_optimizer, cosine_lr, linear_lr, lr_at, muon_param_groups, newton_schulz, wsd_lr


def test_newton_schulz_flattens_the_singular_values() -> None:
    torch.manual_seed(0)
    G = torch.randn(32, 64) @ torch.diag(torch.logspace(-2, 1, 64))  # badly conditioned
    s_before = torch.linalg.svdvals(G)
    s_after = torch.linalg.svdvals(newton_schulz(G, steps=5, dtype=torch.float32))
    assert s_before.max() / s_before.min() > 50
    assert s_after.min() > 0.3 and s_after.max() < 1.3  # Muon's quintic lands in roughly [0.5, 1.5]
    tall = newton_schulz(torch.randn(64, 16), dtype=torch.float32)
    assert tall.shape == (64, 16)


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon needs PyTorch >= 2.9")
def test_muon_matches_pytorch_reference() -> None:
    torch.manual_seed(0)
    start = torch.randn(24, 40)
    w_ours = torch.nn.Parameter(start.clone())
    w_ref = torch.nn.Parameter(start.clone())
    ours = Muon([{"params": [w_ours], "use_muon": True}], lr=0.01, weight_decay=0.1)
    ref = torch.optim.Muon([w_ref], lr=0.01, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
    for _ in range(3):
        g = torch.randn(24, 40)
        for w, opt in ((w_ours, ours), (w_ref, ref)):
            w.grad = g.clone()
            opt.step()
    # Both run Newton-Schulz in bfloat16, but PyTorch fuses the steps into addmm, so rounding
    # differs slightly. Compare the total update: same direction and the same size within 1%.
    d_ours, d_ref = (w_ours - start).flatten(), (w_ref - start).flatten()
    assert torch.nn.functional.cosine_similarity(d_ours, d_ref, dim=0) > 0.999
    assert abs(d_ours.norm() / d_ref.norm() - 1) < 0.01


def test_muon_adamw_groups_match_torch_adamw() -> None:
    torch.manual_seed(0)
    p_ours = torch.nn.Parameter(torch.randn(10))
    p_ref = torch.nn.Parameter(p_ours.detach().clone())
    ours = Muon([{"params": [p_ours], "use_muon": False}], lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
    ref = torch.optim.AdamW([p_ref], lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    for _ in range(5):
        g = torch.randn(10)
        for p, opt in ((p_ours, ours), (p_ref, ref)):
            p.grad = g.clone()
            opt.step()
    assert torch.allclose(p_ours, p_ref, atol=1e-6)


def test_muon_groups_keep_embeddings_and_head_on_adamw() -> None:
    model = ModernTransformer(ModernConfig(vocab_size=64, context_length=16, n_embed=32, n_head=4, n_blocks=2))
    muon, adam_decay, adam_no_decay = muon_param_groups(model, weight_decay=0.1)
    embed_id = id(model.token_embed.weight)
    assert embed_id not in {id(p) for p in muon["params"]}
    assert embed_id in {id(p) for p in adam_decay["params"]}
    assert all(p.ndim == 2 for p in muon["params"]) and all(p.ndim == 1 for p in adam_no_decay["params"])
    n_grouped = sum(len(g["params"]) for g in (muon, adam_decay, adam_no_decay))
    assert n_grouped == len(list(model.parameters()))


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_both_optimizers_train_a_tiny_model(name: str) -> None:
    torch.manual_seed(0)
    model = Transformer(n_head=4, n_embed=32, context_length=16, vocab_size=50, N_BLOCKS=2)
    opt = build_optimizer(model, name, lr=3e-3, weight_decay=0.0)
    data = torch.randint(0, 50, (8, 17))
    losses = []
    for _ in range(40):
        _, loss = model(data[:, :-1], data[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.6 * losses[0]  # memorizes the fixed batch


def test_schedules_hit_their_anchor_points() -> None:
    kw = dict(warmup_steps=10, max_steps=110, lr=1.0, min_lr=0.1)
    for fn in (cosine_lr, wsd_lr, linear_lr):
        assert fn(0, **kw) == pytest.approx(0.1)  # first warmup step
        assert fn(9, **kw) == pytest.approx(1.0)  # end of warmup
        assert fn(110, **kw) == pytest.approx(0.1) and fn(500, **kw) == pytest.approx(0.1)
    assert cosine_lr(60, **kw) == pytest.approx(0.55)  # halfway down the cosine
    assert linear_lr(60, **kw) == pytest.approx(0.55)
    assert wsd_lr(80, **kw) == pytest.approx(1.0)  # still in the stable phase
    assert wsd_lr(99, **kw) == pytest.approx(0.55, abs=0.03)  # halfway through the final 20%
    assert lr_at("wsd", 80, **kw) == wsd_lr(80, **kw)
    # A bad name is a ValueError, or a TypeError first when the runtime shape/type checks are on.
    with pytest.raises((ValueError, TypeError)):
        lr_at("constant", 0, **kw)  # type: ignore[arg-type]

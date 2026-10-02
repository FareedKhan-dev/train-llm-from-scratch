"""
Tests for the modern decoder (src/models/modern).

Most tests check a property that must hold rather than a hard-coded number: causality,
cached decoding matching a full forward pass, rotary embeddings depending only on relative
position, grouped-query attention matching multi-head attention with copied heads, and so on.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from src.models.modern import (
    GroupedQueryAttention,
    ModernConfig,
    ModernTransformer,
    MoE,
    RMSNorm,
    apply_rope,
    rope_cache,
)

BASE = ModernConfig(vocab_size=97, context_length=32, n_embed=32, n_head=4, n_blocks=2, n_kv_head=2)

VARIANTS = {
    "gqa": BASE,
    "mha": replace(BASE, n_kv_head=None),
    "mqa": replace(BASE, n_kv_head=1),
    "mla": replace(BASE, attention="mla", kv_latent_dim=12, rope_dim=4),
    "window": replace(BASE, sliding_window=5),
    "moe": replace(BASE, n_experts=4, moe_top_k=2, n_shared_experts=1),
    "gated_no_qknorm": replace(BASE, attn_gate=True, qk_norm=False, tie_embeddings=False),
}


def _model(cfg: ModernConfig, seed: int = 0) -> ModernTransformer:
    torch.manual_seed(seed)
    return ModernTransformer(cfg).eval()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_forward_shapes_and_loss(name: str) -> None:
    cfg = VARIANTS[name]
    model = _model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 16))
    targets = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0
    # a freshly initialized model should be close to a uniform guess: loss ~ ln(vocab)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


@pytest.mark.parametrize("name", list(VARIANTS))
def test_causality(name: str) -> None:
    """Changing a token must not change the predictions for earlier positions."""
    cfg = VARIANTS[name]
    model = _model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 20))
    changed = idx.clone()
    changed[:, 12] = (changed[:, 12] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-5)
    assert not torch.allclose(a[:, 12:], b[:, 12:], atol=1e-5)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_kv_cache_matches_full_forward(name: str) -> None:
    """Prefill 7 tokens, then decode one token at a time: logits must equal a full pass."""
    cfg = VARIANTS[name]
    model = _model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 18))
    with torch.no_grad():
        full, _ = model(idx)
        cache = model.new_cache(batch_size=2)
        steps = [model.lm_head(model.forward_hidden(idx[:, :7], cache))]
        for t in range(7, 18):
            steps.append(model.lm_head(model.forward_hidden(idx[:, t : t + 1], cache)))
    assert cache.length == 18
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4)


def test_sliding_window_limits_what_a_token_sees() -> None:
    """With one block and window w, position t only depends on tokens t-w+1 .. t."""
    cfg = replace(BASE, n_blocks=1, sliding_window=4)
    model = _model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    changed = idx.clone()
    changed[:, 5] = (changed[:, 5] + 1) % cfg.vocab_size  # outside the window of position 9..15
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[:, 9:], b[:, 9:], atol=1e-5)
    assert not torch.allclose(a[:, 5:9], b[:, 5:9], atol=1e-5)


def test_rope_scores_depend_only_on_relative_position() -> None:
    torch.manual_seed(0)
    cos, sin = rope_cache(dim=16, max_len=64)
    q, k = torch.randn(16), torch.randn(16)

    def score(m: int, n: int) -> float:
        qm = apply_rope(q[None], cos[m : m + 1], sin[m : m + 1])[0]
        kn = apply_rope(k[None], cos[n : n + 1], sin[n : n + 1])[0]
        return float(qm @ kn)

    assert math.isclose(score(10, 3), score(40, 33), rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(score(5, 5), float(q @ k), rel_tol=1e-4, abs_tol=1e-4)  # same position: no change


def test_rope_is_a_rotation_of_each_pair() -> None:
    """Pair (x[i], x[i + d/2]) at position m is rotated by the angle m * theta_i."""
    torch.manual_seed(0)
    dim, m = 8, 7
    cos, sin = rope_cache(dim=dim, max_len=16)
    x = torch.randn(dim)
    out = apply_rope(x[None], cos[m : m + 1], sin[m : m + 1])[0]
    half = dim // 2
    for i in range(half):
        angle = m * 10_000.0 ** (-2 * i / dim)
        rot = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        expected = rot @ torch.stack([x[i], x[i + half]])
        assert torch.allclose(torch.stack([out[i], out[i + half]]), expected, atol=1e-5)
    assert torch.allclose(out.norm(), x.norm(), atol=1e-5)  # rotations keep the length


def test_rmsnorm_matches_torch() -> None:
    x = torch.randn(4, 10, 32)
    ours, ref = RMSNorm(32, eps=1e-6), torch.nn.RMSNorm(32, eps=1e-6)
    with torch.no_grad():
        ours.weight.uniform_(0.5, 1.5)
        ref.weight.copy_(ours.weight)
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


def test_gqa_equals_mha_with_shared_heads() -> None:
    """GQA with 2 KV heads == MHA whose 4 KV heads are copies (heads 0,1 share KV 0; 2,3 share KV 1)."""
    gqa_cfg = replace(BASE, n_kv_head=2, qk_norm=False)
    mha_cfg = replace(BASE, n_kv_head=4, qk_norm=False)
    torch.manual_seed(0)
    gqa, mha = GroupedQueryAttention(gqa_cfg, 0), GroupedQueryAttention(mha_cfg, 0)
    D = BASE.head_dim
    with torch.no_grad():
        mha.q_proj.weight.copy_(gqa.q_proj.weight)
        mha.o_proj.weight.copy_(gqa.o_proj.weight)
        for proj in ("k_proj", "v_proj"):
            w = getattr(gqa, proj).weight.view(2, D, -1)  # (kv_heads, D, C)
            getattr(mha, proj).weight.copy_(w.repeat_interleave(2, dim=0).reshape(4 * D, -1))
    x = torch.randn(2, 9, BASE.n_embed)
    cos, sin = rope_cache(D, 9)
    assert torch.allclose(gqa(x, cos, sin), mha(x, cos, sin), atol=1e-5)


def test_mla_caches_much_less_than_mha() -> None:
    mha = _model(replace(BASE, n_kv_head=None)).new_cache(1)
    mla = _model(VARIANTS["mla"]).new_cache(1)
    assert mla.k.shape[-1] == 12 and mla.v.shape[-1] == 4  # latent + shared rotary key
    assert mla.bytes_per_token() * 3 < mha.bytes_per_token()


def test_moe_routes_tokens_and_balancing_loss_trains_the_router() -> None:
    torch.manual_seed(0)
    moe = MoE(n_embed=16, hidden=32, n_experts=4, top_k=2)
    x = torch.randn(2, 10, 16)
    y = moe(x)
    assert y.shape == x.shape
    assert moe.tokens_per_expert is not None and moe.tokens_per_expert.sum().item() == 2 * 10 * 2
    assert moe.aux_loss is not None and moe.aux_loss.item() >= 0.99  # == 1 only for perfect balance
    moe.aux_loss.backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0


def test_moe_loss_includes_aux_and_active_params_are_smaller() -> None:
    model = _model(VARIANTS["moe"])
    idx = torch.randint(0, BASE.vocab_size, (2, 8))
    logits, loss = model(idx, idx)
    plain = F.cross_entropy(logits.reshape(-1, BASE.vocab_size), idx.reshape(-1))
    assert model.aux_loss is not None
    assert torch.allclose(loss, plain + BASE.moe_aux_loss_coef * model.aux_loss)
    assert model.active_params() < model.num_params()


def test_post_training_adds_the_moe_balancing_loss() -> None:
    from src.models.transformer import Transformer
    from src.post_training.utils import moe_balance_loss

    model = _model(VARIANTS["moe"])
    logits, _ = model(torch.randint(0, BASE.vocab_size, (2, 8)))  # no targets, like the SFT trainer
    extra = moe_balance_loss(torch.nn.DataParallel(model))  # wrappers are looked through
    assert isinstance(extra, torch.Tensor) and model.aux_loss is not None
    assert torch.allclose(extra, BASE.moe_aux_loss_coef * model.aux_loss)
    extra.backward()
    assert model.blocks[0].mlp.router.weight.grad is not None
    assert moe_balance_loss(_model(BASE)) == 0.0  # dense modern model
    assert moe_balance_loss(Transformer(n_head=2, n_embed=16, context_length=8, vocab_size=32, N_BLOCKS=1)) == 0.0


def test_tied_embeddings_share_one_matrix() -> None:
    tied, untied = _model(BASE), _model(replace(BASE, tie_embeddings=False))
    assert tied.lm_head.weight is tied.token_embed.weight
    assert untied.num_params() - tied.num_params() == BASE.vocab_size * BASE.n_embed


@pytest.mark.parametrize("name", ["gqa", "moe"])
def test_gradient_checkpointing_gives_the_same_gradients(name: str) -> None:
    cfg = VARIANTS[name]
    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    grads = []
    for ckpt in (False, True):
        model = _model(cfg).train()
        model.gradient_checkpointing = ckpt
        _, loss = model(idx, idx)
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


@pytest.mark.parametrize("name", ["gqa", "mla", "window"])
def test_cached_and_uncached_generation_agree(name: str) -> None:
    model = _model(VARIANTS[name])
    prompt = torch.randint(0, BASE.vocab_size, (2, 5))
    a = model.generate(prompt, 20, top_k=1, use_cache=True)
    b = model.generate(prompt, 20, top_k=1, use_cache=False)
    assert torch.equal(a, b)


def test_generation_can_run_past_the_context_window() -> None:
    model = _model(BASE)
    out = model.generate(torch.zeros(1, 4, dtype=torch.long), max_new_tokens=3 * BASE.context_length)
    assert out.shape == (1, 4 + 3 * BASE.context_length)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModernConfig(n_embed=30, n_head=4)
    with pytest.raises(ValueError, match="n_kv_head"):
        ModernConfig(n_embed=32, n_head=4, n_kv_head=3)
    assert ModernConfig.from_mapping({"n_embed": 64, "n_head": 4, "lr": 1e-3}).n_embed == 64


def test_post_training_wrappers_accept_the_modern_model() -> None:
    from src.post_training.reward_model import RewardModel
    from src.post_training.rollout import compute_logprobs, generate_with_logprobs
    from src.post_training.value_head import TransformerWithValueHead

    model = _model(BASE)
    actor = TransformerWithValueHead(model)
    idx = torch.randint(0, BASE.vocab_size, (2, 8))
    logits, values = actor(idx)
    assert logits.shape == (2, 8, BASE.vocab_size) and values.shape == (2, 8)
    reward = RewardModel(_model(BASE))(idx, seq_lengths=torch.tensor([8, 5]))
    assert reward.shape == (2,)
    rb = generate_with_logprobs(model, idx[:, :4], max_new_tokens=6)
    lp, mask = compute_logprobs(model, rb.sequences, rb.response_mask, requires_grad=False)
    assert torch.allclose(rb.gen_logprobs[rb.response_mask[:, 4:]], lp[mask], atol=1e-4)

"""Speculative decoding and int8 weight quantization."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference.quantize import Int8Linear, model_size_bytes, quantize_int8
from src.inference.speculative import speculative_generate
from src.models.modern import ModernConfig, ModernTransformer
from src.models.transformer import Transformer


def _classic(seed: int, vocab: int = 13) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(n_head=2, n_embed=16, context_length=32, vocab_size=vocab, N_BLOCKS=1).eval()


def _greedy(model: nn.Module, idx: torch.Tensor, n: int) -> torch.Tensor:
    for _ in range(n):
        logits, _ = model(idx[:, -32:])
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return idx


def test_greedy_speculative_decoding_equals_greedy_target_decoding() -> None:
    target, draft = _classic(0), _classic(1)  # an unrelated, random draft
    prompt = torch.tensor([[1, 2, 3]])
    ours, stats = speculative_generate(target, draft, prompt, 20, k=4, greedy=True)
    assert torch.equal(ours, _greedy(target, prompt, 20))
    assert stats.generated == 20 and stats.target_calls <= 20


def test_a_perfect_draft_is_always_accepted() -> None:
    target = _classic(0)
    g = torch.Generator().manual_seed(0)
    _, stats = speculative_generate(target, copy.deepcopy(target), torch.tensor([[1, 2]]), 24, k=4, generator=g)
    assert stats.acceptance_rate > 0.95
    assert stats.tokens_per_target_call > 4  # k accepted guesses + 1 bonus token per target pass


def test_sampling_keeps_the_target_distribution_whatever_the_draft() -> None:
    target, draft = _classic(0, vocab=6), _classic(7, vocab=6)
    prompt = torch.tensor([[1, 4, 2]])
    with torch.no_grad():
        p_target = F.softmax(target(prompt)[0][0, -1], dim=-1)
        p_draft = F.softmax(draft(prompt)[0][0, -1], dim=-1)
    assert (p_target - p_draft).abs().max() > 0.05  # the draft really is different
    g = torch.Generator().manual_seed(0)
    counts = torch.zeros(6)
    n = 3000
    for _ in range(n):
        out, _ = speculative_generate(target, draft, prompt, 1, k=3, generator=g)
        counts[out[0, -1]] += 1
    assert (counts / n - p_target).abs().max() < 0.03


def test_int8_linear_is_accurate_and_four_times_smaller() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(256, 128)
    q = Int8Linear(linear)
    x = torch.randn(8, 256)
    with torch.no_grad():
        rel = (q(x) - linear(x)).norm() / linear(x).norm()
    assert rel < 0.01
    assert q.weight_int8.dtype == torch.int8
    assert model_size_bytes(q) < 0.3 * model_size_bytes(linear)


@pytest.mark.parametrize("arch", ["classic", "modern"])
def test_quantized_models_give_nearly_the_same_logits(arch: str) -> None:
    torch.manual_seed(0)
    if arch == "classic":
        model: nn.Module = Transformer(n_head=4, n_embed=64, context_length=32, vocab_size=100, N_BLOCKS=2).eval()
    else:
        model = ModernTransformer(ModernConfig(vocab_size=100, context_length=32, n_embed=64, n_head=4, n_blocks=2)).eval()
    idx = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        before = model(idx)[0]
        quantized = quantize_int8(copy.deepcopy(model))
        after = quantized(idx)[0]
    assert F.cosine_similarity(before.flatten(), after.flatten(), dim=0) > 0.999
    assert any(isinstance(m, Int8Linear) for m in quantized.modules())
    assert model_size_bytes(quantized) < model_size_bytes(model)
    if arch == "modern":  # the tied output layer is not quantized, so the tie survives
        assert quantized.lm_head.weight is quantized.token_embed.weight

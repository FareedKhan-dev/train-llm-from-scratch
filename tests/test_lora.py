"""LoRA adapters on both architectures."""

from __future__ import annotations

import pytest
import torch

from src.models.lora import LoRALinear, apply_lora, lora_parameter_count, merge_lora
from src.models.modern import ModernConfig, ModernTransformer
from src.models.transformer import Transformer


def _classic() -> Transformer:
    torch.manual_seed(0)
    return Transformer(n_head=4, n_embed=32, context_length=16, vocab_size=60, N_BLOCKS=2)


def _modern() -> ModernTransformer:
    torch.manual_seed(0)
    return ModernTransformer(ModernConfig(vocab_size=60, context_length=16, n_embed=32, n_head=4, n_blocks=2, n_kv_head=2))


@pytest.mark.parametrize("make", [_classic, _modern])
def test_lora_starts_as_the_identity_and_trains_few_params(make) -> None:
    model = make().eval()
    idx = torch.randint(0, 60, (2, 10))
    before = model(idx)[0]
    wrapped = apply_lora(model, rank=4, alpha=8)
    assert wrapped and all(isinstance(model.get_submodule(n), LoRALinear) for n in wrapped)
    assert torch.allclose(model(idx)[0], before)  # B = 0 at init: same function as before
    trainable, total = lora_parameter_count(model)
    assert 0 < trainable < 0.2 * total
    assert all(("lora_" in n) == p.requires_grad for n, p in model.named_parameters())


@pytest.mark.parametrize("make", [_classic, _modern])
def test_merge_gives_the_same_outputs(make) -> None:
    model = make().eval()
    apply_lora(model, rank=4, alpha=8)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=0.1)
    idx = torch.randint(0, 60, (2, 10))
    adapted = model(idx)[0]
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert torch.allclose(model(idx)[0], adapted, atol=1e-5)
    assert all(p.requires_grad for p in model.parameters())


def test_lora_learns_and_keeps_the_base_frozen() -> None:
    model = _classic()
    base_weight = model.attn_blocks[0].attn.heads[0].query.weight.detach().clone()
    apply_lora(model, rank=8, alpha=16)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    data = torch.randint(0, 60, (8, 11))
    first = None
    for _ in range(60):
        _, loss = model(data[:, :-1], data[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.85 * first
    assert torch.equal(model.attn_blocks[0].attn.heads[0].query.base.weight, base_weight)

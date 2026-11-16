"""
Gradient accumulation in SFT gives the gradient of one big batch.

Before 1.0.0 each micro-batch's masked mean was divided by grad_accum, so a token in a
micro-batch with few trained tokens weighed far more than a token in a full one: with the
batches below the accumulated gradient had a cosine of 0.66 with the real one.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.train_sft import accumulate_gradients
from src.models.transformer import Transformer
from src.post_training.distributed import DDPContext
from src.post_training.sft import count_targets, sft_loss


def _model() -> Transformer:
    torch.manual_seed(0)
    return Transformer(n_head=2, n_embed=16, context_length=16, vocab_size=50, N_BLOCKS=1)


def _grads(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def test_accumulated_gradient_equals_the_full_batch_gradient() -> None:
    gen = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 50, (2, 16), generator=gen)
    mask = torch.zeros(2, 16, dtype=torch.long)
    mask[0, -2:] = 1   # a short answer: 2 trained targets in this row
    mask[1, 1:] = 1    # a long one: 15 trained targets
    assert count_targets(mask) == 17

    full = _model()
    logits, _ = full(tokens)
    sft_loss(logits, tokens, mask).backward()

    accum = _model()
    ctx = DDPContext(rank=0, local_rank=0, world_size=1, device="cpu")
    micro = [(tokens[:1], mask[:1], 0), (tokens[1:], mask[1:], 0)]
    logged = accumulate_gradients(accum, micro, SimpleNamespace(amp_dtype="fp32"), ctx)

    g_full, g_accum = _grads(full), _grads(accum)
    assert torch.allclose(g_accum, g_full, rtol=1e-4, atol=1e-7)
    with torch.no_grad():
        assert abs(logged - float(sft_loss(full(tokens)[0], tokens, mask))) < 1e-5


def test_num_tokens_divides_the_summed_loss() -> None:
    logits = torch.randn(1, 4, 7)
    tokens = torch.randint(0, 7, (1, 4))
    mask = torch.tensor([[0, 1, 1, 1]])
    mean = sft_loss(logits, tokens, mask)
    assert torch.allclose(sft_loss(logits, tokens, mask, num_tokens=6), mean * 3 / 6)

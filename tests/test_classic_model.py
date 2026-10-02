"""The original (classic) Transformer: shapes, causality and forward_embedding."""

from __future__ import annotations

import torch

from src.models.transformer import Transformer


def _model(n_blocks: int = 3) -> Transformer:
    torch.manual_seed(0)
    return Transformer(n_head=4, n_embed=32, context_length=16, vocab_size=50, N_BLOCKS=n_blocks).eval()


def test_forward_shapes_loss_and_causality() -> None:
    model = _model()
    idx = torch.randint(0, 50, (2, 12))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 12, 50) and loss is not None
    changed = idx.clone()
    changed[:, 8] = (changed[:, 8] + 1) % 50
    with torch.no_grad():
        assert torch.allclose(model(changed)[0][:, :8], logits[:, :8], atol=1e-5)


def test_forward_embedding_works_with_several_blocks() -> None:
    """Regression: it used to feed 4x-wide MLP activations into the next block and crash."""
    model = _model(n_blocks=3)
    idx = torch.randint(0, 50, (2, 10))
    hidden, residual = model.forward_embedding(idx)
    assert hidden.shape == (2, 10, 4 * 32) and residual.shape == (2, 10, 32)
    # the residual is the last block's stream after attention, before its MLP
    with torch.no_grad():
        x = model._pre_attn_pass(idx)
        for block in model.attn_blocks[:-1]:
            x = block(x)
        last = model.attn_blocks[-1]
        expected = x + last.attn(last.ln1(x))
    assert torch.allclose(residual, expected, atol=1e-6)


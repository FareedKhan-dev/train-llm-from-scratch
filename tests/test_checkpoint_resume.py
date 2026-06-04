"""
Regression tests for the legacy pretraining checkpoint/resume helpers.

Run from the repo root:
    PYTHONPATH=. python tests/test_checkpoint_resume.py
"""

from __future__ import annotations

import tempfile

import torch

from scripts.train_transformer import (
    checkpoint_path,
    list_checkpoints,
    resolve_resume_path,
    restore_training_checkpoint,
    save_training_checkpoint,
    prune_old_checkpoints,
)
from src.models.transformer import Transformer


def _tiny_model():
    torch.manual_seed(0)
    return Transformer(n_head=2, n_embed=8, context_length=8, vocab_size=32, N_BLOCKS=1)


def _tiny_config():
    return {
        "t_lr": 1e-3,
        "t_lr_decayed": 1e-4,
        "t_lr_decay_step": 10,
        "device": "cpu",
    }


def test_checkpoint_round_trip_and_latest_resume():
    cfg = _tiny_config()
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["t_lr"])

    # Populate optimizer state so the round trip proves optimizer resume works too.
    idx = torch.randint(0, 32, (2, 4))
    _, loss = model(idx, idx)
    loss.backward()
    optimizer.step()

    losses = [3.0, 2.0, 1.0]
    with tempfile.TemporaryDirectory() as tmp:
        first = checkpoint_path(tmp, 2)
        second = checkpoint_path(tmp, 3)
        save_training_checkpoint(first, model, optimizer, cfg, losses, step=2)
        save_training_checkpoint(second, model, optimizer, cfg, losses + [0.5], step=3)
        prune_old_checkpoints(tmp, keep_last=1)

        remaining = list_checkpoints(tmp)
        assert remaining == [second]
        assert resolve_resume_path("latest", tmp) == second

        restored = _tiny_model()
        restored_optim = torch.optim.AdamW(restored.parameters(), lr=cfg["t_lr"])
        next_step, restored_losses = restore_training_checkpoint(
            second,
            restored,
            restored_optim,
            cfg,
            "cpu",
        )

        assert next_step == 4
        assert restored_losses == losses + [0.5]
        assert restored_optim.state_dict()["state"]
        for original, loaded in zip(model.parameters(), restored.parameters()):
            assert torch.allclose(original, loaded)


if __name__ == "__main__":
    test_checkpoint_round_trip_and_latest_resume()
    print("ALL CHECKPOINT RESUME TESTS PASSED")

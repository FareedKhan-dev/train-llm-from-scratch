"""
Checkpoint save/load regression tests.

Covers issue #36: a model trained with ``torch.compile`` (and/or DDP) used to save keys like
``_orig_mod.lm_head.weight``. Every later stage then failed to match the keys and quietly
started from random weights. These tests pin the fix from both sides: checkpoints are now saved
with clean keys, and older prefixed checkpoints still load.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from config.post_training_config import SFTConfig, smoke
from src.checkpoint import (
    load_checkpoint,
    load_model_weights,
    model_state_from_checkpoint,
    strip_wrapper_prefixes,
    unwrap_model,
)
from src.models.transformer import Transformer
from src.post_training.inference import load_model_from_ckpt
from src.post_training.reward_model import RewardModel, load_reward_model
from src.post_training.utils import build_model_from_config, load_backbone_from_ckpt, save_stage_ckpt


def _weights_equal(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.fixture()
def cfg() -> SFTConfig:
    return smoke(SFTConfig)


def test_strip_wrapper_prefixes_handles_any_nesting() -> None:
    state = {
        "module._orig_mod.lm_head.weight": 1,
        "_orig_mod.module.token_embed.weight": 2,
        "module.module.layer_norm.bias": 3,
        "clean.key": 4,
    }
    assert strip_wrapper_prefixes(state) == {
        "lm_head.weight": 1,
        "token_embed.weight": 2,
        "layer_norm.bias": 3,
        "clean.key": 4,
    }
    reward = {"module.transformer.lm_head.weight": 5, "module.reward_head.weight": 6}
    assert strip_wrapper_prefixes(reward, ("transformer.",)) == {
        "lm_head.weight": 5,
        "reward_head.weight": 6,
    }


def test_unwrap_model_removes_compile_and_data_parallel(cfg: SFTConfig) -> None:
    model = build_model_from_config(cfg)
    wrapped = torch.compile(nn.DataParallel(model))  # compile wraps lazily, no forward needed
    assert unwrap_model(wrapped) is model
    assert unwrap_model(model) is model


def test_compiled_model_checkpoint_round_trip(cfg: SFTConfig, tmp_path: str) -> None:
    """The exact failure from issue #36: save a compiled model, load it into a bare one."""
    model = build_model_from_config(cfg)
    path = os.path.join(tmp_path, "compiled.pt")
    save_stage_ckpt(path, torch.compile(model), None, stage="pretrain", cfg=cfg, step=0)

    saved_keys = load_checkpoint(path)["model_state_dict"].keys()
    assert not any(k.startswith(("_orig_mod.", "module.")) for k in saved_keys)

    assert _weights_equal(load_backbone_from_ckpt(cfg, path, "cpu"), model)
    assert _weights_equal(load_model_from_ckpt(path, "cpu"), model)


def test_old_prefixed_checkpoints_still_load(cfg: SFTConfig, tmp_path: str) -> None:
    """Files written before the fix carry ``module._orig_mod.`` keys; they must load as-is."""
    model = build_model_from_config(cfg)
    prefixed = {f"module._orig_mod.{k}": v for k, v in model.state_dict().items()}
    path = os.path.join(tmp_path, "old.pt")
    torch.save({"model_state_dict": prefixed, "cfg": cfg.__dict__}, path)

    assert _weights_equal(load_backbone_from_ckpt(cfg, path, "cpu"), model)
    assert _weights_equal(load_model_from_ckpt(path, "cpu"), model)


def test_reward_checkpoint_loads_for_chat_and_as_reward_model(cfg: SFTConfig, tmp_path: str) -> None:
    rm = RewardModel(build_model_from_config(cfg))
    with torch.no_grad():
        rm.reward_head.weight.normal_()
    path = os.path.join(tmp_path, "reward.pt")
    save_stage_ckpt(path, torch.compile(rm), None, stage="reward", cfg=cfg, step=0)

    # chat / eval read the backbone out of the reward checkpoint ("transformer." keys)
    assert _weights_equal(load_model_from_ckpt(path, "cpu"), rm.transformer)
    # PPO with reward_source="rm" restores backbone + head
    restored = load_reward_model(cfg, path, "cpu")
    assert torch.equal(restored.reward_head.weight, rm.reward_head.weight)


def test_legacy_checkpoint_dims_are_read_from_config(tmp_path: str) -> None:
    """Checkpoints from scripts/train_transformer.py store their sizes under ``config``."""
    model = Transformer(n_head=2, n_embed=16, context_length=8, vocab_size=64, N_BLOCKS=1)
    config = {"n_head": 2, "n_embed": 16, "context_length": 8, "vocab_size": 64, "n_blocks": 1}
    path = os.path.join(tmp_path, "legacy.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
    assert _weights_equal(load_model_from_ckpt(path, "cpu"), model)


def test_causal_masks_are_not_saved_and_old_checkpoints_with_them_still_load(tmp_path: str) -> None:
    """Every classic head used to save its (context x context) mask: ~1.6 GiB in the 400M config."""
    model = Transformer(n_head=4, n_embed=32, context_length=64, vocab_size=50, N_BLOCKS=2)
    assert not any(k.endswith(".tril") for k in model.state_dict())
    masks = {f"attn_blocks.{b}.attn.heads.{h}.tril": torch.tril(torch.ones(64, 64)) for b in range(2) for h in range(4)}
    path = os.path.join(tmp_path, "old_format.pt")
    torch.save({"model_state_dict": {**model.state_dict(), **masks}}, path)

    fresh = Transformer(n_head=4, n_embed=32, context_length=64, vocab_size=50, N_BLOCKS=2)
    fresh.load_state_dict(model_state_from_checkpoint(load_checkpoint(path)))  # strict, like resume does
    assert _weights_equal(fresh, model)


def test_missing_parameters_fail_loudly(cfg: SFTConfig) -> None:
    model = build_model_from_config(cfg)
    partial = {k: v for k, v in model.state_dict().items() if not k.startswith("lm_head")}
    with pytest.raises(RuntimeError, match="missing"):
        load_model_weights(build_model_from_config(cfg), partial, source="partial.pt")


def test_legacy_trainer_resumes_from_prefixed_checkpoint(tmp_path: str) -> None:
    from scripts.train_transformer import restore_training_checkpoint

    model = Transformer(n_head=2, n_embed=8, context_length=8, vocab_size=32, N_BLOCKS=1)
    prefixed = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    path = os.path.join(tmp_path, "checkpoint_step_00000004.pt")
    torch.save({"model_state_dict": prefixed, "last_completed_step": 4, "losses": [1.0]}, path)

    fresh = Transformer(n_head=2, n_embed=8, context_length=8, vocab_size=32, N_BLOCKS=1)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    cfg = {"t_lr": 1e-3, "t_lr_decayed": 1e-4, "t_lr_decay_step": 10}
    next_step, losses = restore_training_checkpoint(path, fresh, fresh_opt, cfg, "cpu")
    assert next_step == 5 and losses == [1.0]
    assert _weights_equal(fresh, model)

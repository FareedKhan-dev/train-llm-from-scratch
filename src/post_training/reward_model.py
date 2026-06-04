"""
Reward model: a scalar reward head on top of a :class:`Transformer` backbone.

Following InstructGPT, the reward of a full sequence is read off the hidden state of its
**last real token**. We train it with the Bradley-Terry pairwise loss (see
``reward_train.py``) on (chosen, rejected) preference pairs. The trained RM then provides
the scalar reward signal for PPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import torch as _torch

from src.models.transformer import Transformer
from src.post_training.utils import build_model_from_config, gather_last, unwrap


def load_reward_model(cfg, ckpt_path: str, device: str) -> "RewardModel":
    """Reconstruct a trained :class:`RewardModel` (backbone + reward head) from a reward
    checkpoint saved by ``scripts/train_reward.py``."""
    backbone = build_model_from_config(cfg)
    rm = RewardModel(backbone)
    ck = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    rm.load_state_dict(state, strict=True)
    rm.to(device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    return rm


class RewardModel(nn.Module):
    """Wrap a :class:`Transformer` and add a scalar reward head (no ``lm_head`` used)."""

    def __init__(self, transformer: Transformer) -> None:
        super().__init__()
        self.transformer = transformer
        n_embed = transformer.lm_head.in_features
        self.reward_head = nn.Linear(n_embed, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)  # start near-zero rewards

    @property
    def context_length(self) -> int:
        return self.transformer.context_length

    def token_rewards(self, idx: torch.Tensor) -> torch.Tensor:
        """Per-token scalar reward (B, T) -- useful for diagnostics / dense shaping."""
        hidden = self.transformer.forward_hidden(idx)
        return self.reward_head(hidden).squeeze(-1)

    def forward(self, idx: torch.Tensor, seq_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Scalar reward per sequence (B,).

        Args:
            idx: (B, T) token ids.
            seq_lengths: (B,) number of real (non-pad) tokens per row. If None, uses the
                full length T (assumes no padding).
        """
        rewards = self.token_rewards(idx)  # (B, T)
        if seq_lengths is None:
            return rewards[:, -1]
        return gather_last(rewards, seq_lengths)

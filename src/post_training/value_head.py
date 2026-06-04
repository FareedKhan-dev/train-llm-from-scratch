"""
Actor-critic wrapper for PPO.

PPO needs a per-token *value* estimate V(s_t) alongside the policy logits. We get both
from a single backbone: the wrapper reuses the :class:`Transformer`'s ``forward_hidden``
and ``lm_head`` for the policy, and adds a small scalar value head on top of the same
hidden states. Sharing the backbone is the standard, memory-cheap choice; the value head
is initialized to output ~0 so early training is stable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.transformer import Transformer


class TransformerWithValueHead(nn.Module):
    """Wrap a :class:`Transformer` and add a scalar value head for PPO.

    ``forward(idx)`` returns ``(logits, values)`` where ``logits`` are the policy logits
    (B, T, vocab) and ``values`` are per-token value estimates (B, T).
    """

    def __init__(self, transformer: Transformer) -> None:
        super().__init__()
        self.transformer = transformer
        n_embed = transformer.lm_head.in_features
        self.value_head = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, 1),
        )
        # Start with values ~0 so the critic does not destabilize the policy early on.
        last = self.value_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    # Expose the backbone's context length so rollout's cap detection works.
    @property
    def context_length(self) -> int:
        return self.transformer.context_length

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.transformer.forward_hidden(idx)        # (B, T, n_embed)
        logits = self.transformer.lm_head(hidden)            # (B, T, vocab)
        values = self.value_head(hidden).squeeze(-1)         # (B, T)
        return logits, values

    @torch.no_grad()
    def value_only(self, idx: torch.Tensor) -> torch.Tensor:
        """Per-token values without computing the (large) vocab logits -- used when
        scoring rollouts where only V(s_t) is needed."""
        hidden = self.transformer.forward_hidden(idx)
        return self.value_head(hidden).squeeze(-1)

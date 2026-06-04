"""
Prompt iterator for RL (PPO / GRPO). Reads a JSONL of ``{"prompt", "gold"}`` (gold is the
verifiable numeric answer, or null) and yields lists of sampled rows. Rows are sharded
across ranks so each GPU optimizes on its own prompts.
"""

from __future__ import annotations

import json
from typing import Iterator

import numpy as np


def load_prompt_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_prompt_iterator(
    path: str,
    prompts_per_iter: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
) -> Iterator[list[dict]]:
    """Infinitely yield lists of ``prompts_per_iter`` rows (this rank's shard)."""
    rows = load_prompt_rows(path)[rank::world_size]
    rng = np.random.default_rng(seed + rank)
    n = len(rows)
    while True:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        for s in range(0, n - prompts_per_iter + 1, prompts_per_iter):
            yield [rows[i] for i in idx[s:s + prompts_per_iter]]

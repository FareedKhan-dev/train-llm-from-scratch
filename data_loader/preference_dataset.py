"""
Batch iterator over preference pairs for reward-model and DPO training.

Reads a JSONL file of ``{"prompt", "chosen", "rejected"}`` (produced by
``scripts/prepare_preference_data.py``). Each side is rendered through the chat template
so we get, for the chosen and rejected responses to the same prompt:
  - token ids of ``prompt + response + EOT``
  - a response mask (1 over the completion, used by DPO)
  - the true sequence length (used by the reward model to read the last-token reward)

Right-padding is safe here because the model's attention is causal: the last real token
never attends to padding that comes after it, and the response mask zeros padded
positions in the loss.
"""

from __future__ import annotations

import json
from typing import Iterator

import numpy as np
import torch

from src.post_training.chat_template import EOT_ID, encode_chat


def _encode_side(prompt: str, response: str, max_len: int) -> tuple[list[int], list[int]]:
    ids, mask = encode_chat([{"role": "user", "content": prompt},
                             {"role": "assistant", "content": response}])
    return ids[:max_len], mask[:max_len]


def _collate(rows: list[dict], max_len: int, device: str) -> dict:
    enc = [( _encode_side(r["prompt"], r["chosen"], max_len),
             _encode_side(r["prompt"], r["rejected"], max_len)) for r in rows]
    # Pad chosen and rejected to a single common length so they can share one forward.
    L = max(max(len(c[0]), len(j[0])) for c, j in enc)

    def pad(seq, fill):
        return seq + [fill] * (L - len(seq))

    ch_ids, ch_mask, ch_len, rj_ids, rj_mask, rj_len = [], [], [], [], [], []
    for (cids, cmask), (jids, jmask) in enc:
        ch_len.append(len(cids)); rj_len.append(len(jids))
        ch_ids.append(pad(cids, EOT_ID)); ch_mask.append(pad(cmask, 0))
        rj_ids.append(pad(jids, EOT_ID)); rj_mask.append(pad(jmask, 0))

    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    return {
        "chosen_ids": t(ch_ids, torch.long), "chosen_mask": t(ch_mask, torch.long), "chosen_len": t(ch_len, torch.long),
        "rejected_ids": t(rj_ids, torch.long), "rejected_mask": t(rj_mask, torch.long), "rejected_len": t(rj_len, torch.long),
    }


def get_preference_iterator(
    path: str,
    batch_size: int,
    max_len: int,
    device: str = "cpu",
    *,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    infinite: bool = True,
) -> Iterator[dict]:
    """Yield collated preference batches (dict of tensors). Rows are sharded across ranks."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[rank::world_size]
    rng = np.random.default_rng(7 + rank)
    while True:
        order = np.arange(len(rows))
        if shuffle:
            rng.shuffle(order)
        for s in range(0, len(order) - batch_size + 1, batch_size):
            batch = [rows[i] for i in order[s:s + batch_size]]
            yield _collate(batch, max_len, device)
        if not infinite:
            return

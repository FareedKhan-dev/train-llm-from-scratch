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
from itertools import zip_longest
from typing import Iterator

import numpy as np
import torch

from src.post_training.chat_template import EOT_ID, encode_chat


def _encode_response(response: str) -> list[int]:
    """Encode assistant content plus its EOT, without duplicating the role header."""
    ids, mask = encode_chat([{"role": "assistant", "content": response}])
    start = mask.index(1)  # non-empty responses and the EOT always provide a target
    return ids[start:]


def _encode_pair(prompt: str, chosen: str, rejected: str, max_len: int):
    """Truncate a pair over one shared prompt while retaining response targets."""
    prompt_ids, _ = encode_chat(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
    )
    assistant_header, _ = encode_chat([], add_generation_prompt=True)
    chosen_ids = _encode_response(chosen)
    rejected_ids = _encode_response(rejected)

    difference_at = next(
        (i for i, pair in enumerate(zip_longest(chosen_ids, rejected_ids)) if pair[0] != pair[1]),
        None,
    )
    if difference_at is None:
        raise ValueError("chosen and rejected responses encode identically")

    # Keep the complete assistant header plus the prompt's EOT and at least one prompt
    # token. Preserve both responses in full when they fit beside that shared context;
    # otherwise shorten the prompt only enough to retain the first differing target.
    min_shared_prompt = min(len(prompt_ids), len(assistant_header) + 2)
    required_response = difference_at + 1
    max_response_with_context = max_len - min_shared_prompt
    if required_response > max_response_with_context:
        raise ValueError(
            f"max_len={max_len} cannot retain shared prompt context and distinguish responses"
        )
    longest_response = max(len(chosen_ids), len(rejected_ids))
    reserved_response = (
        longest_response if longest_response <= max_response_with_context else required_response
    )
    prompt_budget = min(len(prompt_ids), max_len - reserved_response)
    shared_prompt = prompt_ids[-prompt_budget:]
    response_budget = max_len - len(shared_prompt)

    def side(response_ids: list[int]) -> tuple[list[int], list[int]]:
        response_ids = response_ids[:response_budget]
        ids = shared_prompt + response_ids
        mask = [0] * len(shared_prompt) + [1] * len(response_ids)
        return ids, mask

    return side(chosen_ids), side(rejected_ids)


def _collate(rows: list[dict], max_len: int, device: str) -> dict:
    enc = [_encode_pair(r["prompt"], r["chosen"], r["rejected"], max_len) for r in rows]
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

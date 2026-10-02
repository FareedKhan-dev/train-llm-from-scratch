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

Truncation (when ``prompt + response`` does not fit in ``max_len``) follows the usual
recipe for preference training: both sides share one left-truncated prompt, so the model
always compares the two answers under the same context, and the prompt never takes more
than half of the window when the answers are long too. See :func:`encode_preference_pair`.
"""

from __future__ import annotations

import json
from itertools import zip_longest
from typing import Iterator

import numpy as np
import torch

from src.post_training.chat_template import EOT_ID, encode_chat

Encoded = tuple[list[int], list[int]]  # (token ids, response mask)


def _encode_response(response: str) -> list[int]:
    """Encode assistant content plus its EOT, without duplicating the role header."""
    ids, mask = encode_chat([{"role": "assistant", "content": response}])
    start = mask.index(1)  # non-empty responses and the EOT always provide a target
    return ids[start:]


def _first_difference(a: list[int], b: list[int]) -> int | None:
    """Index of the first token where ``a`` and ``b`` differ (None if they are identical)."""
    return next((i for i, (x, y) in enumerate(zip_longest(a, b)) if x != y), None)


def encode_preference_pair(prompt: str, chosen: str, rejected: str, max_len: int) -> tuple[Encoded, Encoded]:
    """
    Encode ``prompt + chosen`` and ``prompt + rejected`` into at most ``max_len`` tokens each.

    The rules, in order:

    1. Both sides share the same prompt tokens, so DPO and the reward model compare the two
       answers under exactly the same context.
    2. If everything fits, nothing is cut.
    3. Otherwise the prompt is cut from the left (its end, including the ``<|assistant|>``
       header, is what matters for the answer), but it keeps at least half of the window
       when the answers are long as well. The answers are cut from the right.
    4. The cut always keeps the first token where the two answers differ, when that is
       possible. A pair that can no longer be told apart carries no learning signal, so it
       just gets identical sides (zero DPO gradient) instead of crashing the run.
    """
    prompt_ids, _ = encode_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)
    header_len = len(encode_chat([], add_generation_prompt=True)[0])
    if max_len <= header_len + 1:
        raise ValueError(f"max_len={max_len} is too small to hold the chat template; use at least 16")
    chosen_ids, rejected_ids = _encode_response(chosen), _encode_response(rejected)
    longest = max(len(chosen_ids), len(rejected_ids))

    # The model must always see the assistant header plus one prompt token.
    min_prompt = min(len(prompt_ids), header_len + 1)
    prompt_keep = min(len(prompt_ids), max(max_len - longest, max_len // 2))
    diff = _first_difference(chosen_ids, rejected_ids)
    if diff is not None:
        prompt_keep = min(prompt_keep, max_len - (diff + 1))
    prompt_keep = max(prompt_keep, min_prompt)

    shared = prompt_ids[len(prompt_ids) - prompt_keep:]
    budget = max(0, max_len - len(shared))

    def side(response_ids: list[int]) -> Encoded:
        kept = response_ids[:budget]
        return shared + kept, [0] * len(shared) + [1] * len(kept)

    return side(chosen_ids), side(rejected_ids)


# Name used by the original fix in #41; kept so existing imports keep working.
_encode_pair = encode_preference_pair


def _collate(rows: list[dict], max_len: int, device: str) -> dict:
    enc = [encode_preference_pair(r["prompt"], r["chosen"], r["rejected"], max_len) for r in rows]
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

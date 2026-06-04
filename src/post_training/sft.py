"""
Supervised Fine-Tuning (SFT) core: a prompt-masked language-modeling loss and a
sequence-packing utility.

The only difference from ordinary next-token pretraining is the per-token ``loss_mask``:
we train the model to produce the *assistant* tokens (mask=1) but not to reproduce the
prompt / role markers (mask=0). The mask comes from
:func:`src.post_training.chat_template.encode_chat`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def sft_loss(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    Masked next-token cross-entropy.

    Args:
        logits:    (B, T, V) model logits for ``tokens``.
        tokens:    (B, T) packed token ids.
        loss_mask: (B, T) 1 on tokens that should be predicted (assistant completions).

    Returns:
        Scalar loss averaged over masked target positions.
    """
    # Predict token t+1 from position t (same shift the base model uses).
    logits = logits[:, :-1, :]
    targets = tokens[:, 1:]
    mask = loss_mask[:, 1:].to(logits.dtype)

    V = logits.size(-1)
    ce = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1).long(), reduction="none")
    ce = ce.view(targets.shape) * mask
    return ce.sum() / mask.sum().clamp(min=1.0)


def pack_examples(
    examples: list[tuple[list[int], list[int]]],
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack a list of ``(ids, loss_mask)`` examples into fixed-length ``context_length``
    rows by concatenating them and slicing. Examples are already EOT-terminated by the
    chat template, so the EOT acts as the separator between packed examples. Trailing
    tokens that don't fill a full row are dropped.

    Returns:
        (tokens, masks) int arrays of shape (N, context_length).
    """
    flat_ids: list[int] = []
    flat_mask: list[int] = []
    for ids, mask in examples:
        flat_ids.extend(ids)
        flat_mask.extend(mask)

    n_rows = len(flat_ids) // context_length
    ids_arr = np.asarray(flat_ids[: n_rows * context_length], dtype=np.int32).reshape(n_rows, context_length)
    mask_arr = np.asarray(flat_mask[: n_rows * context_length], dtype=np.int8).reshape(n_rows, context_length)
    return ids_arr, mask_arr

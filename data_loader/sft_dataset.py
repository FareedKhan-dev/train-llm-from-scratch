"""
Batch iterator for packed SFT data (mirrors ``data_loader.data_loader.get_batch_iterator``
but carries the per-token loss mask alongside the tokens).

The HDF5 file has two aligned datasets, ``tokens`` and ``loss_mask``, both shape
(N, context_length), produced by ``scripts/prepare_sft_data.py``.
"""

from __future__ import annotations

from typing import Iterator

import h5py
import numpy as np
import torch


def get_sft_batch_iterator(
    data_path: str,
    batch_size: int,
    device: str = "cpu",
    *,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    infinite: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Yield ``(tokens, loss_mask, epoch)`` batches from a packed SFT HDF5 file.

    Rows are sharded across ranks (each rank sees a disjoint stride) so DDP data-parallel
    training covers the dataset once per epoch. Set ``infinite=False`` to stop after one
    pass (used for evaluation).
    """
    with h5py.File(data_path, "r") as f:
        tokens = f["tokens"]
        masks = f["loss_mask"]
        n = tokens.shape[0]
        idxs = np.arange(rank, n, world_size)
        epoch = 0
        rng = np.random.default_rng(1234 + rank)
        while True:
            if shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs) - batch_size + 1, batch_size):
                batch = np.sort(idxs[start: start + batch_size])  # sorted for h5py fancy-index
                tk = torch.tensor(np.asarray(tokens[batch]), dtype=torch.long, device=device)
                mk = torch.tensor(np.asarray(masks[batch]), dtype=torch.long, device=device)
                yield tk, mk, epoch
            epoch += 1
            if not infinite:
                return

"""
Named sizes for ``scripts/train_transformer.py --preset NAME``.

Each preset overrides the matching constants of ``config/config.py`` (same key names), so a
preset is just "the config you would have typed by hand". Pick one by hardware:

=========  ===============================  =========================================
preset     data                             meant for
=========  ===============================  =========================================
tiny       data/tiny (TinyStories + BPE)    any laptop CPU, a few minutes
student    data/tiny                        a laptop CPU, about half an hour
small      data/tiny                        a fast CPU or any GPU
13m        data/train, data/val (the Pile)  the 13M model from the README, one GPU
77m        the Pile                         the README's 77M base, one GPU
3b         the Pile                         the original default of config.py (big GPU)
=========  ===============================  =========================================

The CPU presets train on the TinyStories data made by ``scripts/prepare_tiny_data.py``. Its
small BPE vocabulary (4096 by default) is read from the data file at startup, so the vocab
size here is only the fallback for data without that information.

Every preset trains on windows as long as the model's context (``t_context_length ==
context_length``). The original config trains on 16-token windows, which leaves most of
the position embeddings untrained; see the docs for why that matters for generation.
"""

from __future__ import annotations

from typing import Any

TINY_TRAIN = "data/tiny/train.h5"
TINY_DEV = "data/tiny/val.h5"
PILE_TRAIN = "data/train/pile_train.h5"
PILE_DEV = "data/val/pile_dev.h5"


def _cpu(n_embed: int, n_head: int, n_blocks: int, context: int, steps: int, batch: int, lr: float, name: str) -> dict[str, Any]:
    return {
        "vocab_size": 4096,
        "context_length": context,
        "n_embed": n_embed,
        "n_head": n_head,
        "n_blocks": n_blocks,
        "train_path": TINY_TRAIN,
        "dev_path": TINY_DEV,
        "t_batch_size": batch,
        "t_context_length": context,
        "t_train_steps": steps,
        "t_eval_steps": max(100, steps // 10),
        "t_eval_iters": 20,
        "t_lr": lr,
        "t_lr_decayed": lr / 10,
        "t_lr_decay_step": int(steps * 0.8),  # constant, then a 10x drop for the last 20%
        "t_out_path": f"models/{name}.pt",
        "sample_prompt": "Once upon a time",
    }


PRESETS: dict[str, dict[str, Any]] = {
    "tiny": _cpu(n_embed=64, n_head=4, n_blocks=2, context=128, steps=1500, batch=32, lr=2e-3, name="tiny"),
    "student": _cpu(n_embed=192, n_head=6, n_blocks=4, context=256, steps=4000, batch=32, lr=1e-3, name="student"),
    "small": _cpu(n_embed=384, n_head=6, n_blocks=6, context=256, steps=8000, batch=32, lr=6e-4, name="small"),
    # The GPU sizes used in the README, on the Pile data from scripts/data_preprocess.py.
    "13m": {
        "vocab_size": 50304, "context_length": 128, "n_embed": 128, "n_head": 8, "n_blocks": 1,
        "train_path": PILE_TRAIN, "dev_path": PILE_DEV, "t_batch_size": 64, "t_context_length": 128,
        "t_train_steps": 20000, "t_eval_steps": 1000, "t_eval_iters": 100, "t_lr": 5e-4,
        "t_lr_decayed": 5e-5, "t_lr_decay_step": 16000, "t_out_path": "models/13m.pt",
    },
    "77m": {
        "vocab_size": 50304, "context_length": 512, "n_embed": 512, "n_head": 8, "n_blocks": 8,
        "train_path": PILE_TRAIN, "dev_path": PILE_DEV, "t_batch_size": 24, "t_context_length": 512,
        "t_train_steps": 20000, "t_eval_steps": 1000, "t_eval_iters": 100, "t_lr": 3e-4,
        "t_lr_decayed": 3e-5, "t_lr_decay_step": 16000, "t_out_path": "models/77m.pt",
    },
    # The original default of config/config.py (about 3B parameters), kept for reference.
    "3b": {
        "vocab_size": 50304, "context_length": 512, "n_embed": 2048, "n_head": 16, "n_blocks": 64,
        "train_path": PILE_TRAIN, "dev_path": PILE_DEV, "t_batch_size": 32, "t_context_length": 16,
        "t_train_steps": 200000, "t_eval_steps": 1000, "t_eval_iters": 250, "t_lr": 5e-4,
        "t_lr_decayed": 5e-5, "t_lr_decay_step": 50000, "t_out_path": "models/transformer_B.pt",
    },
}


def apply_preset(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a copy of ``config`` with preset ``name`` applied."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose one of {sorted(PRESETS)}")
    return {**config, **PRESETS[name]}

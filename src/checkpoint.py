"""
Checkpoint helpers shared by every training and inference script.

A model is often wrapped before training: ``DistributedDataParallel`` stores it under
``.module`` and ``torch.compile`` stores it under ``._orig_mod``. Calling ``state_dict()`` on a
wrapper bakes those names into every key (``module._orig_mod.attn_blocks.0...``), and a bare
model then fails to load the file. The two helpers below solve this from both sides:

- :func:`unwrap_model` returns the plain module, so checkpoints are saved with clean keys.
- :func:`strip_wrapper_prefixes` cleans keys of checkpoints that were saved from a wrapper,
  so older files still load.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from typing import Any, TypeVar

import torch
import torch.nn as nn

V = TypeVar("V")

# Prefixes added by DistributedDataParallel / DataParallel and by torch.compile.
WRAPPER_PREFIXES: tuple[str, ...] = ("module.", "_orig_mod.")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the plain module behind any mix of DDP / DataParallel / torch.compile wrappers."""
    while True:
        inner = getattr(model, "_orig_mod", None)  # torch.compile's OptimizedModule
        if isinstance(inner, nn.Module):
            model = inner
            continue
        if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
            model = model.module
            continue
        return model


def strip_wrapper_prefixes(
    state_dict: Mapping[str, V], extra_prefixes: tuple[str, ...] = ()
) -> dict[str, V]:
    """
    Remove leading wrapper prefixes from every key, in any order and any nesting depth.

    ``module._orig_mod.lm_head.weight`` and ``_orig_mod.module.lm_head.weight`` both become
    ``lm_head.weight``. Pass ``extra_prefixes`` to strip more, for example ``("transformer.",)``
    to read the backbone out of a reward-model checkpoint.
    """
    prefixes = WRAPPER_PREFIXES + tuple(extra_prefixes)
    cleaned: dict[str, V] = {}
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def load_checkpoint(path: str | os.PathLike[str], map_location: Any = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict on any PyTorch version (our files hold metadata, not just tensors)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 1.13 has no weights_only argument
        return torch.load(path, map_location=map_location)


def model_state_from_checkpoint(
    checkpoint: Mapping[str, Any], extra_prefixes: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Return the model weights of a checkpoint with wrapper prefixes removed.

    Accepts both our checkpoint dicts (weights under ``model_state_dict``) and raw state dicts.
    """
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    return strip_wrapper_prefixes(state, extra_prefixes)


def model_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return the model settings stored in a checkpoint.

    Post-training stages save them under ``cfg`` and the legacy trainer under ``config``.
    Very old checkpoints have neither, which gives an empty dict.
    """
    cfg = checkpoint.get("cfg") or checkpoint.get("config") or {}
    return dict(cfg)


def load_model_weights(
    model: nn.Module, state: Mapping[str, torch.Tensor], *, source: str = "checkpoint"
) -> None:
    """
    Copy matching weights from ``state`` into ``model``.

    Extra keys are ignored (a reward or value head saved next to the backbone, for example).
    A missing *parameter* is an error: silently keeping random weights is how a broken
    checkpoint turns into a training run that quietly starts from scratch. Missing buffers
    (the causal mask, position indices) are fine, because the model rebuilds them on init.
    """
    own_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in own_keys}
    missing, _ = model.load_state_dict(filtered, strict=False)
    param_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    missing_params = [k for k in missing if k in param_names]
    if missing_params:
        raise RuntimeError(
            f"{source} is missing {len(missing_params)} model parameters "
            f"(for example {missing_params[:3]}). Check that the checkpoint was saved from a "
            "model with the same architecture and size."
        )


def atomic_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """``torch.save`` to a temporary file first, then rename, so a crash never leaves half a file."""
    path = os.fspath(path)
    target_dir = os.path.dirname(path) or "."
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target_dir, prefix=f".{os.path.basename(path)}.", suffix=".tmp", delete=False
    ) as tmp_file:
        tmp_path = tmp_file.name
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

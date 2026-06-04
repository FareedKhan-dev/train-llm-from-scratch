"""
JSON config loader for the post-training stages.

Industry-standard, learning-friendly: each stage's knobs live in a small editable JSON file
under ``configs/`` (typed by the dataclasses in :mod:`config.post_training_config`). This
loader resolves a final dataclass instance by merging four layers, lowest precedence first:

    1. dataclass field defaults        (config.post_training_config.<Stage>Config)
    2. configs/base.json               (shared model + runtime fields)
    3. the stage JSON (configs/sft.json, ...)   (that stage's hyperparameters)
    4. CLI --field overrides           (highest precedence)

JSON ``null`` maps to Python ``None`` cleanly -- the right way to set ``str | None`` fields
like ``amp_dtype``. Unknown keys are warned about, not fatal.

When ``json_path`` lives in a sub-dir with its own ``base.json`` (e.g. ``configs/smoke/sft.json``),
that sibling ``base.json`` is used automatically -- so the smoke configs shrink the model too.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from typing import Any


def _deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge ``src`` into ``dst`` (nested-dict aware; future-proof)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _resolve_base(json_path: str | None, base_path: str | None) -> str:
    if base_path is not None:
        return base_path
    if json_path:
        sibling = os.path.join(os.path.dirname(json_path), "base.json")
        if os.path.exists(sibling):
            return sibling
    return "configs/base.json"


def load_config(
    cfg_cls,
    json_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    base_path: str | None = None,
):
    """
    Resolve ``cfg_cls`` from ``base.json`` + the stage JSON + CLI overrides.

    Args:
        cfg_cls: the stage dataclass (e.g. ``SFTConfig``).
        json_path: path to the stage JSON (e.g. ``configs/sft.json``); None = base + defaults.
        overrides: parsed CLI ``--field`` values (None values are ignored).
        base_path: shared base JSON; if None, uses the sibling ``base.json`` of ``json_path``
            (so ``configs/smoke/sft.json`` picks up ``configs/smoke/base.json``), else
            ``configs/base.json``.

    Returns:
        an instance of ``cfg_cls`` with the resolved values.
    """
    base = _resolve_base(json_path, base_path)
    merged: dict[str, Any] = {}
    for path in (base, json_path):
        if path and os.path.exists(path):
            with open(path) as fh:
                _deep_merge(merged, json.load(fh))

    field_names = {f.name for f in fields(cfg_cls)}
    for key in list(merged):
        if key not in field_names:
            print(f"[config] ignoring unknown key '{key}' for {cfg_cls.__name__}")
            merged.pop(key)

    if overrides:
        merged.update({k: v for k, v in overrides.items() if k in field_names and v is not None})

    return cfg_cls(**merged)

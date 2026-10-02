"""
Typed JSON config loader for the post-training stages.

Each stage's knobs live in a small editable JSON file under ``configs/`` (typed by the
dataclasses in :mod:`config.post_training_config`). This loader resolves a final dataclass
instance by merging four layers, lowest precedence first:

    1. dataclass field defaults        (config.post_training_config.<Stage>Config)
    2. configs/base.json               (shared model + runtime fields)
    3. the stage JSON (configs/sft.json, ...)   (that stage's hyperparameters)
    4. CLI --field overrides           (highest precedence)

Every value is checked against the type annotation of its field, so mistakes fail early
and say where they came from:

    ConfigError: configs/dpo.json: loss_type must be one of ['dpo', 'ipo', ...], got 'dop'
    ConfigError: configs/sft.json: unknown key 'learning_rate' (did you mean 'lr'?)

Conversions are the ones you would expect: JSON ``null`` (or the CLI strings ``none``/``null``)
becomes ``None`` for optional fields, ``"1e-5"`` becomes a float, ``2000.0`` becomes an int,
``"true"``/``"false"`` become booleans, and ``$VARS`` in strings are expanded. Keys that start
with an underscore (``"_comment"``) are ignored, so JSON files can carry notes.

When ``json_path`` lives in a sub-dir with its own ``base.json`` (e.g. ``configs/smoke/sft.json``),
that sibling ``base.json`` is used automatically, so the smoke configs shrink the model too.
"""

from __future__ import annotations

import difflib
import json
import os
import types
from dataclasses import fields
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

from config.post_training_config import ConfigError

T = TypeVar("T")

_TRUE = {"true", "1", "yes", "y", "on"}
_FALSE = {"false", "0", "no", "n", "off"}
_NONE = {"none", "null"}


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
        sibling = os.path.join(os.path.dirname(json_path), "base.json").replace(os.sep, "/")  # same slashes as json_path
        if os.path.exists(sibling):
            return sibling
    return "configs/base.json"


def type_name(annotation: Any) -> str:
    """Short human name of a field type, for help texts and error messages."""
    if get_origin(annotation) is Literal:
        return "{" + ",".join(str(a) for a in get_args(annotation)) + "}"
    if get_origin(annotation) in (Union, types.UnionType):
        return " | ".join(type_name(a) for a in get_args(annotation))
    return "None" if annotation is type(None) else getattr(annotation, "__name__", str(annotation))


def coerce(value: Any, annotation: Any, name: str = "value") -> Any:
    """Convert ``value`` to ``annotation`` (bool/int/float/str/None/Literal/unions) or raise."""
    origin = get_origin(annotation)
    if origin is Literal:
        options = get_args(annotation)
        if value in options:
            return value
        raise ConfigError(f"{name} must be one of {list(options)}, got {value!r}")
    if origin in (Union, types.UnionType):
        args = get_args(annotation)
        if type(None) in args and (value is None or (isinstance(value, str) and value.strip().lower() in _NONE)):
            return None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return coerce(value, arg, name)
            except ConfigError:
                continue
        raise ConfigError(f"{name} must be {type_name(annotation)}, got {value!r}")
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
        raise ConfigError(f"{name} must be true or false, got {value!r}")
    if annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                number = float(value.replace("_", ""))
            except ValueError:
                number = float("nan")
            if number.is_integer():
                return int(number)
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if annotation is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace("_", ""))
            except ValueError:
                pass
        raise ConfigError(f"{name} must be a number, got {value!r}")
    if annotation is str:
        if isinstance(value, str):
            return os.path.expandvars(value)
        raise ConfigError(f"{name} must be a string, got {value!r}")
    return value


def field_types(cfg_cls: type) -> dict[str, Any]:
    """Resolved type of every dataclass field (works with ``from __future__ import annotations``)."""
    hints = get_type_hints(cfg_cls)
    return {f.name: hints[f.name] for f in fields(cfg_cls)}


def load_config(
    cfg_cls: type[T],
    json_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    base_path: str | None = None,
) -> T:
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
        an instance of ``cfg_cls`` with the resolved, type-checked values.

    Raises:
        ConfigError: unknown key, wrong type, or a value that fails the dataclass checks.
    """
    types_by_field = field_types(cfg_cls)
    base = _resolve_base(json_path, base_path)
    merged: dict[str, Any] = {}
    source: dict[str, str] = {}
    for path in (base, json_path):
        if not (path and os.path.exists(path)):
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a JSON object at the top level")
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if key not in types_by_field:
                hint = difflib.get_close_matches(key, types_by_field, n=1)
                suggestion = f" (did you mean '{hint[0]}'?)" if hint else ""
                raise ConfigError(f"{path}: unknown key '{key}' for {cfg_cls.__name__}{suggestion}")
            merged[key] = value
            source[key] = path

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key not in types_by_field:
            raise ConfigError(f"command line: unknown option --{key} for {cfg_cls.__name__}")
        merged[key] = value
        source[key] = "command line"

    values = {}
    for key, value in merged.items():
        try:
            values[key] = coerce(value, types_by_field[key], key)
        except ConfigError as e:
            raise ConfigError(f"{source[key]}: {e}") from None
    try:
        return cfg_cls(**values)
    except ConfigError as e:
        origin = ", ".join(sorted({source[k] for k in values})) or "defaults"
        raise ConfigError(f"{e} (from {origin})") from None

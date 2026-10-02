"""
Shared CLI helper: turn any stage config dataclass into ``--field value`` arguments so
every training script can override hyperparameters without bespoke argparse blocks.

Values arrive from the command line as strings and are converted with the same typed rules
as the JSON files (see :func:`config.loader.coerce`), so ``--n_kv_head 4`` becomes the int 4,
``--amp_dtype none`` becomes ``None`` and ``--loss_type dop`` is rejected with the valid choices.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, asdict, fields
from types import SimpleNamespace
from typing import Any, Literal, TypeVar, get_args, get_origin

from config.loader import coerce, field_types, load_config, type_name
from config.post_training_config import ConfigError

T = TypeVar("T")


def _build_parser(cfg_cls: type, extra: dict | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    types_by_field = field_types(cfg_cls)
    for f in fields(cfg_cls):
        annotation = types_by_field[f.name]
        default = f.default if f.default is not MISSING else None
        choices = list(get_args(annotation)) if get_origin(annotation) is Literal else None
        p.add_argument(
            f"--{f.name}",
            type=str,
            default=None,
            metavar=type_name(annotation) if choices is None else None,
            choices=[str(c) for c in choices] if choices else None,
            help=f"default: {default}",
        )
    for flag, kwargs in (extra or {}).items():
        p.add_argument(flag, **kwargs)
    return p


def parse_config(cfg_cls: type[T], extra: dict | None = None) -> tuple[T, SimpleNamespace]:
    """
    Parse CLI overrides for a config dataclass.

    Returns ``(cfg, extras)`` where ``cfg`` is an instance of ``cfg_cls`` with any provided
    ``--field`` overrides applied, and ``extras`` is a namespace of any non-config args
    declared via ``extra`` (mapping ``"--flag" -> dict(argparse kwargs)``).
    """
    p = _build_parser(cfg_cls, extra)
    args = vars(p.parse_args())
    types_by_field = field_types(cfg_cls)
    try:
        overrides = {k: coerce(v, types_by_field[k], k) for k, v in args.items() if k in types_by_field and v is not None}
        cfg = cfg_cls(**overrides)
    except ConfigError as e:
        p.error(str(e))
    extras = {k: v for k, v in args.items() if k not in types_by_field}
    return cfg, SimpleNamespace(**extras)


def parse_config_with_json(
    cfg_cls: type[T], default_json: str, extra: dict | None = None
) -> tuple[T, SimpleNamespace]:
    """
    Like :func:`parse_config`, but resolves the config from a JSON file too.

    Adds two flags on top of the per-field ``--field`` overrides:
      - ``--config PATH``   : the stage JSON to load (default ``default_json``).
      - ``--print-config``  : print the fully resolved config as JSON and exit.

    Resolution order (low -> high): dataclass defaults < ``configs/base.json`` < the stage
    JSON < CLI ``--field`` overrides (see :func:`config.loader.load_config`).

    Returns ``(cfg, extras)`` exactly like :func:`parse_config`.
    """
    p = _build_parser(cfg_cls, extra)
    p.add_argument("--config", default=default_json, help="stage JSON config to load")
    p.add_argument("--print-config", action="store_true", help="print resolved config and exit")
    args = vars(p.parse_args())

    types_by_field = field_types(cfg_cls)
    reserved = {"config", "print_config"}
    overrides: dict[str, Any] = {k: v for k, v in args.items() if k in types_by_field and v is not None}
    extras = {k: v for k, v in args.items() if k not in types_by_field and k not in reserved}

    try:
        cfg = load_config(cfg_cls, args["config"], overrides)
    except ConfigError as e:
        p.error(str(e))
    if args["print_config"]:
        print(json.dumps(asdict(cfg), indent=2))  # type: ignore[call-overload]
        raise SystemExit(0)
    return cfg, SimpleNamespace(**extras)

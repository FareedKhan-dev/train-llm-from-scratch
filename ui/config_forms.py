"""
Render an editable config form from a stage's dataclass fields and write it back to JSON.

Base/runtime fields (shared) are saved to ``base.json``; the stage's own hyperparameters to
``configs/<stage>.json`` — matching the loader's merge model. Shows the exact resolved
launch command underneath.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields

import streamlit as st

from config.loader import load_config
from config.post_training_config import BaseModelConfig
from ui.jobs import build_argv
from ui.stages import Stage

_BASE_FIELDS = {f.name for f in fields(BaseModelConfig)}


def _widget(f, value):
    t = str(f.type)
    label = f.name
    if t == "bool" or "bool" in t and "|" not in t:
        return st.checkbox(label, bool(value))
    if "int" in t and "|" not in t:
        return int(st.number_input(label, value=int(value), step=1, format="%d"))
    if "float" in t:
        return float(st.number_input(label, value=float(value), format="%g"))
    # str / str|None / unknown -> text; empty string means "unset" (-> null) for optionals
    return st.text_input(label, "" if value is None else str(value))


def _coerce_blank(f, value):
    """Map an empty text box back to None for optional (``| None``) string fields."""
    if isinstance(value, str) and value == "" and "None" in str(f.type):
        return None
    return value


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def render_form(stage: Stage, smoke: bool = False, nproc: int = 1) -> None:
    json_path = stage.smoke_json if smoke else stage.config_json
    base_path = os.path.join(os.path.dirname(json_path), "base.json")
    cur = load_config(stage.cfg_cls, json_path)

    vals: dict = {}
    with st.expander("🧱 Model & runtime  ·  shared `base.json`", expanded=False):
        cols = st.columns(2)
        bf = [f for f in fields(stage.cfg_cls) if f.name in _BASE_FIELDS]
        for i, f in enumerate(bf):
            with cols[i % 2]:
                vals[f.name] = _coerce_blank(f, _widget(f, getattr(cur, f.name)))

    with st.expander(f"⚙️ {stage.title} hyperparameters  ·  `{json_path}`", expanded=True):
        cols = st.columns(2)
        sf = [f for f in fields(stage.cfg_cls) if f.name not in _BASE_FIELDS]
        for i, f in enumerate(sf):
            with cols[i % 2]:
                vals[f.name] = _coerce_blank(f, _widget(f, getattr(cur, f.name)))

    if st.button("💾 Save config", key=f"save_{stage.key}_{smoke}"):
        base_vals = {k: v for k, v in vals.items() if k in _BASE_FIELDS}
        stage_vals = {k: v for k, v in vals.items() if k not in _BASE_FIELDS}
        _write_json(base_path, base_vals)
        _write_json(json_path, stage_vals)
        st.success(f"Saved → `{base_path}` and `{json_path}`")

    st.caption("Resolved launch command")
    argv = build_argv(stage.script, json_path, nproc, stage.multi_gpu)
    st.code(" ".join(argv), language="bash")

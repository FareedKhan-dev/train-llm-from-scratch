"""Shared look-and-feel: page config, CSS injection, header, GPU status, job badges."""

from __future__ import annotations

import streamlit as st

# Palette matching the hand-drawn diagrams.
COLORS = {
    "data": "#27ae60", "proc": "#2c6fbb", "store": "#16a085", "model": "#d48806",
    "rl": "#e67e22", "loss": "#c0392b", "eval": "#8e44ad", "ckpt": "#888888",
}

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
h1, h2, h3 {letter-spacing: .2px;}
.hero {background: linear-gradient(135deg,#16a085 0%, #2c6fbb 100%);
       color: #fff; padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {color:#fff; margin:0 0 .3rem 0; font-size: 1.7rem;}
.hero p {color:#eafaf5; margin:0; font-size: .98rem;}
.stage-card {border-left: 6px solid var(--c,#16a085); background: rgba(127,127,127,.06);
             padding: .8rem 1rem; border-radius: 10px; margin:.3rem 0;}
.badge {display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.78rem;
        font-weight:600; color:#fff;}
.b-run{background:#d48806;} .b-ok{background:#27ae60;} .b-fail{background:#c0392b;}
.b-idle{background:#888;}
div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
"""

_BADGE = {"running": ("b-run", "running"), "finished": ("b-ok", "finished"),
          "failed": ("b-fail", "failed"), "stopped": ("b-idle", "stopped"), "none": ("b-idle", "idle")}


def setup_page(title: str, icon: str = "🧠") -> None:
    st.set_page_config(page_title=f"{title} · Train LLM From Scratch", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls, label = _BADGE.get(status, ("b-idle", status))
    return f'<span class="badge {cls}">{label}</span>'


def gpu_status():
    """Return a list of (idx, name, used_mb, total_mb, util) via nvidia-smi, or []."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.strip().splitlines():
        i, name, used, total, util = [x.strip() for x in line.split(",")]
        rows.append((int(i), name, int(used), int(total), int(util)))
    return rows

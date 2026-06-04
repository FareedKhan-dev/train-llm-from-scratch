"""
Train LLM From Scratch — control panel (Home).

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import os

import streamlit as st

from ui import jobs, theme
from ui.stages import ABS_DOC, STAGES

theme.setup_page("Control Panel", "🧠")
theme.hero("🧠  Train LLM From Scratch — Control Panel",
           "Pretrain → SFT → Reward Model → DPO → PPO → GRPO · evaluate · chat — all from one place.")

# Master pipeline diagram (the hand-drawn overview).
overview = ABS_DOC("docs/diagrams/00_overview.png")
if os.path.exists(overview):
    st.image(overview, use_container_width=True)

st.markdown(
    "Use the **pages in the sidebar** to prepare data, configure and launch each training stage "
    "(with live logs + metric charts), evaluate on GSM8K, and chat with any checkpoint."
)

# --- GPU status ---
st.subheader("🖥️  GPUs")
gpus = theme.gpu_status()
if gpus:
    cols = st.columns(len(gpus))
    for col, (i, name, used, total, util) in zip(cols, gpus):
        col.metric(f"GPU {i} · {name.split(' ')[-1]}", f"{used/1024:.0f}/{total/1024:.0f} GB", f"{util}% util")
else:
    st.info("No GPUs detected (nvidia-smi unavailable) — CPU / smoke mode only.")

# --- Job status ---
st.subheader("📋  Jobs")
busy = jobs.gpu_busy()
if busy:
    st.warning(f"A GPU job is currently running: **{busy}**. Other GPU launches are guarded until it finishes.")
active = jobs.active_jobs()
if active:
    for rec in active:
        s = jobs.status(rec["job_id"])
        st.markdown(
            f'<div class="stage-card"><b>{rec["job_id"]}</b> &nbsp; {theme.status_badge(s)} '
            f'&nbsp; <code>{" ".join(rec["cmd"][:3])} …</code></div>', unsafe_allow_html=True)
else:
    st.caption("No jobs launched yet.")

# --- Pipeline map ---
st.subheader("🗺️  Pipeline")
for key, sgt in STAGES.items():
    s = jobs.status(key)
    st.markdown(
        f'<div class="stage-card" style="--c:{theme.COLORS.get("model")}">'
        f'{sgt.emoji} <b>{sgt.title}</b> &nbsp; {theme.status_badge(s)} '
        f'&nbsp; <span style="opacity:.7">{sgt.script}</span></div>', unsafe_allow_html=True)

st.divider()
st.caption("Docs: see the MkDocs site (`mkdocs serve`) or the `docs/` folder. "
           "CLI reference: `POST_TRAINING.md`.")

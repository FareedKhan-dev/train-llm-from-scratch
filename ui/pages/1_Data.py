"""Data preparation page: download + preprocess every dataset as background jobs."""

from __future__ import annotations

import os
import re
import sys
import time

import streamlit as st

from ui import jobs, theme
from ui.stages import DATA_DIR, DATA_SCRIPTS

theme.setup_page("Data", "📦")
theme.hero("📦  Data preparation",
           "Download & preprocess the corpora — Pile (pretrain), Alpaca/Dolly/GSM8K (SFT), "
           "HH-RLHF/UltraFeedback (preferences), GSM8K + arithmetic (RL).")

st.caption("Each button launches the real `scripts/prepare_*.py` as a background job (sets "
           "`HF_HOME=/ephemeral/hf_cache`). Large downloads take a while — watch the log below.")

for name, args in DATA_SCRIPTS.items():
    job_id = "data_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    s = jobs.status(job_id)
    c1, c2 = st.columns([3, 1])
    c1.markdown(f'<div class="stage-card" style="--c:{theme.COLORS["data"]}">'
                f'<b>{name}</b> &nbsp; {theme.status_badge(s)}<br>'
                f'<code>{" ".join(args)}</code></div>', unsafe_allow_html=True)
    if c2.button("Prepare", key=f"prep_{job_id}", disabled=(s == "running")):
        env_argv = [sys.executable, *args]
        rec = jobs.launch(job_id, env_argv, kind="cpu")
        # data scripts need the HF cache env; relaunch path uses jobs env + HF_HOME
        os.environ.setdefault("HF_HOME", "/ephemeral/hf_cache")
        st.session_state["last_data_job"] = job_id
        st.rerun()

st.subheader("Live log")
last = st.session_state.get("last_data_job")
sel = st.selectbox("Job", [j["job_id"] for j in jobs.active_jobs() if j["job_id"].startswith("data_")] or ["—"],
                   index=0)
job_to_show = sel if sel != "—" else last
if job_to_show:
    st.code(jobs.tail_log(job_to_show, 10000) or "(no log yet)", language="text")
    if jobs.status(job_to_show) == "running" and st.toggle("Auto-refresh", value=True):
        time.sleep(3); st.rerun()

st.subheader("Files in /ephemeral/data")
if os.path.isdir(DATA_DIR):
    rows = []
    for fn in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, fn)
        if os.path.isfile(p):
            rows.append({"file": fn, "size": f"{os.path.getsize(p)/1e6:.1f} MB"})
    st.dataframe(rows, use_container_width=True, hide_index=True) if rows else st.caption("empty")
else:
    st.caption("/ephemeral/data does not exist yet.")

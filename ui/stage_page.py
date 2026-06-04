"""
Reusable renderer for a training-stage page: theory + diagram → config form → launch /
stop (with the GPU-busy guard) → live log → metric charts. Each stage page is a 3-liner
that calls :func:`render_stage_page`.
"""

from __future__ import annotations

import streamlit as st

from ui import jobs, metrics, theme
from ui.config_forms import render_form
from ui.docs_render import render_doc
from ui.stages import STAGES


def render_stage_page(stage_key: str) -> None:
    stage = STAGES[stage_key]
    theme.setup_page(stage.title, stage.emoji)
    theme.hero(f"{stage.emoji}  {stage.title}", f"Configure, launch, and monitor the {stage.title} stage.")

    job_status = jobs.status(stage.key)
    st.markdown(f"**Job status:** {theme.status_badge(job_status)}", unsafe_allow_html=True)

    tab_run, tab_theory = st.tabs(["🚀 Run", "📖 Theory"])

    with tab_theory:
        render_doc(st, stage.doc_md, stage.diagram_png)

    with tab_run:
        c1, c2 = st.columns([1, 1])
        with c1:
            smoke = st.toggle("Smoke mode (tiny model, few steps, CPU)", value=False, key=f"smoke_{stage.key}")
        with c2:
            ngpu = theme.gpu_status()
            max_gpu = max(1, len(ngpu))
            nproc = st.slider("GPUs (torchrun)", 1, max_gpu, min(2, max_gpu) if stage.multi_gpu and not smoke else 1,
                              disabled=smoke or not stage.multi_gpu, key=f"nproc_{stage.key}")
            if smoke:
                nproc = 1

        st.subheader("Configuration")
        render_form(stage, smoke=smoke, nproc=nproc)

        st.subheader("Launch")
        busy = jobs.gpu_busy()
        guard = (not smoke) and busy is not None and busy != stage.key
        if guard:
            st.warning(f"🚧 GPUs are occupied by job **{busy}**. Stop it, or use Smoke mode (CPU), "
                       f"before launching {stage.title} on GPU.")
        cols = st.columns(3)
        with cols[0]:
            if st.button("▶️ Launch", type="primary", disabled=(job_status == "running" or guard),
                         key=f"launch_{stage.key}"):
                cfg_json = stage.smoke_json if smoke else stage.config_json
                argv = jobs.build_argv(stage.script, cfg_json, nproc, stage.multi_gpu)
                kind = "cpu" if smoke else ("gpu" if stage.multi_gpu else "cpu")
                jobs.launch(stage.key, argv, kind=kind)
                st.rerun()
        with cols[1]:
            if st.button("⏹️ Stop", disabled=(job_status != "running"), key=f"stop_{stage.key}"):
                jobs.stop(stage.key)
                st.rerun()
        with cols[2]:
            auto = st.toggle("Auto-refresh", value=(job_status == "running"), key=f"auto_{stage.key}")

        st.subheader("Live log")
        st.code(jobs.tail_log(stage.key, 12000) or "(no log yet — launch the stage)", language="text")

        st.subheader("Metrics")
        log = metrics.latest_log(stage.log_prefix)
        if log:
            df = metrics.load_metrics(log)
            cols_to_plot = metrics.metric_columns(df) if not df.empty else []
            if "step" in df.columns and cols_to_plot:
                st.line_chart(df.set_index("step")[cols_to_plot])
                st.caption(f"from `{log}` · {len(df)} logged points")
            else:
                st.info("No metrics logged yet.")
        else:
            st.info(f"No `{stage.log_prefix}_*.jsonl` metrics found yet.")

        if auto:
            import time
            time.sleep(3)
            st.rerun()

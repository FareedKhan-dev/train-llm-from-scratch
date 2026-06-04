"""Evaluate a checkpoint on GSM8K (greedy), in-process."""

from __future__ import annotations

import glob

import streamlit as st

from ui import theme

theme.setup_page("Evaluate", "📊")
theme.hero("📊  Evaluate on GSM8K", "Greedy GSM8K accuracy for any checkpoint, with sample generations.")

ckpts = sorted(glob.glob("/ephemeral/ckpts/*.pt"))
if not ckpts:
    st.warning("No checkpoints found in /ephemeral/ckpts. Train a stage first.")
    st.stop()

ckpt = st.selectbox("Checkpoint", ckpts, index=len(ckpts) - 1)
c1, c2, c3 = st.columns(3)
limit = c1.slider("Num questions", 5, 200, 20, step=5)
max_new = c2.slider("Max new tokens", 64, 400, 256, step=32)
device = c3.selectbox("Device", ["cuda", "cpu"], index=0)

if st.button("▶️ Run GSM8K eval", type="primary"):
    with st.spinner(f"Generating + scoring {limit} GSM8K problems on {device} …"):
        from src.post_training.evaluation import gsm8k_accuracy, load_gsm8k_eval
        from src.post_training.inference import load_model_from_ckpt
        model = load_model_from_ckpt(ckpt, device)
        qa = load_gsm8k_eval("test", limit=limit)
        res = gsm8k_accuracy(model, qa, device=device, max_new_tokens=max_new, greedy=True,
                             return_samples=min(5, limit))
    st.metric("GSM8K accuracy", f"{res['accuracy']*100:.1f}%", f"{res['correct']}/{res['n']} correct")
    st.subheader("Sample generations")
    for s in res["samples"]:
        with st.expander(("✅ " if s["correct"] else "❌ ") + s["q"][:90]):
            st.markdown(f"**Gold:** `{s['gold']}` · **Correct:** {s['correct']}")
            st.code(s["response"][:1200], language="text")

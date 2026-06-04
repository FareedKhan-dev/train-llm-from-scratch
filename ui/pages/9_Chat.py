"""Chat with any checkpoint, in-process (no subprocess)."""

from __future__ import annotations

import glob

import streamlit as st

from ui import theme

theme.setup_page("Chat", "💬")
theme.hero("💬  Chat", "Talk to any checkpoint — chat template for instruction models, raw mode for the base.")

ckpts = sorted(glob.glob("/ephemeral/ckpts/*.pt"))
if not ckpts:
    st.warning("No checkpoints in /ephemeral/ckpts yet. Train a stage (or run a smoke job) first.")
    st.stop()

with st.sidebar:
    st.header("Generation")
    ckpt = st.selectbox("Checkpoint", ckpts, index=len(ckpts) - 1)
    device = st.selectbox("Device", ["cuda", "cpu"], index=0)
    raw = st.toggle("Raw mode (base continuation, no chat template)", value=False)
    greedy = st.toggle("Greedy (deterministic)", value=False)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05, disabled=greedy)
    top_p = st.slider("top-p", 0.1, 1.0, 0.95, 0.05, disabled=greedy)
    max_new = st.slider("Max new tokens", 16, 512, 256, 16)
    system = st.text_input("System prompt (chat mode)", "")
    if st.button("🗑️ Clear history"):
        st.session_state.pop("chat_msgs", None)


@st.cache_resource(show_spinner="Loading checkpoint …")
def _load(path, dev):
    from src.post_training.inference import load_model_from_ckpt
    return load_model_from_ckpt(path, dev)


model = _load(ckpt, device)
st.caption(f"Loaded `{ckpt}` · {sum(p.numel() for p in model.parameters())/1e6:.0f}M params on {device} · "
           f"mode={'raw' if raw else 'chat'}")

st.session_state.setdefault("chat_msgs", [])
for m in st.session_state["chat_msgs"]:
    st.chat_message(m["role"]).markdown(m["content"])

prompt = st.chat_input("Ask something (e.g. 'What is 13 + 29?')")
if prompt:
    st.session_state["chat_msgs"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Generating …"):
        from src.post_training.inference import generate_reply
        reply = generate_reply(model, prompt, device=device, system=system or None, raw=raw,
                               max_new_tokens=max_new, temperature=temperature,
                               top_p=top_p if top_p < 1 else None, greedy=greedy)
        st.markdown(reply if reply.strip() else "_(empty generation)_")
    st.session_state["chat_msgs"].append({"role": "assistant", "content": reply})

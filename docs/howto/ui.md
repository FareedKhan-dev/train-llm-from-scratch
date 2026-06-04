# The control-panel UI

A polished **Streamlit** app that wraps the whole pipeline — theory, one-click training, live logs,
metric charts, evaluation, and chat — so you can drive everything without memorizing commands.

```bash
pip install -e ".[ui]"
streamlit run ui/app.py
```

## What's inside

- **Home** — the pipeline diagram, live job status, and GPU status at a glance.
- **Data** — launch the `prepare_*` scripts and watch the files appear under `/ephemeral/data`.
- **A page per stage** (Pretrain, SFT, Reward, DPO, PPO, GRPO). Each page gives you:
    1. the **theory + hand-drawn diagram** for that stage (pulled straight from these docs),
    2. a **config form** that writes the stage's `configs/*.json`,
    3. a **Launch** button that runs the real training script as a background job, and
    4. a **live log tail** + **metric charts** (loss / reward / KL / accuracy) read from the JSONL.
- **Evaluate** — run GSM8K accuracy on any checkpoint and inspect sample generations.
- **Chat** — talk to any checkpoint in-process, with temperature / top-p / top-k controls.

## How launching works

The app builds the same command you'd type (`torchrun … scripts/train_sft.py --config …`), runs it as a
detached background process, and streams its log file into the page — so jobs keep running even if you
navigate away or refresh. A **Stop** button cleanly terminates the job (and all `torchrun` workers).

!!! warning "One GPU job at a time"
    Two multi-GPU `torchrun` jobs would fight over the same cards and OOM. The UI's **GPU-busy guard**
    disables *Launch* for a GPU stage while another GPU job is running (e.g. while pretraining occupies
    both H100s). CPU/smoke runs and the in-process **Chat** page always stay available.

!!! tip "Smoke mode"
    Each stage page has a *smoke* toggle that uses the tiny `configs/smoke/*.json` (small model, few
    steps, CPU) so you can validate the whole flow in seconds before committing to a real run.

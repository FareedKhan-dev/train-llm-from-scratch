"""
Measure how fast model sizes train on this machine, so you can pick a preset (and a run
length) that fits your hardware before committing to a long run.

For every preset and architecture it builds the model, runs a few warm-up steps, then times
real training steps (forward, backward, AdamW) on random tokens with the preset's batch and
context sizes. It prints a Markdown table you can paste into an issue or the docs.

    python scripts/benchmark.py                                   # laptop presets, both architectures
    python scripts/benchmark.py --presets 13m 77m --device cuda   # GPU sizes
    python scripts/benchmark.py --presets student --arch modern --steps 20
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the repo without installing

from config.config import default_config  # noqa: E402
from config.presets import PRESETS, apply_preset  # noqa: E402
from src.device import configure_cpu_threads, resolve_device  # noqa: E402
from src.models.factory import build_model  # noqa: E402


def cpu_name() -> str:
    return platform.processor() or platform.machine()


def benchmark(preset: str, arch: str, device: str, steps: int, warmup: int, vocab: int | None) -> dict:
    cfg = apply_preset(dict(default_config), preset)
    cfg["arch"] = arch
    if vocab:
        cfg["vocab_size"] = vocab
    torch.manual_seed(0)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, T = cfg["t_batch_size"], cfg["t_context_length"]
    data = torch.randint(0, cfg["vocab_size"], (B, T + 1), device=device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def step() -> None:
        _, loss = model(data[:, :-1], data[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    step_time = (time.perf_counter() - t0) / steps
    params = sum(p.numel() for p in model.parameters())
    return {
        "preset": preset,
        "arch": arch,
        "params": params,
        "tokens_per_s": B * T / step_time,
        "step_ms": 1000 * step_time,
        "run_minutes": cfg["t_train_steps"] * step_time / 60,
        "train_steps": cfg["t_train_steps"],
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.startswith("cuda") else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--presets", nargs="+", default=["tiny", "student", "small"], choices=sorted(PRESETS))
    p.add_argument("--arch", nargs="+", default=["classic", "modern"], choices=["classic", "modern"])
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=10, help="timed steps per measurement")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--vocab", type=int, default=None, help="override the vocabulary size")
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: all cores)")
    args = p.parse_args()

    device = resolve_device(args.device)
    threads = configure_cpu_threads(args.threads) if device == "cpu" else None
    where = cpu_name() + (f", {threads} threads" if threads else "")
    if device.startswith("cuda"):
        where = torch.cuda.get_device_name()
    print(f"Benchmarking on {device} ({where}), PyTorch {torch.__version__}\n")
    print("| preset | arch | parameters | tokens/s | ms/step | full run (steps) | peak GPU memory |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for preset in args.presets:
        for arch in args.arch:
            r = benchmark(preset, arch, device, args.steps, args.warmup, args.vocab)
            mem = f"{r['peak_gib']:.2f} GiB" if r["peak_gib"] is not None else "n/a"
            print(f"| {r['preset']} | {r['arch']} | {r['params'] / 1e6:.2f}M | {r['tokens_per_s']:,.0f} | "
                  f"{r['step_ms']:.0f} | {r['run_minutes']:.0f} min ({r['train_steps']}) | {mem} |", flush=True)


if __name__ == "__main__":
    main()

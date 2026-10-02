"""
Regenerate the data plots used by the docs (the PNGs next to this file).

Every plot is computed from the repo's own code or from real training logs, so a change in
the code shows up in the figures:

    python docs/diagrams/make_plots.py            # all plots
    python docs/diagrams/make_plots.py rope       # one plot

Training curves read logs/runs/*.log, written by the student-track runs described in
docs/student/README.md; they are skipped if the logs are missing.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.modern import apply_rope, attention_mask, rope_cache, rope_frequencies  # noqa: E402
from src.optim import cosine_lr, linear_lr, newton_schulz, wsd_lr  # noqa: E402

OUT = Path(__file__).resolve().parent

# Palette: categorical slots 1-3 of the validated default palette (light surface), plus ink.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
OFF = "#f0efec"  # neutral, for "masked" cells

# Use the first of these fonts that is installed; matplotlib ships DejaVu Sans, so one always is.
_installed = {f.name for f in font_manager.fontManager.ttflist}
FONTS = [f for f in ("Segoe UI", "Helvetica Neue", "Arial") if f in _installed][:1] + ["DejaVu Sans"]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # no "font not found" noise

plt.rcParams.update({
    "font.family": FONTS,
    "font.size": 10,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 1.0,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote docs/diagrams/{name}.png")


def label(ax: plt.Axes, x: float, y: float, text: str, **kw: object) -> None:
    """Direct label in a text token (never the series color)."""
    ax.annotate(text, (x, y), color=INK_2, fontsize=9, **kw)


# ---------------------------------------------------------------------------------------------
def lr_schedules() -> None:
    steps = range(0, 10_001, 20)
    kw = dict(warmup_steps=500, max_steps=10_000, lr=1.0, min_lr=0.1)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for fn, color, name in ((cosine_lr, BLUE, "cosine"), (wsd_lr, ORANGE, "WSD (warmup-stable-decay)"),
                            (linear_lr, AQUA, "linear")):
        ax.plot(list(steps), [fn(s, **kw) for s in steps], color=color, label=name)
    label(ax, 7700, 0.13, "cosine", ha="left")
    label(ax, 5200, 1.025, "WSD: flat, then a short decay", ha="left")
    label(ax, 1300, 0.84, "linear", ha="left")
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate (fraction of peak)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Three learning-rate schedules with the same warmup and peak")
    ax.legend(loc="lower left")
    save(fig, "lr_schedules")


def rope() -> None:
    dim, positions = 64, torch.arange(0, 96)
    cos, sin = rope_cache(dim, 256)
    freqs = rope_frequencies(dim)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    for i, color, name in ((0, BLUE, "pair 0 (fast)"), (8, ORANGE, "pair 8"), (16, AQUA, "pair 16 (slow)")):
        left.plot(positions, torch.cos(positions * freqs[i]), color=color, label=name)
    label(left, 1, -1.24, "fast pairs tell neighbors apart, slow pairs track long distances")
    left.set_ylim(-1.34, 1.6)
    left.set_xlabel("position m")
    left.set_ylabel("cos(m * theta_i)")
    left.set_title("Each pair rotates at its own speed")
    left.legend(loc="upper right", ncol=3)

    torch.manual_seed(0)
    q, k = torch.randn(dim), torch.randn(dim)

    def score(m: int, n: int) -> float:
        qm = apply_rope(q[None], cos[m : m + 1], sin[m : m + 1])[0]
        kn = apply_rope(k[None], cos[n : n + 1], sin[n : n + 1])[0]
        return float(qm @ kn)

    dist = list(range(0, 41))
    near = [score(40, 40 - d) for d in dist]
    far = [score(200, 200 - d) for d in dist]
    right.plot(dist, near, color=BLUE, label="query at position 40")
    right.plot(dist, far, linestyle="none", marker="o", markersize=5, markerfacecolor=ORANGE,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label="query at position 200")
    label(right, 0.5, max(near) - 1.6, "the two curves are identical:\nonly the distance m - n matters")
    right.set_xlabel("distance between query and key (m - n)")
    right.set_ylabel("attention score q . k")
    right.set_title("The score depends only on relative position")
    right.legend(loc="lower center")
    save(fig, "rope")


def rounded_hbar(ax: plt.Axes, y: float, width: float, height: float, color: str) -> None:
    """A horizontal bar with a rounded data-end and a square baseline."""
    r = height / 2 * 0.9
    ax.add_patch(FancyBboxPatch((0, y - height / 2), width, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=0, facecolor=color, mutation_aspect=1))
    ax.add_patch(Rectangle((0, y - height / 2), min(width, r * 3), height, linewidth=0, facecolor=color))


def kv_cache_memory() -> None:
    layers, heads, head_dim, bytes_per = 16, 32, 64, 2  # a 1B-class model in bf16
    n_embed = heads * head_dim
    rows = [
        ("MHA, 32 KV heads", 2 * layers * heads * head_dim * bytes_per),
        ("GQA, 8 KV heads", 2 * layers * 8 * head_dim * bytes_per),
        ("MLA, latent 512 + rope 32", layers * (n_embed // 4 + 32) * bytes_per),
        ("MQA, 1 KV head", 2 * layers * 1 * head_dim * bytes_per),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    biggest = rows[0][1] / 1024
    for i, (_name, b) in enumerate(rows):
        y = len(rows) - 1 - i
        kib = b / 1024
        rounded_hbar(ax, y, kib, 0.5, BLUE)
        gb_32k = b * 32_768 / 1024**3
        ax.annotate(f"{kib:.0f} KiB per token  ({gb_32k:.2f} GiB for 32K tokens)", (kib + biggest * 0.015, y),
                    va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, biggest * 1.75)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("KV cache per token (KiB, bf16)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("What generation stores per token (16 layers, 32 heads of 64)")
    save(fig, "kv_cache_memory")


def attention_masks() -> None:
    T = 12
    full, _ = attention_mask(T, T, 0, None, torch.device("cpu"))
    full = torch.tril(torch.ones(T, T, dtype=torch.bool)) if full is None else full
    window, _ = attention_mask(T, T, 0, 4, torch.device("cpu"))
    cached, _ = attention_mask(4, T, 8, None, torch.device("cpu"))
    panels = [(full, "causal, fresh forward pass"), (window, "sliding window, W = 4"),
              (cached, "4 new tokens after 8 cached")]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.7), gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, (mask, title) in zip(axes, panels):
        rows, cols = mask.shape
        offset = T - rows  # draw the cached panel's queries at their real positions
        for qi in range(rows):
            for kj in range(cols):
                color = BLUE if bool(mask[qi, kj]) else OFF
                ax.add_patch(Rectangle((kj, qi + offset), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=1.6))
        ax.set_xlim(0, T)
        ax.set_ylim(T, offset)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("key position")
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([0.5, T - 0.5], ["0", str(T - 1)])
        ax.set_yticks([offset + 0.5, T - 0.5], [str(offset), str(T - 1)])
        ax.tick_params(length=0)
    axes[0].set_ylabel("query position")
    fig.text(0.5, 0.012, "blue: the query may attend to that key    light: masked", ha="center", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "attention_masks.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("wrote docs/diagrams/attention_masks.png")


def muon_singular_values() -> None:
    torch.manual_seed(0)
    rows, cols = 128, 256
    U, _ = torch.linalg.qr(torch.randn(rows, rows))
    V, _ = torch.linalg.qr(torch.randn(cols, rows))
    S = torch.logspace(1, -2, rows)  # a gradient-like spectrum: a few big directions, many small ones
    G = U @ torch.diag(S) @ V.T
    before = torch.linalg.svdvals(G)
    after = torch.linalg.svdvals(newton_schulz(G, steps=5, dtype=torch.float32))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    idx = range(1, rows + 1)
    ax.plot(idx, before, color=BLUE, label="momentum matrix M")
    ax.plot(idx, after, color=ORANGE, label="after 5 Newton-Schulz steps")
    ax.set_yscale("log")
    near_one = float(((after > 0.5) & (after < 1.5)).float().mean())
    label(ax, 20, float(before[20]) * 1.5, "the raw update: a few directions dominate")
    label(ax, 52, float(after.max()) * 1.5, f"after: {near_one:.0%} of the directions are between 0.5 and 1.5")
    ax.set_xlabel("singular value index (largest first)")
    ax.set_ylabel("singular value (log scale)")
    ax.set_title("Newton-Schulz orthogonalization flattens the spectrum")
    ax.legend(loc="lower left")
    save(fig, "muon_singular_values")


EVAL_RE = re.compile(r"Step: (\d+), Train loss: ([\d.]+), Dev loss: ([\d.]+)")


def read_eval(path: Path) -> tuple[list[int], list[float]]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    pairs = [(int(m.group(1)), float(m.group(3))) for m in EVAL_RE.finditer(text)]
    final = re.search(r"Finished training\. Train loss: [\d.]+, Dev loss: ([\d.]+)", text)
    steps, losses = [p[0] for p in pairs], [p[1] for p in pairs]
    if final and steps:
        steps.append(steps[-1] + (steps[1] - steps[0] if len(steps) > 1 else 1))
        losses.append(float(final.group(1)))
    return steps, losses


def training_curves(name: str, runs: list[tuple[str, str, str]], title: str, vocab: int) -> None:
    logs = ROOT / "logs" / "runs"
    present = [(f, c, n) for f, c, n in runs if (logs / f).exists()]
    if not present:
        print(f"skipped {name}: no logs in logs/runs/")
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for file, color, run_name in present:
        steps, losses = read_eval(logs / file)
        ax.plot(steps, losses, color=color, label=run_name)
        ax.plot(steps[-1], losses[-1], marker="o", markersize=8, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=2)
        label(ax, steps[-1], losses[-1], f"  {run_name}: {losses[-1]:.2f}", va="center")
    uniform = math.log(vocab)
    right = max(read_eval(logs / f)[0][-1] for f, _, _ in present)
    ax.axhline(uniform, color=AXIS, linewidth=1.0)
    label(ax, right * 0.42, uniform + 0.12, f"a uniform guess: ln({vocab}) = {uniform:.2f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("dev loss (cross-entropy)")
    ax.set_ylim(2.0, uniform + 0.6)
    ax.set_xlim(-right * 0.02, right * 1.32)
    ax.set_title(title)
    ax.legend(loc="upper right")
    save(fig, name)


PLOTS = {
    "lr_schedules": lr_schedules,
    "rope": rope,
    "kv_cache_memory": kv_cache_memory,
    "attention_masks": attention_masks,
    "muon_singular_values": muon_singular_values,
    "tiny_loss_curves": lambda: training_curves(
        "tiny_loss_curves",
        [("tiny_classic.log", BLUE, "classic, 636K params"), ("tiny_modern.log", ORANGE, "modern, 369K params")],
        "The tiny preset on a laptop CPU: same data, same steps", 4096),
}

if __name__ == "__main__":
    for key in sys.argv[1:] or list(PLOTS):
        PLOTS[key]()

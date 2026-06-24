"""
Plot the real training artifacts produced by the GPU run into the README images:
  - images/loss_curve.png       from the pretraining stdout (step / train loss / dev loss)
  - images/gsm8k_accuracy.png   from the across-stages GSM8K table (stage_table.jsonl)

Usage:
    python images/plot_artifacts.py <artifacts_dir>
where <artifacts_dir> holds pretrain_stdout.txt and stage_table.jsonl.
"""
import json, os, re, sys
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
ART = sys.argv[1] if len(sys.argv) > 1 else OUT

mpl.rcParams["font.family"] = ["Comic Sans MS", "Ink Free", "DejaVu Sans"]
BLUE, ORANGE, PURPLE = "#1565c0", "#e8730c", "#7b3fbf"

def plot_loss():
    path = os.path.join(ART, "pretrain_stdout.txt")
    if not os.path.exists(path):
        print("no pretrain_stdout.txt, skipping loss curve"); return
    txt = open(path, encoding="utf-8", errors="ignore").read()
    train = [(int(s), float(l)) for s, l in re.findall(r"^step (\d+) \| loss ([\d.]+)", txt, re.M)]
    ev = re.findall(r"\[eval\] step (\d+) \| train ([\d.]+) \| dev ([\d.]+)", txt)
    ev = [(int(s), float(a), float(b)) for s, a, b in ev]
    if not train:
        print("no step lines found, skipping loss curve"); return
    fig, ax = plt.subplots(figsize=(9, 5.2))
    xs, ys = zip(*train)
    ax.plot(xs, ys, color=BLUE, lw=2.2, label="train loss")
    if ev:
        es, et, ed = zip(*ev)
        ax.plot(es, ed, color=ORANGE, lw=2.2, marker="o", ms=5, label="dev loss")
    ax.axhline(10.83, color="#999", ls="--", lw=1.2, label="uniform-guess loss  ln(50304)")
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Pretraining a 77M base on The Pile (2x L40)", fontsize=15)
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "loss_curve.png"), dpi=150, facecolor="white")
    print("wrote loss_curve.png  (last train=%.3f%s)" % (ys[-1], (", dev=%.3f" % ed[-1]) if ev else ""))

def plot_gsm8k():
    path = os.path.join(ART, "stage_table.jsonl")
    if not os.path.exists(path):
        print("no stage_table.jsonl, skipping accuracy chart"); return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    order = ["base_pretrained", "sft", "dpo", "ppo", "grpo"]
    rows.sort(key=lambda r: order.index(r["label"]) if r["label"] in order else 99)
    labels = [r["label"].replace("base_pretrained", "base") for r in rows]
    accs = [100.0 * r["accuracy"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accs, color=[BLUE, "#d48806", "#e67e22", "#c0392b", PURPLE][:len(labels)],
                  edgecolor="#333", linewidth=1.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x()+b.get_width()/2, a, f"{a:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("GSM8K accuracy across stages (77M, 2x L40)", fontsize=15)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "gsm8k_accuracy.png"), dpi=150, facecolor="white")
    print("wrote gsm8k_accuracy.png ", dict(zip(labels, [round(a,1) for a in accs])))

if __name__ == "__main__":
    plot_loss()
    plot_gsm8k()

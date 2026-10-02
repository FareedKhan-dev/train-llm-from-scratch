"""
How big is a model, how much compute does training it take, and will it fit?

The model is built on PyTorch's "meta" device, which has shapes but no memory, so even the
3B config is instant on a laptop.

    python scripts/model_report.py --preset 77m
    python scripts/model_report.py --preset student --arch modern --tokens 50e6
    python scripts/model_report.py --config configs/base.json --tflops 990 --mfu 0.4 --gpus 8

Everything printed is an estimate from standard formulas:

- training FLOPs per token = 6 x (weights a token multiplies) + attention (PaLM, appendix B)
- compute-optimal data: about 20 tokens per parameter (Chinchilla, Hoffmann et al. 2022)
- AdamW training memory: 16 bytes per parameter (fp32 weights, gradients, two moments),
  plus activations, which depend on batch and sequence length (Korthikanti et al. 2022)
- KV cache: what generation stores per token (see docs/modern/attention.md)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the repo without installing

from config.config import default_config
from config.presets import PRESETS, apply_preset
from src.models.factory import build_model
from src.models.modern import ModernTransformer


def human(n: float, unit: str = "") -> str:
    for scale, suffix in ((1e18, "E"), (1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{suffix}{unit}"
    return f"{n:.0f}{unit}"


def gib(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.2f} GiB"


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        cfg.setdefault("t_context_length", cfg.get("context_length"))
        cfg.setdefault("t_batch_size", cfg.get("batch_size", 8))
    else:
        cfg = apply_preset(dict(default_config), args.preset) if args.preset else dict(default_config)
    if args.arch:
        cfg["arch"] = args.arch
    return cfg


def estimate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Parameter counts, FLOPs and memory of the model ``cfg`` describes (no memory used)."""
    with torch.device("meta"):  # shapes only, no memory
        model = build_model(cfg)
    modern = isinstance(model, ModernTransformer)
    V, C, L, H = cfg["vocab_size"], cfg["n_embed"], cfg["n_blocks"], cfg["n_head"]
    T = cfg.get("t_context_length") or cfg["context_length"]
    D = C // H

    total = sum(p.numel() for p in model.parameters())
    active = model.active_params() if modern else total
    lookup = model.token_embed.weight.numel() + (0 if modern else model.position_embed.weight.numel())
    tied = modern and model.config.tie_embeddings
    matmul_params = active - lookup + (V * C if tied else 0)  # the output layer is a matmul, tied or not
    mla = modern and model.config.attention == "mla"
    qk_dim = D + model.config.mla_rope_dim if mla else D
    first_block = model.blocks[0] if modern else model.attn_blocks[0]
    if mla:
        kv_per_token = L * (model.config.latent_dim + model.config.mla_rope_dim) * 2
    else:
        kv_heads = model.config.kv_heads if modern else H
        kv_per_token = 2 * L * kv_heads * D * 2
    batch = cfg.get("t_batch_size") or 8
    return {
        "arch": "modern" if modern else "classic",
        "window": T,
        "batch": batch,
        "total": total,
        "active": active,
        "lookup": lookup,
        "per_block": sum(p.numel() for p in first_block.parameters()),
        "flops_per_token": 6 * matmul_params + 6 * L * H * T * (qk_dim + D),
        "adam_bytes": 16 * total,
        "activation_bytes": L * batch * T * C * (34 + 5 * H * T / C),  # bf16, standard GPT block
        "kv_bytes_per_token": kv_per_token,
    }


def report(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    e = estimate(cfg)
    tokens = args.tokens or 20 * e["total"]
    total_flops = e["flops_per_token"] * tokens
    seconds = total_flops / (args.tflops * 1e12 * args.mfu * args.gpus)
    name = args.preset or args.config or "config/config.py"
    ctx = cfg["context_length"]
    lines = [
        f"Model: {name} ({e['arch']}) | vocab {cfg['vocab_size']}, n_embed {cfg['n_embed']}, "
        f"{cfg['n_blocks']} blocks, {cfg['n_head']} heads, context {ctx}",
        "",
        "Parameters",
        f"  total                     {e['total']:>15,}  ({human(e['total'])})",
    ]
    if e["active"] != e["total"]:
        lines.append(f"  active per token (MoE)    {e['active']:>15,}  ({human(e['active'])})")
    lines += [
        f"  embedding tables          {e['lookup']:>15,}  ({100 * e['lookup'] / e['total']:.0f}% of the total)",
        f"  per block                 {e['per_block']:>15,}",
        "",
        "Compute",
        f"  training FLOPs per token  {human(e['flops_per_token'], 'FLOP')} (at {e['window']}-token windows)",
        f"  Chinchilla-optimal data   {human(20 * e['total'])} tokens (20 per parameter)",
        f"  training on {human(tokens)} tokens = {human(total_flops, 'FLOP')}",
        f"  at {args.tflops:g} TFLOP/s x {args.gpus} device(s) and {args.mfu:.0%} utilization: "
        f"{seconds / 3600:.1f} hours ({seconds / 86400:.1f} days)",
        "",
        "Memory",
        f"  weights (bf16, inference) {gib(2 * e['total'])}",
        f"  AdamW training state      {gib(e['adam_bytes'])} (fp32 weights + grads + 2 moments, 16 B/param)",
        f"  activations, batch {e['batch']:<4}   ~{gib(e['activation_bytes'])} (bf16; gradient checkpointing cuts most)",
        f"  KV cache per token        {e['kv_bytes_per_token'] / 1024:.1f} KiB (bf16), "
        f"{gib(e['kv_bytes_per_token'] * ctx)} for one full {ctx}-token context",
    ]
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--config", default=None, help="a JSON config, e.g. configs/base.json")
    p.add_argument("--arch", choices=["classic", "modern"], default=None)
    p.add_argument("--tokens", type=float, default=None, help="training tokens (default: Chinchilla-optimal)")
    p.add_argument("--tflops", type=float, default=100.0, help="peak TFLOP/s per device (bf16)")
    p.add_argument("--mfu", type=float, default=0.4, help="model FLOPs utilization, 0.3-0.5 is typical")
    p.add_argument("--gpus", type=int, default=1)
    args = p.parse_args()
    report(load_settings(args), args)


if __name__ == "__main__":
    main()

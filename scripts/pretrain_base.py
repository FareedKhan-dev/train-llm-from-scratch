"""
Pretrain the mid-size (~400M) base model from scratch on the Pile HDF5 corpus.

This is the shared starting checkpoint for every post-training stage. It upgrades the
original ``train_transformer.py`` recipe with the things needed to actually train a
mid-size model on 2x H100: DistributedDataParallel, bf16 autocast, gradient accumulation,
an LR schedule with warmup (cosine, WSD or linear), weight-decay param groups, the Muon
optimizer as an option, and periodic checkpointing. The original ``train_transformer.py``
is left untouched.

Single GPU:
    python scripts/pretrain_base.py
Both GPUs:
    torchrun --standalone --nproc_per_node=2 scripts/pretrain_base.py
The modern architecture with Muon and a warmup-stable-decay schedule:
    python scripts/pretrain_base.py --arch modern --n_kv_head 4 --optimizer muon --lr_schedule wsd

Override any config field from the CLI, e.g. ``--batch_size 16 --train_steps 50000``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the repo without installing

import contextlib
import os
import time
from dataclasses import asdict

import torch

from config.post_training_config import PretrainConfig
from data_loader.data_loader import get_batch_iterator
from src.checkpoint import load_checkpoint, model_state_from_checkpoint
from src.optim import build_optimizer, lr_at, set_lr
from src.post_training.cli import parse_config_with_json
from src.post_training.distributed import cleanup, ddp_setup, ddp_wrap, reduce_scalar
from src.post_training.logging_utils import MetricsLogger
from src.post_training.utils import (
    amp_autocast,
    build_model_from_config,
    save_stage_ckpt,
    set_seed,
    unwrap,
)


@torch.no_grad()
def estimate_loss(model, cfg, ctx, iters: int) -> dict[str, float]:
    model.eval()
    out = {}
    for split, path in [("train", cfg.train_path), ("dev", cfg.dev_path)]:
        if not os.path.exists(path):
            continue
        it = get_batch_iterator(path, cfg.batch_size, cfg.context_length, device=ctx.device)
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = next(it)
            with amp_autocast(cfg.amp_dtype, ctx.device):
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    cfg, extras = parse_config_with_json(
        PretrainConfig, "configs/pretrain.json",
        extra={"--resume": dict(type=str, default=None, help="checkpoint to resume from")})
    resume = extras.resume
    ctx = ddp_setup(cfg.device)
    # Different data shuffle per rank (the loader shuffles via numpy global RNG).
    set_seed(cfg.seed + ctx.rank)

    model = build_model_from_config(cfg).to(ctx.device)
    start_step = 0
    ck = None
    if resume and os.path.exists(resume):
        ck = load_checkpoint(resume, map_location="cpu")
        # Strip DDP / torch.compile prefixes so checkpoints from older runs load too (issue #36).
        model.load_state_dict(model_state_from_checkpoint(ck))
        # The saved step already finished its optimizer update, so continue with the next one.
        start_step = int(ck.get("step", -1)) + 1
        if ctx.is_main:
            print(f"Resumed from {resume}, continuing at step {start_step}")

    if cfg.compile:
        model = torch.compile(model)
    model = ddp_wrap(model, ctx)

    optimizer = build_optimizer(unwrap(model), cfg.optimizer, cfg.lr, cfg.weight_decay)
    if ck is not None and ck.get("optimizer_state_dict"):
        optimizer.load_state_dict(ck["optimizer_state_dict"])
    del ck

    logger = None
    if ctx.is_main:
        n_params = sum(p.numel() for p in unwrap(model).parameters())
        print(f"Model parameters: {n_params:,} (~{n_params/1e6:.0f}M) | arch={cfg.arch} | "
              f"optimizer={cfg.optimizer} | schedule={cfg.lr_schedule} | world_size={ctx.world_size}")
        print(f"Effective batch = {cfg.batch_size}*{cfg.grad_accum}*{ctx.world_size} "
              f"= {cfg.batch_size*cfg.grad_accum*ctx.world_size} seqs/step")
        logger = MetricsLogger("pretrain", cfg.log_dir, use_wandb=cfg.use_wandb,
                               wandb_project=cfg.wandb_project, config=asdict(cfg))

    batch_iter = get_batch_iterator(cfg.train_path, cfg.batch_size, cfg.context_length, device=ctx.device)
    tokens_per_step = cfg.batch_size * cfg.context_length * cfg.grad_accum * ctx.world_size

    model.train()
    t0 = time.perf_counter()
    for step in range(start_step, cfg.train_steps):
        lr = lr_at(cfg.lr_schedule, step, warmup_steps=cfg.warmup_steps, max_steps=cfg.train_steps,
                   lr=cfg.lr, min_lr=cfg.min_lr)
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(cfg.grad_accum):
            xb, yb = next(batch_iter)
            # Only sync grads on the last micro-step (DDP optimization).
            sync = (micro == cfg.grad_accum - 1) or not ctx.enabled
            cm = model.no_sync() if (ctx.enabled and not sync) else contextlib.nullcontext()
            with cm, amp_autocast(cfg.amp_dtype, ctx.device):
                _, loss = model(xb, yb)
                loss = loss / cfg.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if ctx.is_main and step % 20 == 0:
            dt = time.perf_counter() - t0
            tok_s = tokens_per_step * 20 / dt if step > start_step else 0.0
            t0 = time.perf_counter()
            print(f"step {step} | loss {accum_loss:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")
            if logger:
                logger.log(step, {"train_loss": accum_loss, "lr": lr, "tok_per_s": tok_s})

        if step > start_step and step % cfg.eval_steps == 0:
            ev = estimate_loss(model, cfg, ctx, cfg.eval_iters)
            ev = {k: reduce_scalar(v, ctx) for k, v in ev.items()}
            if ctx.is_main:
                print(f"  [eval] step {step} | " + " | ".join(f"{k} {v:.4f}" for k, v in ev.items()))
                if logger:
                    logger.log(step, {f"eval_{k}": v for k, v in ev.items()})

        if ctx.is_main and step > start_step and step % cfg.save_every == 0:
            save_stage_ckpt(cfg.out_ckpt, model, optimizer, stage="pretrain",
                            cfg=cfg, step=step, metrics={"train_loss": accum_loss})
            print(f"  saved checkpoint -> {cfg.out_ckpt} (step {step})")

    if ctx.is_main:
        save_stage_ckpt(cfg.out_ckpt, model, optimizer, stage="pretrain", cfg=cfg,
                        step=cfg.train_steps, metrics={"train_loss": accum_loss})
        print(f"Done. Final checkpoint -> {cfg.out_ckpt}")
        if logger:
            logger.close()
    cleanup(ctx)


if __name__ == "__main__":
    main()

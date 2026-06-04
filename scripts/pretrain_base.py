"""
Pretrain the mid-size (~400M) base model from scratch on the Pile HDF5 corpus.

This is the shared starting checkpoint for every post-training stage. It upgrades the
original ``train_transformer.py`` recipe with the things needed to actually train a
mid-size model on 2x H100: DistributedDataParallel, bf16 autocast, gradient accumulation,
a cosine LR schedule with warmup, weight-decay param groups, and periodic checkpointing.
The original ``train_transformer.py`` is left untouched.

Single GPU:
    PYTHONPATH=. python scripts/pretrain_base.py
Both GPUs:
    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/pretrain_base.py

Override any config field from the CLI, e.g. ``--batch_size 16 --train_steps 50000``.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from config.post_training_config import PretrainConfig
from data_loader.data_loader import get_batch_iterator
from src.post_training.cli import parse_config_with_json
from src.post_training.distributed import ddp_setup, ddp_wrap, cleanup, reduce_scalar
from src.post_training.logging_utils import MetricsLogger
from src.post_training.optim import configure_optimizer, cosine_lr
from src.post_training.utils import (
    amp_autocast, build_model_from_config, save_stage_ckpt, set_seed, unwrap,
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
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(ck["model_state_dict"])
        start_step = ck.get("step", 0)
        if ctx.is_main:
            print(f"Resumed from {resume} at step {start_step}")

    if cfg.compile:
        model = torch.compile(model)
    model = ddp_wrap(model, ctx)

    optimizer = configure_optimizer(unwrap(model), cfg.lr, cfg.weight_decay)
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if ck.get("optimizer_state_dict"):
            optimizer.load_state_dict(ck["optimizer_state_dict"])

    logger = None
    if ctx.is_main:
        n_params = sum(p.numel() for p in unwrap(model).parameters())
        print(f"Model parameters: {n_params:,} (~{n_params/1e6:.0f}M) | world_size={ctx.world_size}")
        print(f"Effective batch = {cfg.batch_size}*{cfg.grad_accum}*{ctx.world_size} "
              f"= {cfg.batch_size*cfg.grad_accum*ctx.world_size} seqs/step")
        logger = MetricsLogger("pretrain", cfg.log_dir, use_wandb=cfg.use_wandb,
                               wandb_project=cfg.wandb_project, config=vars(cfg).copy() if hasattr(cfg, "__dict__") else None)

    batch_iter = get_batch_iterator(cfg.train_path, cfg.batch_size, cfg.context_length, device=ctx.device)
    tokens_per_step = cfg.batch_size * cfg.context_length * cfg.grad_accum * ctx.world_size

    model.train()
    t0 = time.perf_counter()
    for step in range(start_step, cfg.train_steps):
        lr = cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=cfg.train_steps,
                       lr=cfg.lr, min_lr=cfg.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(cfg.grad_accum):
            xb, yb = next(batch_iter)
            # Only sync grads on the last micro-step (DDP optimization).
            sync = (micro == cfg.grad_accum - 1) or not ctx.enabled
            cm = model.no_sync() if (ctx.enabled and not sync) else _nullcm()
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


import contextlib


def _nullcm():
    return contextlib.nullcontext()


if __name__ == "__main__":
    main()

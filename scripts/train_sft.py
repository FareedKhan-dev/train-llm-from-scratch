"""
Supervised Fine-Tuning of the pretrained base on packed instruction data.

Loads the base checkpoint, trains with the prompt-masked SFT loss, periodically reports
masked dev loss, and saves an SFT checkpoint. DDP + bf16, single code path for 1 or N GPUs.

Single GPU:
    PYTHONPATH=. python scripts/train_sft.py
Both GPUs:
    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_sft.py
"""

from __future__ import annotations

import contextlib
import math
import time

import torch

from config.post_training_config import SFTConfig
from data_loader.sft_dataset import get_sft_batch_iterator
from src.post_training.cli import parse_config_with_json
from src.post_training.distributed import ddp_setup, ddp_wrap, cleanup, reduce_scalar
from src.post_training.logging_utils import MetricsLogger
from src.post_training.optim import configure_optimizer, cosine_lr
from src.post_training.sft import sft_loss
from src.post_training.utils import amp_autocast, load_backbone_from_ckpt, save_stage_ckpt, set_seed, unwrap

DEV_PATH = "/ephemeral/data/sft_dev_packed.h5"


@torch.no_grad()
def eval_dev(model, cfg, ctx, dev_path: str, max_batches: int = 50) -> float:
    model.eval()
    it = get_sft_batch_iterator(dev_path, cfg.batch_size, device=ctx.device,
                                rank=ctx.rank, world_size=ctx.world_size, shuffle=False, infinite=False)
    total, n = 0.0, 0
    for tokens, mask, _ in it:
        with amp_autocast(cfg.amp_dtype, ctx.device):
            logits, _ = model(tokens)
            loss = sft_loss(logits, tokens, mask)
        total += loss.item(); n += 1
        if n >= max_batches:
            break
    model.train()
    return total / max(1, n)


def main():
    cfg, _ = parse_config_with_json(SFTConfig, "configs/sft.json")
    ctx = ddp_setup(cfg.device)
    set_seed(cfg.seed + ctx.rank)

    model = load_backbone_from_ckpt(cfg, cfg.pretrained_ckpt, ctx.device)
    if cfg.compile:
        model = torch.compile(model)
    model = ddp_wrap(model, ctx)
    optimizer = configure_optimizer(unwrap(model), cfg.lr, cfg.weight_decay)

    logger = None
    if ctx.is_main:
        print(f"SFT from {cfg.pretrained_ckpt} | world_size={ctx.world_size}")
        logger = MetricsLogger("sft", cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project)

    train_it = get_sft_batch_iterator(cfg.data_path, cfg.batch_size, device=ctx.device,
                                      rank=ctx.rank, world_size=ctx.world_size, shuffle=True, infinite=True)

    # Estimate total steps for the cosine schedule from dataset size.
    import h5py
    with h5py.File(cfg.data_path, "r") as f:
        n_rows = f["tokens"].shape[0]
    steps_per_epoch = max(1, n_rows // (cfg.batch_size * ctx.world_size))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.epochs
    if ctx.is_main:
        print(f"{n_rows} packed rows | ~{steps_per_epoch} steps/epoch | total_steps={total_steps}")

    model.train()
    t0 = time.perf_counter()
    for step in range(total_steps):
        lr = cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=total_steps, lr=cfg.lr, min_lr=cfg.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        tokens, mask, epoch = next(train_it)
        if epoch >= cfg.epochs and cfg.max_steps <= 0:
            break
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(cfg.amp_dtype, ctx.device):
            logits, _ = model(tokens)
            loss = sft_loss(logits, tokens, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if ctx.is_main and step % 20 == 0:
            dt = time.perf_counter() - t0; t0 = time.perf_counter()
            print(f"step {step}/{total_steps} | loss {loss.item():.4f} | ppl {math.exp(min(20, loss.item())):.2f} | lr {lr:.2e} | {dt:.1f}s/20")
            if logger:
                logger.log(step, {"train_loss": loss.item(), "lr": lr})

        if step > 0 and step % cfg.eval_steps == 0:
            dev = reduce_scalar(eval_dev(model, cfg, ctx, DEV_PATH), ctx)
            if ctx.is_main:
                print(f"  [eval] step {step} | dev_loss {dev:.4f} | dev_ppl {math.exp(min(20, dev)):.2f}")
                if logger:
                    logger.log(step, {"dev_loss": dev})

        if ctx.is_main and step > 0 and step % cfg.save_every == 0:
            save_stage_ckpt(cfg.out_ckpt, model, optimizer, stage="sft", cfg=cfg, step=step,
                            metrics={"train_loss": loss.item()})

    if ctx.is_main:
        # Use the unwrapped model for the final eval: the other ranks have already reached
        # cleanup(), so calling the DDP-wrapped model here would launch a collective with no
        # peer and hang (NCCL timeout). The periodic eval above runs on all ranks, so it is fine.
        dev = eval_dev(unwrap(model), cfg, ctx, DEV_PATH)
        save_stage_ckpt(cfg.out_ckpt, model, optimizer, stage="sft", cfg=cfg, step=total_steps,
                        metrics={"dev_loss": dev})
        print(f"Done SFT. dev_loss {dev:.4f} -> {cfg.out_ckpt}")
        if logger:
            logger.close()
    cleanup(ctx)


if __name__ == "__main__":
    main()

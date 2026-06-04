"""
Train the reward model on preference pairs with the Bradley-Terry loss.

Initializes the reward backbone from the SFT checkpoint, adds a scalar reward head, and
trains so chosen responses score above rejected ones. Reports held-out preference accuracy.

Single GPU:
    PYTHONPATH=. python scripts/train_reward.py
Both GPUs:
    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_reward.py
"""

from __future__ import annotations

import time

import torch

from config.post_training_config import RewardConfig
from data_loader.preference_dataset import get_preference_iterator
from src.post_training.cli import parse_config_with_json
from src.post_training.distributed import ddp_setup, ddp_wrap, cleanup, reduce_scalar
from src.post_training.logging_utils import MetricsLogger
from src.post_training.optim import configure_optimizer, cosine_lr
from src.post_training.reward_model import RewardModel
from src.post_training.reward_train import bradley_terry_loss, preference_accuracy, reward_margin
from src.post_training.utils import amp_autocast, load_backbone_from_ckpt, save_stage_ckpt, set_seed, unwrap

TEST_PATH = "/ephemeral/data/preferences_test.jsonl"


def _pair_rewards(rm, batch, cfg, ctx):
    """Forward chosen+rejected in one pass; return (chosen_rewards, rejected_rewards)."""
    B = batch["chosen_ids"].size(0)
    ids = torch.cat([batch["chosen_ids"], batch["rejected_ids"]], dim=0)
    lens = torch.cat([batch["chosen_len"], batch["rejected_len"]], dim=0)
    with amp_autocast(cfg.amp_dtype, ctx.device):
        rewards = rm(ids, seq_lengths=lens).float()
    return rewards[:B], rewards[B:]


@torch.no_grad()
def eval_accuracy(rm, cfg, ctx, max_batches: int = 100) -> tuple[float, float]:
    rm.eval()
    it = get_preference_iterator(TEST_PATH, cfg.batch_size, cfg.max_len, device=ctx.device,
                                 rank=ctx.rank, world_size=ctx.world_size, shuffle=False, infinite=False)
    acc, marg, n = 0.0, 0.0, 0
    for batch in it:
        cr, rr = _pair_rewards(rm, batch, cfg, ctx)
        acc += preference_accuracy(cr, rr).item()
        marg += reward_margin(cr, rr).item()
        n += 1
        if n >= max_batches:
            break
    rm.train()
    return acc / max(1, n), marg / max(1, n)


def main():
    cfg, _ = parse_config_with_json(RewardConfig, "configs/reward.json")
    ctx = ddp_setup(cfg.device)
    set_seed(cfg.seed + ctx.rank)

    backbone = load_backbone_from_ckpt(cfg, cfg.sft_ckpt, ctx.device)
    rm = RewardModel(backbone).to(ctx.device)
    rm = ddp_wrap(rm, ctx)
    optimizer = configure_optimizer(unwrap(rm), cfg.lr, cfg.weight_decay)

    import json
    with open(cfg.pref_path) as f:
        n_rows = sum(1 for line in f if line.strip())
    total_steps = max(1, (n_rows // (cfg.batch_size * ctx.world_size)) * cfg.epochs)

    logger = None
    if ctx.is_main:
        print(f"Reward model from {cfg.sft_ckpt} | {n_rows} pairs | total_steps={total_steps}")
        logger = MetricsLogger("reward", cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project)

    train_it = get_preference_iterator(cfg.pref_path, cfg.batch_size, cfg.max_len, device=ctx.device,
                                       rank=ctx.rank, world_size=ctx.world_size, shuffle=True, infinite=True)

    rm.train()
    t0 = time.perf_counter()
    for step in range(total_steps):
        lr = cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=total_steps, lr=cfg.lr, min_lr=cfg.lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch = next(train_it)
        cr, rr = _pair_rewards(rm, batch, cfg, ctx)
        loss = bradley_terry_loss(cr, rr)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), cfg.grad_clip)
        optimizer.step()

        if ctx.is_main and step % 20 == 0:
            acc = preference_accuracy(cr, rr).item()
            dt = time.perf_counter() - t0; t0 = time.perf_counter()
            print(f"step {step}/{total_steps} | loss {loss.item():.4f} | train_acc {acc:.3f} | lr {lr:.2e} | {dt:.1f}s/20")
            if logger:
                logger.log(step, {"train_loss": loss.item(), "train_acc": acc, "lr": lr})

        if step > 0 and step % cfg.eval_steps == 0:
            acc, marg = eval_accuracy(rm, cfg, ctx)
            acc, marg = reduce_scalar(acc, ctx), reduce_scalar(marg, ctx)
            if ctx.is_main:
                print(f"  [eval] step {step} | test_acc {acc:.3f} | margin {marg:.3f}")
                if logger:
                    logger.log(step, {"test_acc": acc, "test_margin": marg})

        if ctx.is_main and step > 0 and step % cfg.save_every == 0:
            save_stage_ckpt(cfg.out_ckpt, rm, optimizer, stage="reward", cfg=cfg, step=step,
                            metrics={"train_loss": loss.item()})

    if ctx.is_main:
        acc, marg = eval_accuracy(rm, cfg, ctx)
        save_stage_ckpt(cfg.out_ckpt, rm, optimizer, stage="reward", cfg=cfg, step=total_steps,
                        metrics={"test_acc": acc, "test_margin": marg})
        print(f"Done RM. test_acc {acc:.3f} margin {marg:.3f} -> {cfg.out_ckpt}")
        if logger:
            logger.close()
    cleanup(ctx)


if __name__ == "__main__":
    main()

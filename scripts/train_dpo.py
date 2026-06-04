"""
Direct Preference Optimization (and ORPO / KTO variants) on preference pairs.

The policy is initialized from the SFT checkpoint; a frozen deep copy of it serves as the
DPO/KTO reference (ORPO is reference-free). Reports implicit-reward accuracy on held-out
preferences and GSM8K dev accuracy.

    PYTHONPATH=. python scripts/train_dpo.py --loss_type dpo --beta 0.1
    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_dpo.py
"""

from __future__ import annotations

import time

import torch

from config.post_training_config import DPOConfig
from data_loader.preference_dataset import get_preference_iterator
from src.post_training.cli import parse_config_with_json
from src.post_training.distributed import ddp_setup, ddp_wrap, cleanup, reduce_scalar
from src.post_training.dpo import dpo_loss, orpo_loss, kto_loss, implicit_accuracy
from src.post_training.logging_utils import MetricsLogger
from src.post_training.optim import configure_optimizer, cosine_lr
from src.post_training.rollout import sequence_logprobs
from src.post_training.utils import (
    amp_autocast, load_backbone_from_ckpt, make_frozen_copy, save_stage_ckpt, set_seed, unwrap,
)

TEST_PATH = "/ephemeral/data/preferences_test.jsonl"


def _logps(model, ids, mask, requires_grad):
    return sequence_logprobs(model, ids, mask, requires_grad=requires_grad)


def _compute_losses(policy, ref, batch, cfg, ctx):
    B = batch["chosen_ids"].size(0)
    ids = torch.cat([batch["chosen_ids"], batch["rejected_ids"]], dim=0)
    mask = torch.cat([batch["chosen_mask"], batch["rejected_mask"]], dim=0)

    with amp_autocast(cfg.amp_dtype, ctx.device):
        psum, pn = _logps(policy, ids, mask, requires_grad=True)
    pc, pr, ncn, nrn = psum[:B], psum[B:], pn[:B], pn[B:]

    if cfg.loss_type == "orpo":
        return orpo_loss(pc, pr, ncn, nrn, orpo_lambda=cfg.orpo_lambda)

    with torch.no_grad(), amp_autocast(cfg.amp_dtype, ctx.device):
        rsum, _ = _logps(ref, ids, mask, requires_grad=False)
    rc, rr = rsum[:B], rsum[B:]
    if cfg.loss_type == "kto":
        return kto_loss(pc, pr, rc, rr, beta=cfg.beta)
    return dpo_loss(pc, pr, rc, rr, beta=cfg.beta)


@torch.no_grad()
def eval_implicit_acc(policy, ref, cfg, ctx, max_batches: int = 100) -> tuple[float, float]:
    policy.eval()
    it = get_preference_iterator(TEST_PATH, cfg.batch_size, cfg.max_len, device=ctx.device,
                                 rank=ctx.rank, world_size=ctx.world_size, shuffle=False, infinite=False)
    acc, marg, n = 0.0, 0.0, 0
    for batch in it:
        loss, cr, rr = _compute_losses(policy, ref, batch, cfg, ctx)
        acc += implicit_accuracy(cr, rr).item()
        marg += (cr - rr).mean().item()
        n += 1
        if n >= max_batches:
            break
    policy.train()
    return acc / max(1, n), marg / max(1, n)


def main():
    cfg, _ = parse_config_with_json(DPOConfig, "configs/dpo.json")
    ctx = ddp_setup(cfg.device)
    set_seed(cfg.seed + ctx.rank)

    policy = load_backbone_from_ckpt(cfg, cfg.sft_ckpt, ctx.device)
    ref = make_frozen_copy(policy, device=ctx.device) if cfg.loss_type != "orpo" else None
    policy = ddp_wrap(policy, ctx)
    optimizer = configure_optimizer(unwrap(policy), cfg.lr, cfg.weight_decay)

    with open(cfg.pref_path) as f:
        n_rows = sum(1 for line in f if line.strip())
    total_steps = max(1, (n_rows // (cfg.batch_size * ctx.world_size)) * cfg.epochs)

    logger = None
    if ctx.is_main:
        print(f"DPO[{cfg.loss_type}] from {cfg.sft_ckpt} | {n_rows} pairs | total_steps={total_steps} | beta={cfg.beta}")
        logger = MetricsLogger(f"dpo_{cfg.loss_type}", cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project)

    train_it = get_preference_iterator(cfg.pref_path, cfg.batch_size, cfg.max_len, device=ctx.device,
                                       rank=ctx.rank, world_size=ctx.world_size, shuffle=True, infinite=True)

    policy.train()
    t0 = time.perf_counter()
    for step in range(total_steps):
        lr = cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=total_steps, lr=cfg.lr, min_lr=cfg.lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch = next(train_it)
        loss, cr, rr = _compute_losses(policy, ref, batch, cfg, ctx)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        if ctx.is_main and step % 20 == 0:
            acc = implicit_accuracy(cr, rr).item()
            dt = time.perf_counter() - t0; t0 = time.perf_counter()
            print(f"step {step}/{total_steps} | loss {loss.item():.4f} | acc {acc:.3f} | "
                  f"r_chosen {cr.mean().item():.3f} r_rejected {rr.mean().item():.3f} | {dt:.1f}s/20")
            if logger:
                logger.log(step, {"train_loss": loss.item(), "train_acc": acc,
                                  "r_chosen": cr.mean().item(), "r_rejected": rr.mean().item(), "lr": lr})

        if step > 0 and step % cfg.eval_steps == 0:
            acc, marg = eval_implicit_acc(policy, ref, cfg, ctx)
            acc, marg = reduce_scalar(acc, ctx), reduce_scalar(marg, ctx)
            if ctx.is_main:
                print(f"  [eval] step {step} | test_acc {acc:.3f} | margin {marg:.3f}")
                if logger:
                    logger.log(step, {"test_acc": acc, "test_margin": marg})

        if ctx.is_main and step > 0 and step % cfg.save_every == 0:
            save_stage_ckpt(cfg.out_ckpt, policy, optimizer, stage=f"dpo_{cfg.loss_type}", cfg=cfg, step=step,
                            metrics={"train_loss": loss.item()})

    if ctx.is_main:
        acc, marg = eval_implicit_acc(policy, ref, cfg, ctx)
        save_stage_ckpt(cfg.out_ckpt, policy, optimizer, stage=f"dpo_{cfg.loss_type}", cfg=cfg, step=total_steps,
                        metrics={"test_acc": acc, "test_margin": marg})
        print(f"Done DPO[{cfg.loss_type}]. test_acc {acc:.3f} margin {marg:.3f} -> {cfg.out_ckpt}")
        if logger:
            logger.close()
    cleanup(ctx)


if __name__ == "__main__":
    main()

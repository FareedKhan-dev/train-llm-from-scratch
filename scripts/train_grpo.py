"""
GRPO / RLVR on GSM8K (DeepSeek-R1 style), from scratch -- no critic, group-relative
advantages, verifiable reward.

Per iteration: for each prompt sample a group of G completions, score each with the GSM8K
verifier, compute group-relative advantages, and update with a clipped surrogate +
KL-to-reference penalty. An arithmetic warm-up curriculum runs first so the policy gets
non-zero reward variance before facing full GSM8K.

The 2025 variants are flags (see src/post_training/grpo.py for what each one changes):

    python scripts/train_grpo.py                                            # GRPO (repo default)
    python scripts/train_grpo.py --adv_norm none --loss_agg seq-mean-token-sum-norm   # Dr. GRPO
    python scripts/train_grpo.py --clip_high 0.28 --filter_groups true --kl_coef 0    # DAPO
    python scripts/train_grpo.py --ratio_level sequence --clip 0.0003 --clip_high 0.0004  # GSPO
    torchrun --standalone --nproc_per_node=2 scripts/train_grpo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the repo without installing

import time

import torch

from config.post_training_config import GRPOConfig
from data_loader.prompt_dataset import get_prompt_iterator
from src.post_training.cli import parse_config_with_json
from src.post_training.chat_template import decode, encode_prompt
from src.post_training.distributed import ddp_setup, ddp_wrap, cleanup, reduce_scalar
from src.post_training.evaluation import gsm8k_accuracy, load_gsm8k_eval
from src.post_training.grpo import group_advantages, grpo_loss
from src.post_training.logging_utils import MetricsLogger
from src.post_training.optim import configure_optimizer
from src.post_training.rewards import reward_gsm8k
from src.post_training.rollout import compute_logprobs, rollout_prompts
from src.post_training.utils import (
    amp_autocast, load_backbone_from_ckpt, make_frozen_copy, save_stage_ckpt, set_seed, unwrap,
)


def seq_lengths_from_mask(response_mask, prompt_lens):
    N, T = response_mask.shape
    pos = torch.arange(T, device=response_mask.device)
    last = torch.where(response_mask, pos[None, :], torch.full_like(response_mask, -1, dtype=torch.long)).max(dim=1).values
    return torch.where(last >= 0, last + 1, prompt_lens)


def main():
    cfg, _ = parse_config_with_json(GRPOConfig, "configs/grpo.json")
    ctx = ddp_setup(cfg.device)
    set_seed(cfg.seed + ctx.rank)

    policy = load_backbone_from_ckpt(cfg, cfg.sft_ckpt, ctx.device)
    ref = make_frozen_copy(policy, device=ctx.device)
    policy_ddp = ddp_wrap(policy, ctx)
    optimizer = configure_optimizer(unwrap(policy_ddp), cfg.lr, weight_decay=0.0)

    eval_set = None
    logger = MetricsLogger("grpo", cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project) if ctx.is_main else None
    if ctx.is_main:
        print(f"GRPO from {cfg.sft_ckpt} | group_size={cfg.group_size} | world={ctx.world_size}")

    warm_it = get_prompt_iterator(cfg.curriculum_path, cfg.prompts_per_iter, rank=ctx.rank,
                                  world_size=ctx.world_size, seed=cfg.seed)
    main_it = get_prompt_iterator(cfg.prompt_path, cfg.prompts_per_iter, rank=ctx.rank,
                                  world_size=ctx.world_size, seed=cfg.seed)

    G = cfg.group_size
    for it in range(cfg.iterations):
        rows = next(warm_it if it < cfg.curriculum_iters else main_it)
        # Replicate each prompt G times, group-contiguously.
        base_prompts = [encode_prompt([{"role": "user", "content": r["prompt"]}]) for r in rows]
        prompts = [p for p in base_prompts for _ in range(G)]
        golds = [r.get("gold") for r in rows for _ in range(G)]

        policy.eval()
        with amp_autocast(cfg.amp_dtype, ctx.device):
            seqs, rmask, plens = rollout_prompts(policy, prompts, cfg.rollout_len, device=ctx.device,
                                                 temperature=cfg.temperature, top_p=cfg.top_p if cfg.top_p < 1 else None)
        resp = rmask[:, 1:]

        seq_lens = seq_lengths_from_mask(rmask, plens)
        responses = [decode(seqs[i, plens[i]:seq_lens[i]].tolist()) for i in range(len(prompts))]
        rewards = torch.tensor([reward_gsm8k(responses[i], golds[i]) for i in range(len(prompts))],
                               device=ctx.device, dtype=torch.float32)
        adv = group_advantages(rewards, G, scale=cfg.adv_norm)

        with torch.no_grad(), amp_autocast(cfg.amp_dtype, ctx.device):
            old_logp, _ = compute_logprobs(policy, seqs, rmask, temperature=cfg.temperature, requires_grad=False)
            ref_logp, _ = compute_logprobs(ref, seqs, rmask, temperature=cfg.temperature, requires_grad=False)
        old_logp, ref_logp = old_logp.float(), ref_logp.float()

        # DAPO dynamic sampling: a group where every answer got the same reward has zero
        # advantage everywhere, so it only adds the KL term and compute. Its tokens are masked
        # out of the loss. (Rows are masked, not removed, so every GPU still runs the same
        # number of backward passes, which DDP requires.)
        keep = torch.ones(seqs.size(0), dtype=torch.bool, device=ctx.device)
        if cfg.filter_groups:
            keep = (rewards.view(-1, G).std(dim=1) > 1e-6).repeat_interleave(G)
        train_mask = resp & keep[:, None]

        policy.train()
        N = seqs.size(0)
        agg = {"loss": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
        for _ in range(cfg.grpo_epochs):
            perm = torch.randperm(N, device=ctx.device)
            for s in range(0, N, max(1, G)):  # minibatch ~ one group's worth
                mb = perm[s:s + max(1, G)]
                if not ctx.enabled and not keep[mb].any():
                    continue  # nothing to learn from these rows (only safe to skip on a single process)
                with amp_autocast(cfg.amp_dtype, ctx.device):
                    new_logp, _ = compute_logprobs(policy_ddp, seqs[mb], rmask[mb], temperature=cfg.temperature, requires_grad=True)
                loss, st = grpo_loss(new_logp.float(), old_logp[mb], ref_logp[mb], adv[mb], train_mask[mb],
                                     clip=cfg.clip, kl_coef=cfg.kl_coef, clip_high=cfg.clip_high,
                                     loss_agg=cfg.loss_agg, ratio_level=cfg.ratio_level, max_len=cfg.rollout_len)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_ddp.parameters(), cfg.grad_clip)
                optimizer.step()
                agg["loss"] += loss.item(); agg["kl"] += st["kl"]; agg["clipfrac"] += st["clipfrac"]; agg["n"] += 1

        mean_reward = reduce_scalar(rewards.mean().item(), ctx)
        # Fraction of groups with non-zero reward spread (informative groups).
        grp_std = rewards.view(-1, G).std(dim=1)
        informative = reduce_scalar((grp_std > 1e-6).float().mean().item(), ctx)
        resp_len = reduce_scalar(resp.float().sum(1).mean().item(), ctx)
        n = max(1, agg["n"])
        if ctx.is_main and it % 5 == 0:
            phase = "warmup" if it < cfg.curriculum_iters else "gsm8k"
            print(f"iter {it}[{phase}] | reward {mean_reward:.3f} | informative {informative:.2f} | "
                  f"loss {agg['loss']/n:.4f} | KL {agg['kl']/n:.4f} | clipfrac {agg['clipfrac']/n:.3f} | resp_len {resp_len:.0f}")
            if logger:
                logger.log(it, {"reward": mean_reward, "informative_groups": informative,
                                "loss": agg["loss"]/n, "kl": agg["kl"]/n, "resp_len": resp_len})

        if ctx.is_main and it > 0 and it % cfg.eval_every == 0:
            if eval_set is None:
                eval_set = load_gsm8k_eval("test", limit=200)
            res = gsm8k_accuracy(unwrap(policy_ddp), eval_set, device=ctx.device, max_new_tokens=cfg.rollout_len)
            print(f"  [eval] iter {it} | GSM8K test acc {res['accuracy']:.3f} ({res['correct']}/{res['n']})")
            if logger:
                logger.log(it, {"gsm8k_acc": res["accuracy"]})
        if ctx.is_main and it > 0 and it % cfg.save_every == 0:
            save_stage_ckpt(cfg.out_ckpt, unwrap(policy_ddp), optimizer, stage="grpo", cfg=cfg, step=it,
                            metrics={"reward": mean_reward})

    if ctx.is_main:
        save_stage_ckpt(cfg.out_ckpt, unwrap(policy_ddp), optimizer, stage="grpo", cfg=cfg, step=cfg.iterations, metrics={})
        print(f"Done GRPO -> {cfg.out_ckpt}")
        if logger:
            logger.close()
    cleanup(ctx)


if __name__ == "__main__":
    main()

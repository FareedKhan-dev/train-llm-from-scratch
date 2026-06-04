"""
PPO RLHF on GSM8K (the classic InstructGPT recipe), from scratch.

Per iteration: roll out completions with the current policy, score them (verifiable GSM8K
reward, or a trained reward model), add a per-token KL-to-reference penalty, compute GAE
advantages with the shared value head, then run several clipped-surrogate update epochs.
Reports mean reward, KL, value loss, clip fraction, and held-out GSM8K accuracy.

    PYTHONPATH=. python scripts/train_ppo.py --reward_source verifier
    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/train_ppo.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from config.post_training_config import PPOConfig
from data_loader.prompt_dataset import get_prompt_iterator
from src.post_training.cli import parse_config_with_json
from src.post_training.chat_template import EOT_ID, decode, encode_prompt
from src.post_training.distributed import ddp_setup, ddp_wrap, cleanup, reduce_scalar
from src.post_training.evaluation import gsm8k_accuracy, load_gsm8k_eval
from src.post_training.logging_utils import MetricsLogger
from src.post_training.optim import configure_optimizer
from src.post_training.ppo import compute_gae, whiten, ppo_policy_loss, ppo_value_loss, approx_kl
from src.post_training.reward_model import load_reward_model
from src.post_training.rewards import reward_gsm8k
from src.post_training.rollout import compute_logprobs, rollout_prompts
from src.post_training.utils import (
    amp_autocast, load_backbone_from_ckpt, make_frozen_copy, masked_mean, save_stage_ckpt, set_seed, unwrap,
)
from src.post_training.value_head import TransformerWithValueHead


def actor_logp_values(actor, seqs, temperature):
    """One forward through the actor-critic -> (logp, values) in the action frame (B,T-1)."""
    logits, values = actor(seqs)
    logits = logits[:, :-1, :]
    logp_all = F.log_softmax(logits.float() / max(temperature, 1e-6), dim=-1)
    logp = logp_all.gather(-1, seqs[:, 1:, None]).squeeze(-1)
    return logp, values[:, :-1]


def seq_lengths_from_mask(response_mask, prompt_lens):
    """Real token count per row = last response position + 1 (or prompt_len if no response)."""
    N, T = response_mask.shape
    pos = torch.arange(T, device=response_mask.device)
    last = torch.where(response_mask, pos[None, :], torch.full_like(response_mask, -1, dtype=torch.long)).max(dim=1).values
    return torch.where(last >= 0, last + 1, prompt_lens)


def main():
    cfg, _ = parse_config_with_json(PPOConfig, "configs/ppo.json")
    ctx = ddp_setup(cfg.device)
    set_seed(cfg.seed + ctx.rank)

    backbone = load_backbone_from_ckpt(cfg, cfg.sft_ckpt, ctx.device)
    ref = make_frozen_copy(backbone, device=ctx.device)
    actor = TransformerWithValueHead(backbone).to(ctx.device)
    actor_ddp = ddp_wrap(actor, ctx)
    optimizer = configure_optimizer(unwrap(actor_ddp), cfg.lr, weight_decay=0.0)

    rm = load_reward_model(cfg, cfg.reward_ckpt, ctx.device) if cfg.reward_source == "rm" else None

    eval_set = None  # loaded lazily on first eval to keep startup/smoke runs offline
    logger = MetricsLogger("ppo", cfg.log_dir, use_wandb=cfg.use_wandb, wandb_project=cfg.wandb_project) if ctx.is_main else None
    if ctx.is_main:
        print(f"PPO from {cfg.sft_ckpt} | reward={cfg.reward_source} | world={ctx.world_size}")

    prompt_it = get_prompt_iterator(cfg.prompt_path, cfg.prompts_per_iter, rank=ctx.rank,
                                    world_size=ctx.world_size, seed=cfg.seed)

    for it in range(cfg.iterations):
        rows = next(prompt_it)
        prompts = [encode_prompt([{"role": "user", "content": r["prompt"]}]) for r in rows]
        golds = [r.get("gold") for r in rows]

        # --- rollout ---
        actor.eval()
        with amp_autocast(cfg.amp_dtype, ctx.device):
            seqs, rmask, plens = rollout_prompts(actor, prompts, cfg.rollout_len, device=ctx.device,
                                                 temperature=cfg.temperature, top_p=cfg.top_p if cfg.top_p < 1 else None)
        resp = rmask[:, 1:]                       # action-frame response mask (N, T-1)

        # --- score ---
        seq_lens = seq_lengths_from_mask(rmask, plens)
        responses = [decode(seqs[i, plens[i]:seq_lens[i]].tolist()) for i in range(len(rows))]
        if cfg.reward_source == "rm":
            with torch.no_grad(), amp_autocast(cfg.amp_dtype, ctx.device):
                task_r = rm(seqs, seq_lengths=seq_lens).float().tolist()
        else:
            task_r = [reward_gsm8k(responses[i], golds[i]) for i in range(len(rows))]

        # --- per-token rewards: KL penalty everywhere + task reward at last response token ---
        with torch.no_grad(), amp_autocast(cfg.amp_dtype, ctx.device):
            old_logp, old_values = actor_logp_values(actor, seqs, cfg.temperature)
            ref_logp, _ = compute_logprobs(ref, seqs, rmask, temperature=cfg.temperature, requires_grad=False)
        old_logp, old_values, ref_logp = old_logp.float(), old_values.float(), ref_logp.float()

        rewards = -cfg.kl_coef * (old_logp - ref_logp) * resp.float()
        last_idx = seq_lens - 2                    # action index of the last response token
        last_idx = last_idx.clamp(min=0)
        task_t = torch.tensor(task_r, device=ctx.device, dtype=torch.float32)
        rewards[torch.arange(len(rows), device=ctx.device), last_idx] += task_t

        values_next = torch.cat([old_values[:, 1:], torch.zeros_like(old_values[:, :1])], dim=1)
        adv, returns = compute_gae(rewards, old_values, values_next, resp, gamma=cfg.gamma, lam=cfg.gae_lambda)
        adv = whiten(adv, resp)

        # --- clipped PPO update epochs over minibatches of rollout rows ---
        actor.train()
        N = seqs.size(0)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "clipfrac": 0.0, "kl": 0.0, "n": 0}
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(N, device=ctx.device)
            for s in range(0, N, cfg.minibatch_size):
                mb = perm[s:s + cfg.minibatch_size]
                with amp_autocast(cfg.amp_dtype, ctx.device):
                    new_logp, new_values = actor_logp_values(actor_ddp, seqs[mb], cfg.temperature)
                m = resp[mb]
                p_loss, clipf = ppo_policy_loss(new_logp.float(), old_logp[mb], adv[mb], m, clip=cfg.clip)
                v_loss = ppo_value_loss(new_values.float(), old_values[mb], returns[mb], m, vf_clip=cfg.vf_clip)
                loss = p_loss + cfg.vf_coef * v_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_ddp.parameters(), cfg.grad_clip)
                optimizer.step()
                stats["policy_loss"] += p_loss.item(); stats["value_loss"] += v_loss.item()
                stats["clipfrac"] += clipf.item()
                stats["kl"] += approx_kl(new_logp.float(), old_logp[mb], m).item(); stats["n"] += 1

        mean_reward = reduce_scalar(float(sum(task_r) / max(1, len(task_r))), ctx)
        kl_ref = reduce_scalar(masked_mean(old_logp - ref_logp, resp).item(), ctx)
        resp_len = reduce_scalar(resp.float().sum(1).mean().item(), ctx)
        n = max(1, stats["n"])
        if ctx.is_main and it % 5 == 0:
            print(f"iter {it} | reward {mean_reward:.3f} | KL_ref {kl_ref:.3f} | "
                  f"ploss {stats['policy_loss']/n:.4f} | vloss {stats['value_loss']/n:.4f} | "
                  f"clipfrac {stats['clipfrac']/n:.3f} | resp_len {resp_len:.0f}")
            if logger:
                logger.log(it, {"reward": mean_reward, "kl_ref": kl_ref, "policy_loss": stats["policy_loss"]/n,
                                "value_loss": stats["value_loss"]/n, "clipfrac": stats["clipfrac"]/n, "resp_len": resp_len})

        if ctx.is_main and it > 0 and it % cfg.eval_every == 0:
            if eval_set is None:
                eval_set = load_gsm8k_eval("test", limit=200)
            res = gsm8k_accuracy(unwrap(actor_ddp), eval_set, device=ctx.device, max_new_tokens=cfg.rollout_len)
            print(f"  [eval] iter {it} | GSM8K test acc {res['accuracy']:.3f} ({res['correct']}/{res['n']})")
            if logger:
                logger.log(it, {"gsm8k_acc": res["accuracy"]})
        if ctx.is_main and it > 0 and it % cfg.save_every == 0:
            save_stage_ckpt(cfg.out_ckpt, unwrap(actor_ddp).transformer, optimizer, stage="ppo", cfg=cfg, step=it,
                            metrics={"reward": mean_reward})

    if ctx.is_main:
        save_stage_ckpt(cfg.out_ckpt, unwrap(actor_ddp).transformer, optimizer, stage="ppo", cfg=cfg, step=cfg.iterations, metrics={})
        print(f"Done PPO -> {cfg.out_ckpt}")
        if logger:
            logger.close()
    cleanup(ctx)


if __name__ == "__main__":
    main()

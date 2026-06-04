"""Live-progress GRPO-ascends-reward proof (prints every few iters). See verify_rl_optimizes.py."""
import torch
from src.models.transformer import Transformer
from src.post_training.grpo import group_advantages, grpo_loss
from src.post_training.rollout import generate_with_logprobs, compute_logprobs
from src.post_training.utils import make_frozen_copy, set_seed

VOCAB, TARGET, PL, GEN, G = 64, 7, 4, 8, 8
set_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
policy = Transformer(n_head=4, n_embed=64, context_length=PL + GEN + 2, vocab_size=VOCAB, N_BLOCKS=2).to(dev)
ref = make_frozen_copy(policy, device=dev)
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
prompt = torch.zeros(G, PL, dtype=torch.long, device=dev)
hist = []
for it in range(40):
    rb = generate_with_logprobs(policy, prompt, GEN, temperature=1.0)
    seqs, rmask = rb.sequences, rb.response_mask
    rew = torch.tensor([(seqs[i, PL:] == TARGET).float().mean().item() for i in range(G)], device=dev)
    adv = group_advantages(rew, G)
    old, _ = compute_logprobs(policy, seqs, rmask, requires_grad=False)
    rf, _ = compute_logprobs(ref, seqs, rmask, requires_grad=False)
    new, _ = compute_logprobs(policy, seqs, rmask, requires_grad=True)
    loss, _ = grpo_loss(new, old, rf, adv, rmask[:, 1:], clip=0.2, kl_coef=0.0)
    opt.zero_grad(); loss.backward(); opt.step()
    hist.append(rew.mean().item())
    if it % 5 == 0 or it == 39:
        print(f"iter {it:2d} | mean reward (P[target token]) = {rew.mean().item():.3f}", flush=True)
start, end = sum(hist[:5]) / 5, sum(hist[-5:]) / 5
verdict = "PASS (RL ascends reward)" if end > start + 0.2 else "NO IMPROVEMENT"
print(f"\nRESULT: reward {start:.3f} -> {end:.3f}  => {verdict}", flush=True)

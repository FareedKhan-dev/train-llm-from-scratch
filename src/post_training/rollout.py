"""
Generation + per-token log-probability utilities -- the shared core of every RL
algorithm here (PPO and GRPO).

These are FREE FUNCTIONS rather than methods on ``Transformer`` on purpose: the exact
same log-prob math has to be run against four different parameter sets during PPO/GRPO
(the trainable policy, the frozen reference, the old-policy snapshot, and -- for the
value head -- the actor-critic wrapper), so ``f(model, ...)`` composes far better than a
bound method and keeps the educational model file untouched.

Two key constraints from the repo's model:
- Learned absolute positions cap any sequence at ``context_length``; we enforce
  ``prompt_len + max_new_tokens <= context_length``.
- There is no padding-aware attention mask, so callers pass **equal-length** prompts
  (length-bucketed). Padding only ever appears AFTER a row hits its stop token, and the
  ``response_mask`` makes the loss ignore it.

Numerical note: log-probs are always taken in fp32 (``logits.float()``) even under bf16
autocast, because PPO/GRPO/DPO subtract log-probs and bf16 rounding there is harmful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.post_training.chat_template import EOT_ID


@dataclass
class RolloutBatch:
    """Result of :func:`generate_with_logprobs`.

    Attributes:
        sequences:     (B, P+G) prompt tokens followed by generated tokens.
        response_mask: (B, P+G) bool. True only on generated positions that are real
                       (not post-stop padding); always False over the prompt.
        gen_logprobs:  (B, G) log-prob of each generated token under the sampling policy
                       (full-distribution log-prob at the sampling temperature). 0 on
                       padded/finished positions.
        prompt_len:    int P, the (shared) prompt length.
    """

    sequences: torch.Tensor
    response_mask: torch.Tensor
    gen_logprobs: torch.Tensor
    prompt_len: int


def _logits_from(model, idx: torch.Tensor) -> torch.Tensor:
    """Get logits whether ``model`` returns ``(logits, loss)`` (Transformer) or
    ``(logits, values)`` (value-head wrapper) or a bare logits tensor."""
    out = model(idx)
    return out[0] if isinstance(out, tuple) else out


def filter_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    Apply temperature, then optional top-k and nucleus (top-p) filtering to a batch of
    next-token logits ``(B, vocab)``. Returns logits with filtered entries set to -inf,
    ready to softmax-and-sample. Used ONLY to build the sampling distribution; the
    log-prob we record for the ratio is the full-distribution log-prob (see
    :func:`generate_with_logprobs`).
    """
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumprobs > top_p
        # Keep at least the top token; shift so the token that crosses top_p stays.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))

    return logits


@torch.no_grad()
def generate_with_logprobs(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_tokens: tuple[int, ...] = (EOT_ID,),
    pad_id: int = EOT_ID,
    context_length: int | None = None,
) -> RolloutBatch:
    """
    Autoregressively sample a completion for each prompt and record per-token log-probs.

    ``prompt_ids`` must be (B, P) with a shared length P (length-bucket your prompts).
    Rows stop individually once they emit any ``stop_tokens``; subsequent positions are
    filled with ``pad_id`` and excluded from ``response_mask``.

    No KV cache (kept for clarity): each step re-runs the prefix, which is fine for the
    short sequences here on an H100.
    """
    model.eval()
    device = prompt_ids.device
    B, P = prompt_ids.shape

    cap = context_length if context_length is not None else getattr(model, "context_length", P + max_new_tokens)
    cap = getattr(getattr(model, "transformer", model), "context_length", cap)
    max_new_tokens = min(max_new_tokens, cap - P)
    if max_new_tokens <= 0:
        raise ValueError(f"Prompt length {P} leaves no room under context_length {cap}.")

    stop_set = torch.tensor(stop_tokens, device=device)
    sequences = prompt_ids.clone()
    gen_logprobs = torch.zeros(B, max_new_tokens, device=device)
    gen_mask = torch.zeros(B, max_new_tokens, dtype=torch.bool, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        idx_cond = sequences[:, -cap:]
        logits = _logits_from(model, idx_cond)[:, -1, :]  # (B, vocab)

        # Full-distribution log-prob at the sampling temperature (matches recompute).
        full_logprobs = F.log_softmax(logits.float() / max(temperature, 1e-6), dim=-1)
        # Separate truncated distribution to actually sample from.
        probs = F.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

        active = ~finished
        token_lp = full_logprobs.gather(-1, next_tok[:, None]).squeeze(-1)
        # Finished rows emit padding and contribute nothing.
        next_tok = torch.where(active, next_tok, torch.full_like(next_tok, pad_id))
        gen_logprobs[:, t] = torch.where(active, token_lp, torch.zeros_like(token_lp))
        gen_mask[:, t] = active

        sequences = torch.cat([sequences, next_tok[:, None]], dim=1)
        # A row is finished AFTER (and including) the step where it emits a stop token.
        hit_stop = (next_tok[:, None] == stop_set).any(dim=-1)
        finished = finished | (active & hit_stop)

        if bool(finished.all()):
            # Pad remaining columns so shapes stay (B, max_new_tokens).
            if t + 1 < max_new_tokens:
                rem = max_new_tokens - (t + 1)
                sequences = torch.cat(
                    [sequences, torch.full((B, rem), pad_id, dtype=sequences.dtype, device=device)], dim=1
                )
            break

    response_mask = torch.cat(
        [torch.zeros(B, P, dtype=torch.bool, device=device), gen_mask], dim=1
    )
    return RolloutBatch(
        sequences=sequences,
        response_mask=response_mask,
        gen_logprobs=gen_logprobs,
        prompt_len=P,
    )


def _cap_of(model) -> int:
    inner = getattr(model, "transformer", model)
    return getattr(inner, "context_length", 1024)


@torch.no_grad()
def rollout_prompts(
    model,
    prompts: list[list[int]],
    max_new_tokens: int,
    *,
    device: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    pad_id: int = EOT_ID,
    micro_batch: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate one completion per (possibly variable-length) tokenized prompt and merge the
    results into padded tensors. Length-buckets prompts (the model has no padding-aware
    attention mask) and pads all rollouts to a common width.

    Returns ``(sequences, response_mask, prompt_lens)``:
        sequences:     (N, Tmax) prompt+response, right-padded with ``pad_id``.
        response_mask: (N, Tmax) bool, True on real generated tokens.
        prompt_lens:   (N,) prompt length of each row.
    """
    cap = _cap_of(model)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    pieces = []  # (orig_idxs, sequences, response_mask, prompt_len)
    i = 0
    while i < len(order):
        L = len(prompts[order[i]])
        j = i
        while j < len(order) and len(prompts[order[j]]) == L and (j - i) < micro_batch:
            j += 1
        idxs = order[i:j]
        i = j
        budget = max(1, min(max_new_tokens, cap - L))
        batch = torch.tensor([prompts[k] for k in idxs], dtype=torch.long, device=device)
        rb = generate_with_logprobs(model, batch, budget, temperature=temperature,
                                    top_k=top_k, top_p=top_p, stop_tokens=(EOT_ID,), pad_id=pad_id)
        pieces.append((idxs, rb.sequences, rb.response_mask, L))

    N = len(prompts)
    Tmax = max(seq.size(1) for _, seq, _, _ in pieces)
    sequences = torch.full((N, Tmax), pad_id, dtype=torch.long, device=device)
    response_mask = torch.zeros((N, Tmax), dtype=torch.bool, device=device)
    prompt_lens = torch.zeros(N, dtype=torch.long, device=device)
    for idxs, seq, rmask, L in pieces:
        t = seq.size(1)
        for row, k in enumerate(idxs):
            sequences[k, :t] = seq[row]
            response_mask[k, :t] = rmask[row]
            prompt_lens[k] = L
    return sequences, response_mask, prompt_lens


def compute_logprobs(
    model,
    sequences: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher-forced recomputation of per-token log-probs of ``sequences`` under ``model``.

    Mirrors the model's training shift: ``logits[:, t]`` predicts ``sequences[:, t+1]``,
    so the returned tensors have length ``T-1`` and are aligned to target positions
    ``1..T-1``.

    Args:
        sequences:     (B, T) token ids (prompt + response).
        response_mask: (B, T) bool over response positions (as from :class:`RolloutBatch`).
        temperature:   divide logits by this before log-softmax, matching sampling.
        requires_grad: if False, runs under ``no_grad`` (use for ref / old-policy).

    Returns:
        (logprobs, mask) each (B, T-1). ``logprobs`` is the log-prob of the realized
        next token; ``mask`` is the response mask shifted to align with those targets.
    """
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = _logits_from(model, sequences)[:, :-1, :]  # predict tokens 1..T-1
        logprobs_all = F.log_softmax(logits.float() / max(temperature, 1e-6), dim=-1)
        targets = sequences[:, 1:].unsqueeze(-1)
        logprobs = logprobs_all.gather(-1, targets).squeeze(-1)  # (B, T-1)
    mask = response_mask[:, 1:].to(torch.bool)
    return logprobs, mask


def sequence_logprobs(
    model,
    sequences: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sequence-level summed log-prob over response tokens (used by DPO/KTO/ORPO).

    Returns ``(sum_logprob, n_tokens)`` each shape (B,). The per-token mean is
    ``sum_logprob / n_tokens.clamp(min=1)``.
    """
    lp, mask = compute_logprobs(model, sequences, response_mask, temperature=temperature, requires_grad=requires_grad)
    m = mask.to(lp.dtype)
    return (lp * m).sum(dim=-1), m.sum(dim=-1)


def sequence_entropy(model, sequences: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Per-token predictive entropy ``(B, T-1)`` of ``model`` on ``sequences`` (for an
    optional PPO entropy bonus / diagnostics)."""
    with torch.no_grad():
        logits = _logits_from(model, sequences)[:, :-1, :]
        logp = F.log_softmax(logits.float() / max(temperature, 1e-6), dim=-1)
        return -(logp.exp() * logp).sum(dim=-1)

"""
Speculative decoding (Leviathan et al. 2023; Chen et al. 2023), written from scratch.

Generating with a big model is slow because every token needs a full forward pass, and the
passes cannot overlap: token t+1 depends on token t. Speculative decoding uses a small,
fast *draft* model to guess the next ``k`` tokens, then runs the big *target* model ONCE on
all of them. The target's probabilities decide how many guesses to keep:

    for each draft token x with draft probability q(x) and target probability p(x):
        keep x with probability min(1, p(x) / q(x))
    at the first rejection, sample a replacement from  max(0, p - q)  (renormalized)
    if all k are kept, sample one bonus token from the target's next distribution

This rule makes the output distribution *exactly* the target model's distribution, no
matter how bad the draft is. A good draft just means more tokens kept per target pass. With
greedy decoding the rule reduces to "keep a guess if it equals the target's argmax", so the
result is identical to plain greedy decoding with the target.

This version recomputes the prefix on every call (no KV cache) to keep the algorithm in
view; :attr:`SpeculativeStats.target_calls` counts the expensive passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from src.inference.sampling import filter_logits
from src.models.factory import LanguageModel


@dataclass
class SpeculativeStats:
    proposed: int = 0  # draft tokens proposed
    accepted: int = 0  # draft tokens the target kept
    target_calls: int = 0  # forward passes of the big model
    generated: int = 0  # tokens produced

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(1, self.proposed)

    @property
    def tokens_per_target_call(self) -> float:
        return self.generated / max(1, self.target_calls)


def _probs(logits: Float[Tensor, "n vocab"], temperature: float, top_k: int | None, greedy: bool) -> Float[Tensor, "n vocab"]:
    if greedy:
        return F.one_hot(logits.argmax(dim=-1), logits.size(-1)).float()
    return F.softmax(filter_logits(logits.float(), temperature, top_k), dim=-1)


def _logits(model: LanguageModel, idx: Int[Tensor, "1 seq"], window: int) -> Float[Tensor, "1 visible vocab"]:
    out = model(idx[:, -window:])
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def speculative_generate(
    target: LanguageModel,
    draft: LanguageModel,
    idx: Int[Tensor, "1 seq"],
    max_new_tokens: int,
    *,
    k: int = 4,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
    target_window: int | None = None,
    draft_window: int | None = None,
) -> tuple[Int[Tensor, "1 total"], SpeculativeStats]:
    """Generate ``max_new_tokens`` tokens after ``idx`` (batch size 1) with draft-then-verify.

    ``target_window`` and ``draft_window`` cap how many recent tokens each model sees, like
    ``context_window`` in ``model.generate`` (by default each model's full context). Once the
    text is longer than the target's window, its single verify pass gives the first rows up to
    ``k`` tokens less history than one-token-at-a-time decoding would, so the match with plain
    decoding is exact only while the text fits in the window.
    """
    if idx.size(0) != 1:
        raise ValueError("speculative_generate handles one sequence at a time")
    t_window = min(target_window or target.context_length, target.context_length)
    d_window = min(draft_window or draft.context_length, draft.context_length)
    if t_window <= k:
        raise ValueError(f"the target window ({t_window}) must be longer than k ({k})")
    stats = SpeculativeStats()
    start = idx.size(1)
    while idx.size(1) - start < max_new_tokens:
        n = min(k, max_new_tokens - (idx.size(1) - start))
        # 1. The draft proposes n tokens, one at a time, remembering its probabilities q.
        x, guesses, q_rows = idx, [], []
        for _ in range(n):
            q = _probs(_logits(draft, x, d_window)[:, -1, :], temperature, top_k, greedy)
            tok = torch.multinomial(q, 1, generator=generator)
            guesses.append(tok)
            q_rows.append(q[0])
            x = torch.cat([x, tok], dim=1)
        stats.proposed += n

        # 2. One target pass scores all n guesses, plus the position after them.
        p_rows = _probs(_logits(target, x, t_window)[0, -(n + 1) :, :], temperature, top_k, greedy)
        stats.target_calls += 1

        # 3. Keep each guess with probability min(1, p / q); stop at the first rejection.
        kept: list[Tensor] = []
        next_tok = None
        for i, tok in enumerate(guesses):
            t = int(tok)
            ratio = p_rows[i, t] / q_rows[i][t].clamp(min=1e-12)
            if torch.rand((), generator=generator) < ratio.clamp(max=1.0):
                kept.append(tok)
                continue
            residual = (p_rows[i] - q_rows[i]).clamp(min=0)
            residual = residual / residual.sum() if residual.sum() > 0 else p_rows[i]
            next_tok = torch.multinomial(residual[None], 1, generator=generator)
            break
        if next_tok is None:  # every guess was kept: the target's last row gives a bonus token
            next_tok = torch.multinomial(p_rows[n][None], 1, generator=generator)
        stats.accepted += len(kept)
        new = torch.cat([*kept, next_tok], dim=1) if kept else next_tok
        idx = torch.cat([idx, new], dim=1)
    idx = idx[:, : start + max_new_tokens]
    stats.generated = idx.size(1) - start
    return idx, stats

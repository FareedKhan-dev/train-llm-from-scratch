"""
Fast smoke tests for the post-training core (no training, no datasets, runs on CPU in
seconds). Verifies the load-bearing math so bugs surface before the long runs.

Run from the repo root:
    PYTHONPATH=. python tests/test_post_training_smoke.py
"""

import torch
import torch.nn.functional as F

from src.models.transformer import Transformer
from src.post_training import chat_template as ct
from src.post_training.rollout import (
    generate_with_logprobs, compute_logprobs, filter_logits,
)
from src.post_training.value_head import TransformerWithValueHead
from src.post_training.reward_model import RewardModel
from src.post_training.utils import make_frozen_copy, gather_last, masked_mean, build_model_from_config
from src.post_training.rewards import extract_answer, gsm8k_gold_answer, reward_gsm8k, is_correct


def _tiny_model(vocab=64, ctx=32):
    torch.manual_seed(0)
    return Transformer(n_head=4, n_embed=32, context_length=ctx, vocab_size=vocab, N_BLOCKS=2)


def test_forward_hidden_matches_forward():
    m = _tiny_model().eval()
    idx = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        logits_a, _ = m(idx)
        logits_b = m.lm_head(m.forward_hidden(idx))
    assert torch.allclose(logits_a, logits_b, atol=1e-5), "forward_hidden must reproduce forward logits"
    print("ok  forward_hidden matches forward")


def test_compute_logprobs_matches_manual():
    m = _tiny_model().eval()
    seq = torch.randint(0, 64, (3, 12))
    mask = torch.ones_like(seq, dtype=torch.bool)
    lp, shifted = compute_logprobs(m, seq, mask, temperature=1.0, requires_grad=False)
    with torch.no_grad():
        logits = m(seq)[0][:, :-1, :]
        manual = F.log_softmax(logits.float(), dim=-1).gather(-1, seq[:, 1:, None]).squeeze(-1)
    assert torch.allclose(lp, manual, atol=1e-5), "compute_logprobs disagrees with manual log_softmax/gather"
    assert shifted.shape == (3, 11)
    print("ok  compute_logprobs matches manual computation")


def test_rollout_logprobs_consistent():
    # Recorded sampling log-probs must equal a teacher-forced recompute under the same model+temp.
    m = _tiny_model().eval()
    prompt = torch.randint(0, 64, (4, 5))
    rb = generate_with_logprobs(m, prompt, max_new_tokens=10, temperature=1.0, top_k=None, top_p=None)
    assert rb.sequences.shape[1] == 5 + 10
    assert rb.response_mask[:, :5].sum() == 0, "prompt positions must not be in response_mask"
    lp, shifted = compute_logprobs(m, rb.sequences, rb.response_mask, temperature=1.0, requires_grad=False)
    # Align recorded gen_logprobs (B,G) to shifted recompute on generated positions.
    recorded = rb.gen_logprobs[rb.response_mask[:, 5:]]
    recomputed = lp[shifted]
    assert torch.allclose(recorded, recomputed, atol=1e-4), "sampling vs recompute log-probs diverge"
    print(f"ok  rollout/recompute log-probs consistent ({recorded.numel()} response tokens)")


def test_context_cap_enforced():
    m = _tiny_model(ctx=16).eval()
    prompt = torch.randint(0, 64, (1, 10))
    rb = generate_with_logprobs(m, prompt, max_new_tokens=100)  # asks for too many
    assert rb.sequences.shape[1] <= 16, "must clamp prompt+gen to context_length"
    print("ok  context_length cap enforced")


def test_value_and_reward_heads():
    m = _tiny_model()
    actor = TransformerWithValueHead(m)
    idx = torch.randint(0, 64, (2, 8))
    logits, values = actor(idx)
    assert logits.shape == (2, 8, 64) and values.shape == (2, 8)
    values.sum().backward()  # critic path differentiable
    assert actor.value_head[0].weight.grad is not None

    rm = RewardModel(_tiny_model())
    lengths = torch.tensor([8, 5])
    r = rm(idx, seq_lengths=lengths)
    assert r.shape == (2,)
    r.sum().backward()
    assert rm.reward_head.weight.grad is not None
    print("ok  value head + reward head shapes and backprop")


def test_frozen_copy_and_reductions():
    m = _tiny_model()
    ref = make_frozen_copy(m, device="cpu")
    assert all(not p.requires_grad for p in ref.parameters())
    vals = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1, 1, 0]])
    assert abs(masked_mean(vals, mask).item() - 1.5) < 1e-6
    last = gather_last(torch.tensor([[10.0, 20.0, 30.0]]), torch.tensor([2]))
    assert last.item() == 20.0
    print("ok  frozen copy + masked_mean + gather_last")


def test_chat_template_masking():
    msgs = [{"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<answer>4</answer>"}]
    ids, mask = ct.encode_chat(msgs)
    assert len(ids) == len(mask)
    assert ids[-1] == ct.EOT_ID and mask[-1] == 1, "assistant turn must end in a trained EOT"
    # The assistant content tokens should be the only masked-in region (plus its EOT).
    assert sum(mask) > 0 and sum(mask) < len(mask)
    prompt_ids = ct.encode_prompt(msgs[:1])
    assert ct.decode(prompt_ids).endswith(ct.ASSISTANT_HEADER)
    print(f"ok  chat template ids/mask aligned ({sum(mask)}/{len(mask)} trained tokens)")


def test_reward_parsing():
    assert gsm8k_gold_answer("She has 18 apples.\n#### 18") == 18.0
    assert extract_answer("blah <answer>42</answer> done") == 42.0
    assert extract_answer("the total is 1,234 dollars") == 1234.0
    assert is_correct("<answer>42</answer>", 42.0)
    assert reward_gsm8k("<answer>42</answer>", 42.0) > 1.0           # correct + format
    assert reward_gsm8k("<answer>7</answer>", 42.0) == 0.2           # wrong but formatted
    assert reward_gsm8k("the answer is 42", 42.0) == 1.0            # correct, no format bonus
    print("ok  reward parsing + verifier scoring")


def test_build_from_config():
    from config.post_training_config import smoke, SFTConfig
    cfg = smoke(SFTConfig)
    m = build_model_from_config(cfg)
    assert m.context_length == 64 and m.lm_head.out_features == 256
    print("ok  build_model_from_config from dataclass")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_forward_hidden_matches_forward()
    test_compute_logprobs_matches_manual()
    test_rollout_logprobs_consistent()
    test_context_cap_enforced()
    test_value_and_reward_heads()
    test_frozen_copy_and_reductions()
    test_chat_template_masking()
    test_reward_parsing()
    test_build_from_config()
    print("\nALL SMOKE TESTS PASSED")

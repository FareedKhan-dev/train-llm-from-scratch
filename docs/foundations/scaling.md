# Scaling: Parameters, FLOPs, Memory and Time

Before training anything big, four questions can be answered with arithmetic: how many
parameters, how much compute, how much memory, and how long. `scripts/model_report.py` does it
for any preset or config. It builds the model on PyTorch's `meta` device (shapes, no memory), so
even the 3B config is instant on a laptop:

```bash
python scripts/model_report.py --preset 77m
python scripts/model_report.py --preset student --arch modern --tokens 50e6
python scripts/model_report.py --config configs/base.json --tflops 989 --mfu 0.4 --gpus 8
```

```text
Model: 77m (classic) | vocab 50304, n_embed 512, 8 blocks, 8 heads, context 512

Parameters
  total                          77,031,552  (77.03M)
  embedding tables               26,017,792  (34% of the total)
  per block                       3,150,848

Compute
  training FLOPs per token  331.25MFLOP (at 512-token windows)
  Chinchilla-optimal data   1.54G tokens (20 per parameter)
  training on 1.54G tokens = 510.33PFLOP
  at 100 TFLOP/s x 1 device(s) and 40% utilization: 3.5 hours (0.1 days)

Memory
  weights (bf16, inference) 0.14 GiB
  AdamW training state      1.15 GiB (fp32 weights + grads + 2 moments, 16 B/param)
  activations, batch 24     ~3.47 GiB (bf16; gradient checkpointing cuts most)
  KV cache per token        16.0 KiB (bf16), 0.01 GiB for one full 512-token context
```

The rest of this page explains where each line comes from.

## Counting parameters

With `d = n_embed`, a classic block has four `d x d` matrices in attention (query, key, value,
output) and two in the MLP (`d x 4d` and `4d x d`):

$$
\text{per block} \approx 4d^2 + 8d^2 = 12\,d^2
$$

Biases and norm scales add a few thousand more. For the 3B config (`d = 2048`), `12 d^2` is
50,331,648 and the exact count is 50,352,128. Outside the blocks, the token embedding and the
output layer each hold `vocab_size x d` numbers, which is why they dominate small models (34% of
the 77M model, 71% of the tiny one) and almost vanish in big ones (3% of the 3B model).

## Compute: about 6 FLOPs per parameter per token

A matrix multiply costs 2 FLOPs (a multiply and an add) per weight per token. The backward pass
costs about twice the forward pass: one product for the gradient of the input and one for the
gradient of the weight. So training costs

$$
\text{FLOPs per token} \approx 6N + 12\, L\, d\, T
$$

where `N` counts the weights a token is multiplied by (embeddings are lookups, not
multiplies; with Mixture of Experts only the active experts count), and the second term is
attention itself: `L` layers, each comparing the token with `T` others
([PaLM, appendix B](https://arxiv.org/abs/2204.02311)). Since `N` is about `12 L d^2`, attention
matches the weights when `T` reaches `6d`, about 12,000 tokens for `d = 2048`. Below that, the
`6N` term is the whole story, which is why "6 times parameters times tokens" is the standard
back-of-the-envelope estimate.

## Data: about 20 tokens per parameter

[Chinchilla](https://arxiv.org/abs/2203.15556) trained hundreds of models and found that, for a
fixed compute budget, loss is lowest when parameters and training tokens grow together, at
roughly 20 tokens per parameter. The report uses that as its default `--tokens`.

It is a statement about the cheapest way to *train* a given loss, not a rule. A model that will
serve many users is cheaper overall if it is smaller and trained much longer: Llama 3 8B saw 15
trillion tokens, about 1,900 per parameter.

## Time: FLOPs over achieved throughput

$$
\text{seconds} = \frac{\text{FLOPs per token} \times \text{tokens}}{\text{peak FLOP/s} \times \text{MFU} \times \text{devices}}
$$

*MFU* (model FLOPs utilization) is the fraction of the hardware's peak that training actually
achieves. 30% to 50% is typical for large, well-tuned runs. Small models do much worse because
each operation is too small to keep the GPU busy. The 77M run in the README is a real example:
about 140,000 tokens per second times 331 MFLOP per token is 46 TFLOP/s, on two L40 GPUs with
181 TFLOP/s of bf16 each, an MFU of about 13%.

A few estimates side by side (Chinchilla-optimal data, 40% MFU):

| Model | Training compute | One H100 (989 TFLOP/s) | Eight H100s |
|---|---:|---:|---:|
| 77M | 510 PFLOP | 22 minutes | 3 minutes |
| 355M modern (`configs/base.json`) | 17.3 EFLOP | 12 hours | 1.5 hours |
| 3B (the original `config.py`) | 1,370 EFLOP | 40 days | 5 days |

And on a laptop CPU: the tiny preset trained on 6.1M tokens (14.8 TFLOP) in 5 minutes 17
seconds, about 47 GFLOP/s. The student preset, with wider matrices, ran its first 400 steps
(58 TFLOP) in 6 minutes 50 seconds, about 140 GFLOP/s: bigger models keep the CPU busier. Even
at that speed the 3B model would take around 300 years, which is the honest reason the student
track exists.

## Memory

- **Inference**: 2 bytes per parameter in bf16, plus the KV cache. A 3B model needs 6.4 GiB.
- **Training with AdamW**: about 16 bytes per parameter: fp32 weights (4), gradients (4) and
  Adam's two moments (4 + 4). The 3B model needs 51 GiB *before* activations, more than any
  single consumer GPU. Sharding that state across GPUs (FSDP, ZeRO) is how big runs fit.
- **Activations**: what the backward pass needs from the forward pass. Per layer, about
  `T * batch * d * (34 + 5 * n_head * T / d)` bytes in bf16
  ([Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198)). It grows with batch and
  window, not with the parameter count, and gradient checkpointing (`--grad-checkpointing`)
  trades it for about one extra forward pass.
- **KV cache**: what generation stores per token; see [attention](../modern/attention.md#the-kv-cache-and-why-it-dominates-memory).

## Use it before you train

```bash
# will the 3B model fit on my GPU, and how long would it take?
python scripts/model_report.py --preset 3b --tflops 989 --mfu 0.4 --gpus 8

# how long will the student preset take on my laptop?
python scripts/model_report.py --preset student --arch modern --tokens 32.8e6 --tflops 0.14 --mfu 1
```

The second command uses the student preset's measured rate from above (0.14 TFLOP/s) as the
"peak", with an MFU of 1, and prints 1.2 hours. For your own machine, time a short run of the
preset you plan to use (`--steps 200`), divide its training FLOPs by the seconds it took, and
pass that number. Use the same preset for the measurement: on our laptop the student preset
reached three times the tiny preset's FLOP/s.

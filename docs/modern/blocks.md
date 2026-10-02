# Building Blocks: RMSNorm, SwiGLU, Tied Embeddings and Initialization

These four changes are small in code and big in effect. They are the first things to read in
`src/models/modern/`, because every other part builds on them.

## RMSNorm

LayerNorm (used by the classic model) centers each token vector and scales it to unit
variance, then applies a learned scale and bias:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

[RMSNorm](https://arxiv.org/abs/1910.07467) drops the centering and the bias, and divides by
the root mean square instead:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

It is cheaper (one statistic instead of two) and works just as well, which is why every
recent open model uses it. The code is four lines:

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        x32 = x.float()                                   # squares of bf16 numbers lose precision
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight
```

The statistics are computed in float32 even when the model runs in bfloat16: squaring small
bf16 numbers loses most of their precision, and a slightly wrong norm in every layer adds up.
`tests/test_modern_model.py` checks it against `torch.nn.RMSNorm`.

## SwiGLU

The classic MLP expands each token to 4x its width, applies a ReLU, and projects back:
`proj(relu(hidden(x)))`. A *gated* linear unit computes two projections and lets one of them
gate the other, feature by feature ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)):

$$
\text{SwiGLU}(x) = W_{\text{down}}\big(\text{SiLU}(W_{\text{gate}}\,x) \odot W_{\text{up}}\,x\big),
\qquad \text{SiLU}(z) = z \cdot \sigma(z)
$$

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

There are three matrices instead of two, so the hidden width drops from `4 * n_embed` to about
`8/3 * n_embed` (rounded up to a multiple of 64) to keep the parameter count the same. At the
same size, SwiGLU reaches a lower loss than ReLU or GELU, and nobody fully knows why; the
gate gives each feature a smooth, input-dependent on/off switch, which seems to make the
layer easier to optimize.

## Tied embeddings

The input embedding maps a token id to a vector (`vocab_size x n_embed`), and the output
layer maps a vector back to a score for every token (`n_embed x vocab_size`). Both relate
tokens and vectors, so many models use *one* matrix for both:

```python
self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
if config.tie_embeddings:
    self.lm_head.weight = self.token_embed.weight      # the same Parameter object
```

For small models this is a big saving. With the r50k vocabulary (50304 tokens) and
`n_embed = 128`, each matrix holds 6.4M numbers, which is half of the classic 13M model.
Tying is standard for small models (Gemma, small Llama and Qwen) and optional for big ones,
where the embeddings are a small fraction of the total.

## Initialization

PyTorch's default `nn.Linear` initialization scales with `1 / sqrt(fan_in)`, and
`nn.Embedding` starts at standard deviation 1. The modern model follows GPT-2 instead:
every weight starts from a normal distribution with standard deviation `init_std = 0.02`,
and the two projections that write into the residual stream (`o_proj` in attention, `down`
in the MLP) start smaller still:

$$
\sigma_{\text{residual}} = \frac{0.02}{\sqrt{2 \cdot n_{\text{blocks}}}}
$$

Each block *adds* its output to the residual stream, twice. With `L` blocks, `2L` random
contributions add up, and their total variance grows with `2L`. Scaling each one down by
`sqrt(2L)` keeps the stream at the same size no matter how deep the model is, so a 100-layer
model starts as stable as a 2-layer one.

Two consequences you can check yourself:

- A freshly initialized model predicts nearly uniform probabilities, so the first loss is
  close to `ln(vocab_size)` (8.32 for a 4096-token vocabulary). The test
  `test_forward_shapes_and_loss` checks this for every variant.
- No biases anywhere: they add parameters, and normalization layers make them redundant.

## Where to go next

- [Rotary position embeddings](rope.md), the biggest single change.
- [Attention](attention.md), where most of the memory goes.

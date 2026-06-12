# Objectives, Losses & Perplexity

The architecture defines what the model can compute. The objective defines what behavior training
rewards.

For a decoder-only language model, the base objective is next-token prediction.

## From logits to probability

At position \(t\), the model emits logits:

\[
z_t \in \mathbb{R}^{V}
\]

Softmax turns logits into a distribution:

\[
p_\theta(x_{t+1}=i \mid x_{\leq t})
= \frac{\exp(z_{t,i})}{\sum_{j=1}^{V}\exp(z_{t,j})}
\]

The target is one integer token id \(y_t\). Cross-entropy for that position is:

\[
\ell_t = -\log p_\theta(y_t \mid x_{\leq t})
\]

The batch loss is the mean over positions:

\[
\mathcal{L}_{\text{LM}}
= -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
\log p_\theta(y_{b,t} \mid x_{b,\leq t})
\]

## The shift

The model receives:

\[
x = [t_0,t_1,\ldots,t_{T-1}]
\]

and predicts:

\[
y = [t_1,t_2,\ldots,t_T]
\]

In `src/models/transformer.py`, the simple `forward` path computes cross-entropy over all positions:

```python
logits, loss = model(idx, targets)
flat_logits = logits.view(B * T, C)
targets = targets.view(B * T).long()
loss = F.cross_entropy(flat_logits, targets)
```

The post-training SFT path makes the shift explicit because it needs a mask:

```python
logits = logits[:, :-1, :]
targets = tokens[:, 1:]
mask = loss_mask[:, 1:].to(logits.dtype)
```

## Perplexity

Perplexity is exponentiated cross-entropy:

\[
\text{PPL} = \exp(\mathcal{L})
\]

If the model assigns high probability to the true next tokens, loss falls and perplexity falls.

Interpretation:

- loss near \(\log(V)\) means the model is close to uniform random guessing;
- lower loss means the model is concentrating probability on plausible next tokens;
- validation loss matters more than training loss for generalization.

With `V = 50304`:

\[
\log(V) \approx 10.83
\]

So an untrained model often starts near that value.

## SFT masked loss

SFT still uses next-token cross-entropy, but only assistant completion tokens count:

\[
\mathcal{L}_{\text{SFT}}
=
\frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}
{\sum_{b,t} m_{b,t}}
\]

where \(m_{b,t}=1\) for assistant tokens and \(0\) for prompt tokens.

Implementation in `src/post_training/sft.py`:

```python
ce = F.cross_entropy(
    logits.reshape(-1, V).float(),
    targets.reshape(-1).long(),
    reduction="none",
)
ce = ce.view(targets.shape) * mask
return ce.sum() / mask.sum().clamp(min=1.0)
```

This distinction is crucial. The model should learn how to answer the prompt, not how to predict the
prompt itself.

## Sequence log-probabilities

Preference optimization and RL need the log-probability of an entire response, not only one token.

For response tokens \(a_1,\ldots,a_L\) after prompt \(p\):

\[
\log \pi_\theta(a \mid p)
= \sum_{t=1}^{L}
\log \pi_\theta(a_t \mid p, a_{<t})
\]

The repo computes this in `src/post_training/rollout.py` with `sequence_logprobs`. It applies a
response mask and sums token log-probs over answer positions.

That one primitive is reused by:

- DPO, ORPO, and KTO;
- PPO policy ratios;
- GRPO policy ratios;
- KL measurements against a frozen reference model.

## DPO objective

DPO uses preference pairs: chosen response \(y_w\) and rejected response \(y_l\). It compares the policy
against a frozen reference model:

\[
\Delta_\pi =
\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)
\]

\[
\Delta_{\text{ref}} =
\log \pi_{\text{ref}}(y_w \mid x) - \log \pi_{\text{ref}}(y_l \mid x)
\]

\[
\mathcal{L}_{\text{DPO}}
=
-\log \sigma\left(\beta(\Delta_\pi - \Delta_{\text{ref}})\right)
\]

In `src/post_training/dpo.py`:

```python
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = ref_chosen_logps - ref_rejected_logps
logits = pi_logratios - ref_logratios
loss = -F.logsigmoid(beta * logits).mean()
```

Intuition: make the chosen response more likely than the rejected response, but measure the change
relative to the reference model so the policy does not drift without constraint.

## PPO objective in one picture

PPO samples responses, scores them, and updates the policy using the ratio between new and old action
probabilities:

\[
r_t(\theta) =
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\text{old}}(a_t \mid s_t)}
= \exp(\log \pi_\theta - \log \pi_{\text{old}})
\]

The clipped policy objective is:

\[
\mathcal{L}_{\text{PPO}}
=
-\mathbb{E}_t
\left[
\min
\left(
r_t(\theta) A_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon) A_t
\right)
\right]
\]

The repo implements that in `src/post_training/ppo.py`:

```python
ratio = torch.exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
loss = -masked_mean(torch.min(surr1, surr2), mask)
```

The clip prevents one update from moving too far from the sampled policy.

## GRPO objective in one picture

GRPO avoids a learned value function. For each prompt, it samples a group of responses and normalizes
their rewards within the group:

\[
A_i = \frac{r_i - \text{mean}(r_1,\ldots,r_G)}
{\text{std}(r_1,\ldots,r_G)+\epsilon}
\]

That advantage says: "Was this answer better or worse than its siblings for the same prompt?"

In `src/post_training/grpo.py`:

```python
r = rewards.view(-1, group_size)
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)
```

This is useful for verifiable-reward reasoning tasks because it removes PPO's value head and critic
training loop.

## Objective comparison

| Stage | Data | Main signal | Learns |
|---|---|---|---|
| Pretraining | raw token stream | next-token CE | language modeling |
| SFT | prompt/answer examples | masked next-token CE | instruction following format |
| Reward model | chosen/rejected pairs | Bradley-Terry preference loss | scalar preference scoring |
| DPO | chosen/rejected pairs | sequence log-prob preference loss | preference alignment without RL rollout |
| PPO | sampled responses + reward | clipped policy gradient | reward-seeking behavior under KL control |
| GRPO | grouped sampled responses + verifier | group-relative clipped policy gradient | verifier-driven reasoning without critic |

## Next

Loss gives direction. Optimization decides whether the model can follow it reliably. Continue to
[Optimization & Training Systems](optimization.md).

# Preference Optimization: DPO, IPO, SimPO, ORPO and KTO

All of these learn from the same data: a prompt, a *chosen* answer and a *rejected* answer.
They differ in what they compare and what they need in memory. All five run through one
trainer:

```bash
python scripts/train_dpo.py --loss_type dpo   --beta 0.1
python scripts/train_dpo.py --loss_type dpo   --beta 0.1 --label_smoothing 0.1   # conservative DPO
python scripts/train_dpo.py --loss_type ipo   --beta 0.1
python scripts/train_dpo.py --loss_type simpo --beta 2.0 --simpo_gamma 0.5
python scripts/train_dpo.py --loss_type orpo  --orpo_lambda 1.0
python scripts/train_dpo.py --loss_type kto   --beta 0.1
```

## Why DPO needs no reward model

Classic RLHF trains a reward model `r(x, y)`, then maximizes it with PPO while a KL penalty
keeps the policy near the SFT model `pi_ref`. That objective has a closed-form optimum:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\big(r(x, y)/\beta\big)
\;\;\Longrightarrow\;\;
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Plug this "implicit reward" into the Bradley-Terry preference model; the `Z(x)` terms cancel
because both answers share the prompt, and you get a loss on the policy alone
([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)):

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\Big(\beta \big[\underbrace{\log\tfrac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)}}_{\text{chosen}} - \underbrace{\log\tfrac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}}_{\text{rejected}}\big]\Big)
$$

No reward model, no sampling, no RL loop: just two forward passes per pair through the
policy and a frozen copy of the SFT model. The `beta * log(pi / pi_ref)` terms are logged as
`r_chosen` and `r_rejected`, and "implicit accuracy" is how often the chosen one is higher.

## The variants

| Loss | Reference model | Uses | What it fixes | Typical settings |
|---|---|---|---|---|
| DPO | yes | summed log-probs | the reward model and RL loop | `beta` 0.1 |
| cDPO | yes | summed log-probs | noisy labels: assumes each label is flipped with probability `eps` | `label_smoothing` 0.1 |
| [IPO](https://arxiv.org/abs/2310.12036) | yes | mean log-probs | DPO keeps pushing a pair apart forever; IPO aims at a fixed gap `1/(2 beta)` | `beta` 0.1 |
| [SimPO](https://arxiv.org/abs/2405.14734) | **no** | mean log-probs | memory (no reference copy) and the bias toward long answers | `beta` 2.0, `simpo_gamma` 0.5 |
| [ORPO](https://arxiv.org/abs/2403.07691) | **no** | mean log-probs | merges SFT and alignment into one stage (NLL + odds ratio) | `orpo_lambda` 1.0 |
| [KTO](https://arxiv.org/abs/2402.01306) | yes | summed log-probs | works from good/bad labels on single answers, no pairs needed | `beta` 0.1 |

The formulas, as implemented in `src/post_training/dpo.py`:

$$
\begin{aligned}
\mathcal{L}_{\text{cDPO}} &= -(1-\varepsilon)\log\sigma(\beta h) - \varepsilon \log\sigma(-\beta h) \\
\mathcal{L}_{\text{IPO}} &= \Big(\bar h - \tfrac{1}{2\beta}\Big)^2 \\
\mathcal{L}_{\text{SimPO}} &= -\log \sigma\Big(\tfrac{\beta}{|y_w|}\log\pi(y_w|x) - \tfrac{\beta}{|y_l|}\log\pi(y_l|x) - \gamma\Big)
\end{aligned}
$$

where `h` is the DPO log-ratio gap and `h-bar` the same gap computed with per-token averages.

Two of these deserve a closer look:

- **The length bias.** A summed log-probability gets more negative with every token, so the
  DPO reward of a long answer moves faster than a short one's, and DPO-trained models tend
  to get longer. SimPO divides by the length (and `test_simpo_is_reference_free_and_length_normalized`
  checks that doubling both the length and the total log-prob changes nothing).
- **Overfitting easy pairs.** The DPO loss is never zero; it keeps rewarding a wider gap even
  for pairs it already gets right. IPO's squared loss is zero at its target gap and pushes
  back past it (`test_ipo_is_zero_at_its_target_gap`).

## Truncation matters

Preference pairs often have long prompts. If each side is cut to `max_len` on its own, a long
prompt can push both answers out of the window: the two sequences become identical and the
pair teaches nothing. The data loader keeps one shared, left-truncated prompt for both
answers, gives the prompt at most half of the window when the answers are long too, and
keeps the first token where the two answers differ
(`data_loader/preference_dataset.py`, from #41).

## Which one to use

Start with DPO at `beta = 0.1`. If memory is tight, SimPO drops the reference copy. If your
labels are noisy (crowd-sourced, or judged by a small model), add `label_smoothing`. If you
only have thumbs-up / thumbs-down on single answers, use KTO. If you want to skip SFT, ORPO
does both at once.

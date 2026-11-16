# Changelog

## 1.0.0 (2026-11)

The first stable release. A base model can now be benchmarked the way published models are,
with scores that match EleutherAI's lm-evaluation-harness, and a review of the training code
fixed ten bugs that gave quietly wrong results: resumed runs retrained on data they had
already seen, SFT's gradient accumulation weighted tokens unevenly, and the GSM8K checker that
rewards the RL stages marked some wrong answers correct. The repo also has a Docker image, and
the test suite passes with a GPU visible.

### Compatibility notes

These change results compared with 0.2.0, which is why this is 1.0.0:

- **Mixture of Experts with `moe_top_k=1`** now weighs the chosen expert by its router
  probability instead of exactly 1, as the Switch Transformer does, so the router learns from
  the language-modeling loss. A checkpoint trained with one expert per token on 0.2.0 gives
  different outputs; two or more experts per token are unchanged.
- **The dev loss of MoE models** no longer includes the balancing term (it is added in
  training only), so it reads about `moe_aux_loss_coef` lower than before and now compares
  with dense models.
- **SFT with `grad_accum` above 1** trains on the true full-batch gradient, so its loss curves
  differ from 0.2.0's.
- **GSM8K scores** can move slightly, because the answer checker now reads `.5`, `10-15` and
  `5 + 3 = 8` correctly.
- **The data order**: the loader shuffles with its own seed, so a run seeded on 0.2.0 sees its
  windows in a different order on 1.0.0, and the DDP ranks of `pretrain_base.py` now split
  each batch instead of drawing independently. The final checkpoint of `pretrain_base.py`
  records the last completed step (`train_steps - 1`).
- `requirements.txt` no longer installs `torchvision` and `torchaudio`, which nothing used.

### Added

- **Base-model benchmarks**: `scripts/eval_base.py` and `src/benchmarks/` score ARC-Easy,
  ARC-Challenge and LAMBADA on any checkpoint, with lm-evaluation-harness's prompts, rules and
  metrics. On GPT-2 124M it reproduces the harness exactly: the same acc, acc_norm and
  LAMBADA accuracy on all three tasks, with each of the 19,341 answer log-likelihoods within 0.0006
  of the harness's and no prediction different. The datasets are pinned to a revision. HellaSwag is left out on purpose:
  its source repository was taken down by a DMCA notice in September 2026.
- **A Docker image** for training and for running the checks on a GPU (#43 by @LLiuJJ), with a
  `TORCH_INDEX` build argument for hosts whose driver predates CUDA 13.
- `E2E_DEVICE=cuda` runs the end-to-end test of every training stage on a GPU.
- `get_batch_iterator` takes `seed`, `skip_batches`, `rank` and `world_size`.

### Fixed

- A resumed run trained on the batches of steps 0, 1, 2... again instead of continuing the data
  order, in both `train_transformer.py` and `pretrain_base.py`. The data seed is now saved in
  the checkpoint and the resumed run skips what it already trained on.
- The DDP ranks of `pretrain_base.py` shuffled independently, so they trained on overlapping
  windows.
- `pretrain_base.py` saved `step = train_steps` in its final checkpoint, so extending a finished
  run skipped a step, and resuming one with nothing left to train crashed.
- With one expert per token, the MoE router got no gradient from the language-modeling loss.
- MoE dev loss and perplexity included the balancing term.
- SFT's gradient accumulation averaged micro-batch means, so a token in a micro-batch with few
  trained tokens weighed far more than one in a full micro-batch (a cosine of 0.66 with the true
  gradient on a short and a long answer).
- The GSM8K checker read `<answer>.5</answer>` as 5, the hyphen in `10-15` or `COVID-19` as a
  minus sign, and an `<answer>` holding `5 + 3 = 8` as 5.
- PPO's `ent_coef` did nothing; it now adds the entropy bonus.
- `prepare_pretrain_data.py --max_tokens` was ignored below 8M tokens.
- `data_download.py` kept an HTTP error page or an interrupted download, which every later run
  skipped as already downloaded.
- The test suite failed on any machine with a GPU (found in #43 by @LLiuJJ).
- `requirements.txt` and `requirements-post.txt` pointed pip at the CUDA 11.8 and 12.1 wheel
  indexes, which stopped at torch 2.7.1 and 2.5.1.

## 0.2.0 (2026-10)

A big update: the repo now also runs on a laptop CPU, has a modern version of the model next to
the original one, adds the training and inference techniques that became standard since 2023,
and checks itself with typed configs, shape-checked tests and CI.

### Added

- **Laptop track** (no GPU needed): `scripts/prepare_tiny_data.py` downloads TinyStories (or
  Tiny Shakespeare, or your own text) and trains a BPE tokenizer on it; `--preset tiny`,
  `student` and `small` train in minutes to hours on a CPU. A guide in `docs/student/`.
  Asked for in #38 by @RihanMujawar.
- **Modern decoder** in `src/models/modern/`: RMSNorm, rotary embeddings, SwiGLU,
  grouped-query and multi-query attention, DeepSeek's multi-head latent attention, QK-norm,
  gated attention, sliding-window attention, a KV cache, tied embeddings, depth-scaled
  initialization and Mixture of Experts with a balancing loss and shared experts. Every script
  takes `--arch modern` (or `"arch": "modern"` in the JSON configs).
- **Optimizers and schedules**: Muon (Newton-Schulz orthogonalization, AdamW-matched step
  size), and warmup-stable-decay and linear-to-zero schedules next to cosine.
- **Post-training**: IPO, SimPO and conservative DPO (`--loss_type`, `--label_smoothing`);
  Dr. GRPO, DAPO (clip-higher, dynamic sampling, token-level loss) and GSPO options for GRPO;
  LoRA for SFT (`--lora_rank`), merged back into the weights when saved.
- **Inference**: top-p and min-p sampling for both models, speculative decoding
  (`generate_text.py --draft_model`) and int8 weight-only quantization (`--int8`).
- **Tokenizer**: a byte-level BPE tokenizer written from scratch (`src/tokenizer/bpe.py`),
  the same algorithm as GPT-2 and GPT-4, with a test against the textbook algorithm.
- **Hardware**: `--device auto` picks CUDA, Apple MPS or the CPU; `--threads` for CPUs; an
  experimental, untested TPU path through PyTorch/XLA (`--device xla`), for #8.
- **Tools**: `scripts/benchmark.py` measures training speed per preset on your machine, and
  `scripts/model_report.py` estimates parameters, FLOPs, memory and training time.
  `train_transformer.py --set KEY=VALUE` overrides any config value, the modern model's options
  included.
- **Type safety**: typed and validated configs with clear errors (unknown keys, wrong types,
  impossible shapes), jaxtyping shape annotations checked at runtime in the tests, mypy (strict
  for the new modules), and `py.typed` markers.
- **Tests and CI**: about 170 pytest tests, including one that runs every training script end to
  end on generated data with both architectures; GitHub Actions for lint, types, tests on
  Linux, Windows and macOS, and a strict docs build; pre-commit hooks; a contributing guide and
  issue templates.
- **Docs**: a Modern LLM section (ten pages), pages on BPE, scaling, hardware and type safety,
  ten new diagrams, and plots computed from the repo's own code (`docs/diagrams/make_plots.py`).

### Changed

- Data, checkpoints and logs default to `data/`, `models/` and `logs/` inside the repo instead
  of machine-specific paths, and every script runs from the repo without `PYTHONPATH=.`.
- `scripts/generate_text.py` reads the model size, architecture, tokenizer and training window
  from the checkpoint, so only `--model_path` is needed.
- `scripts/train_transformer.py` takes `--preset`, `--arch`, `--device`, `--steps` and more, and
  the presets train on windows as long as the model's context.
- The classic model no longer stores its causal masks in checkpoints (about 1.6 GiB in the
  400M-parameter configuration); older checkpoints still load.
- `scripts/chat.py` picks its device automatically instead of assuming CUDA.
- PPO and GRPO rollouts, evaluation and chat decode with the KV cache when the model is the
  modern one (more than 5x faster rollouts in a CPU test).
- Requires Python 3.10 or newer.

### Fixed

- Checkpoints saved from a `torch.compile` or DDP model carried `_orig_mod.` and `module.`
  prefixes, so the next stage silently started from random weights. Fixed for new and old
  checkpoints, and a missing parameter is now an error (#36; thanks @LLiuJJ for the report and
  the first fix in #37).
- Preference pairs with long prompts could lose both answers to truncation, which made the two
  sequences identical (#41 by @Iams4kura, with a follow-up that keeps a shared prompt budget
  and never raises).
- SFT ignored the `grad_accum` setting.
- `forward_embedding` crashed with more than one block.
- The Streamlit control panel's job status and stop button did not work on Windows and macOS.
- The shell helpers referred to the original author's virtual environment and disk.
- `requirements.txt` was missing dependencies used by the new code.
- With `--filter_groups`, skipped groups shrank GRPO's per-answer loss averages (the paper's
  average, Dr. GRPO and GSPO), like a random learning-rate cut.
- Mixture-of-Experts models crashed multi-GPU training when an expert got no tokens in a step.
- SFT left out the Mixture-of-Experts balancing loss.

Earlier changes are in the git history.

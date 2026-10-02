# Changelog

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

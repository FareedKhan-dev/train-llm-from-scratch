# Contributing

Thanks for helping. This repo is a teaching project first, so the bar for a change is: does
it stay easy to read, and is it correct? Small, focused pull requests are the easiest to
review and merge.

## Setup

With [uv](https://docs.astral.sh/uv/) (fast, recommended):

```bash
git clone https://github.com/FareedKhan-dev/train-llm-from-scratch.git
cd train-llm-from-scratch
uv sync                      # creates .venv with the package + test tools
uv run pytest -q             # everything runs on a CPU in under a minute
```

With pip:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e . pytest beartype ruff mypy
pytest -q
```

On Linux the default PyTorch wheel bundles CUDA (about 2.5 GB). If you only need the CPU
build, install it first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Checks

The same three checks run in CI on Linux, Windows and macOS:

```bash
ruff check .          # lint (pyflakes, pycodestyle, bugbear, pyupgrade, import order)
mypy                  # types, configured in pyproject.toml
pytest -q             # tests, CPU only
```

Some guidelines that keep the code consistent:

- **No GPU needed for tests.** Use tiny models (a few thousand parameters) and generated data.
  Test a property when you can (causality, a cached forward equals a full one, a loss is zero
  at its target) rather than a hard-coded number.
- **Type everything new.** New functions have annotations; tensor arguments use
  [jaxtyping](https://github.com/patrick-kidger/jaxtyping) shapes such as
  `Float[Tensor, "batch seq vocab"]`. `tests/conftest.py` checks these shapes at runtime.
- **Configs are typed dataclasses.** A new hyperparameter is a field in
  `config/post_training_config.py` (use `Literal[...]` for choices) and a key in the matching
  `configs/*.json`. The loader rejects unknown keys and wrong types.
- **Keep the classic model simple.** `src/models/` is what the README teaches line by line.
  New architecture ideas go in `src/models/modern/` or behind a flag.
- **Write like the README.** Plain words, short sentences, explain why before how.

## Pull requests

- One topic per pull request, with a short description of what changed and why.
- Add or update tests for the behavior you changed, and the docs if a command or flag changed.
- Commit messages follow the existing style: `fix(area): what was wrong`,
  `feat(area): what is new`, `docs: ...`.

## Reporting a bug

Please include the command you ran, the full error, and the output of:

```bash
python -c "import torch, sys; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
```

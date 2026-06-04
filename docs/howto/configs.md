# Configure with JSON

Every training stage is configured by a small, human-editable JSON file under
[`configs/`](https://github.com/FareedKhan-dev/train-llm-from-scratch/tree/main/configs). You edit one
file and see only that stage's knobs — the model architecture and runtime live in the shared
`configs/base.json`.

```
configs/
  base.json        # shared model + runtime (vocab, n_embed, device, amp_dtype, ...)
  pretrain.json    # one file per stage — only that stage's hyperparameters
  sft.json  reward.json  dpo.json  ppo.json  grpo.json
  smoke/           # tiny CPU variant (small model + few steps) for quick tests
    base.json  pretrain.json  sft.json  ...
```

## How a config is resolved

[`config/loader.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/config/loader.py)
merges four layers, **lowest precedence first**:

```
dataclass defaults  <  configs/base.json  <  configs/<stage>.json  <  CLI --field overrides
```

The dataclasses in `config/post_training_config.py` are the typed schema; the JSON is the editable
source; the CLI wins for quick one-offs. JSON `null` maps to Python `None` (the clean way to set a
`str | None` field such as `amp_dtype`).

!!! tip "Smoke configs auto-shrink the model"
    A stage JSON in a sub-folder uses that folder's `base.json`. So `configs/smoke/sft.json` picks up
    `configs/smoke/base.json` (tiny model, CPU) automatically — handy for a fast end-to-end test.

## Editing & inspecting

=== "Edit the JSON"

    ```jsonc
    // configs/sft.json
    {
      "pretrained_ckpt": "/ephemeral/ckpts/base_pretrained.pt",
      "data_path": "/ephemeral/data/sft_packed.h5",
      "lr": 1e-5,
      "epochs": 3,
      "batch_size": 16
    }
    ```

=== "Override on the CLI"

    ```bash
    # --field beats the JSON
    python scripts/train_sft.py --config configs/sft.json --lr 2e-5 --batch_size 8
    ```

=== "Print the resolved config"

    ```bash
    python scripts/train_sft.py --config configs/sft.json --print-config
    # dumps the fully merged config as JSON, then exits
    ```

Every trainer accepts `--config <path>` (defaulting to its stage file), all `--field` overrides, and
`--print-config`. The legacy pretraining path (`scripts/train_transformer.py` + `config/config.py`) is
untouched and still works as the README teaches.

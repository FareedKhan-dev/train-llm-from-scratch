# Hardware: CPU, NVIDIA, Apple Silicon and TPU

Every script takes `--device`. The default, `auto`, picks the best backend on the machine:

```text
auto  ->  cuda (NVIDIA GPU)  ->  mps (Apple Silicon)  ->  cpu
```

You can also ask for one explicitly: `--device cpu`, `--device cuda`, `--device cuda:1`,
`--device mps` or `--device xla`. Asking for a backend the machine does not have prints a
warning and falls back to the CPU instead of crashing. The logic is ten lines in
`src/device.py` (`resolve_device`).

## Install the right PyTorch

```bash
uv venv && uv pip install -e .        # or: python -m venv .venv && pip install -e .
```

On Windows and macOS the default PyTorch wheel is the right one (CPU on Windows, CPU plus MPS
on macOS). On Linux the default wheel bundles CUDA and is about 2.5 GB; on a machine without an
NVIDIA GPU, install the CPU build first:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

For a specific CUDA version, use the matching index from [pytorch.org](https://pytorch.org/get-started/locally/).

## Laptop CPU

The student track (`--preset tiny`, `student`, `small`) is sized for a CPU. Measured on a
laptop with an Intel Core Ultra 7 255H (16 cores), TinyStories data, 4096-token BPE:

| Preset | Architecture | Parameters | Steps | Time | Dev loss |
|---|---|---:|---:|---:|---:|
| `tiny` | classic | 636K | 1,500 | 8 min 3 s | 3.30 |
| `tiny` | modern | 369K | 1,500 | 5 min 17 s | 2.78 |

The `student` preset (2.6M parameters, modern) ran its first 400 steps in under 7 minutes, so
its 4,000 steps take about 70 minutes on a quiet laptop. Our run shared the laptop with other
work and reached a training loss of 1.80 after 3,300 steps and two hours.

Two settings matter on a CPU:

- **Threads.** PyTorch uses one thread per physical core by default, which is usually right.
  Hybrid CPUs (performance plus efficiency cores) are sometimes faster with fewer threads,
  because a matrix multiply waits for its slowest thread. Try a few values:

    ```bash
    python scripts/benchmark.py --presets tiny student --threads 6
    python scripts/benchmark.py --presets tiny student --threads 16
    ```

    and pass the best one to training with `--threads N`.

- **Model shape.** Small matrices leave most of the CPU idle, so a bigger model usually costs
  less extra time than its extra FLOPs suggest. `scripts/benchmark.py` prints steps per second
  and tokens per second for each preset and architecture, so you can choose before committing
  to a long run.

There is no mixed precision on the CPU: `--amp` and `amp_dtype` only apply to GPUs.

## NVIDIA GPUs

Everything in the repo was written for this case. The things worth knowing:

- **bf16 autocast** (`--amp` in `train_transformer.py`, `amp_dtype: "bf16"` in the JSON
  configs) roughly halves activation memory and uses the tensor cores. GPUs older than Ampere
  (the T4 and V100, for example) have no bf16: there, `train_transformer.py --amp --amp-dtype fp16`
  uses fp16 with a gradient scaler. The post-training scripts are written for bf16.
- **Memory knobs** for big configs: `--grad-checkpointing` recomputes activations during the
  backward pass, and `--grad-accum N` splits a batch into N smaller ones. See
  [Scaling](../foundations/scaling.md) for how to estimate memory before you start.
- **Several GPUs**: the post-training scripts use DistributedDataParallel. Launch them with
  `torchrun` (or `python -m torch.distributed.run`, which works everywhere, including Windows):

    ```bash
    torchrun --standalone --nproc_per_node=4 scripts/pretrain_base.py --config configs/pretrain.json
    ```

- **`torch.compile`** (`"compile": true` in the configs) speeds up training once the model is
  compiled. Checkpoints are saved without the compile wrapper, so they load anywhere.

The README has a table of which GPU fits which model size.

## Apple Silicon (MPS)

On an M-series Mac, `auto` picks `mps`, and the tiny and student presets train on the GPU.
Two things to know:

- Some operations are not implemented on MPS yet. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` runs
  them on the CPU instead of failing.
- Very small models can be *faster* on the CPU, because every MPS operation has a fixed launch
  cost. Compare both with `scripts/benchmark.py --device mps` and `--device cpu`.

## Google TPUs (XLA, experimental)

Issue #8 asked for TPU support. `--device xla` works when
[PyTorch/XLA](https://github.com/pytorch/xla) is installed (`torch_xla`), for example on a
Cloud TPU VM or a Colab TPU runtime:

```bash
python scripts/train_transformer.py --preset student --device xla
```

XLA traces operations into a graph and only runs it when told to, so the training loop of
`scripts/train_transformer.py` calls `sync_step()` once per step (`torch_xla.sync()`). This
path is **untested on real TPUs**: there was no TPU available while writing it. The other
scripts accept `--device xla` but do not sync per step yet. Reports and fixes are welcome in
#8.

## Checking what you are running on

`scripts/train_transformer.py` prints the PyTorch version, the device and, on a GPU, its name
and memory when it starts. Include those lines when you open an issue. From Python,
`src.device.describe_device(device)` returns the same facts (or the CPU thread count) as a
string.

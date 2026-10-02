"""
Pick the device to train on and report what it is.

``resolve_device("auto")`` returns the best available backend in this order:
CUDA (NVIDIA GPUs) -> MPS (Apple Silicon) -> CPU. You can also ask for one explicitly:
``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"`` or ``"xla"`` (Google TPUs through PyTorch/XLA,
experimental). Asking for a backend that is not available falls back to the CPU with a
warning instead of crashing, which keeps the scripts usable on any laptop.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import warnings
from typing import Any

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps", "xla")


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _xla_available() -> bool:
    return importlib.util.find_spec("torch_xla") is not None


def resolve_device(requested: str | None = "auto") -> str:
    """Turn ``"auto"`` or an explicit request into a device string that works on this machine."""
    req = (requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return req
        warnings.warn("CUDA was requested but is not available, using the CPU instead.", stacklevel=2)
        return "cpu"
    if req == "mps":
        if _mps_available():
            return "mps"
        warnings.warn("MPS was requested but is not available, using the CPU instead.", stacklevel=2)
        return "cpu"
    if req.startswith("xla"):
        if _xla_available():
            import torch_xla  # noqa: F401

            return "xla"
        warnings.warn("XLA was requested but torch_xla is not installed, using the CPU instead.", stacklevel=2)
        return "cpu"
    if req == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device {requested!r}; choose one of {DEVICE_CHOICES}")


def device_type(device: str | torch.device) -> str:
    """``"cuda:1"`` -> ``"cuda"``; used for autocast and for capability checks."""
    return str(device).split(":", 1)[0]


def autocast(device: str, dtype: str | None) -> contextlib.AbstractContextManager[Any]:
    """Mixed-precision context for ``dtype`` in ``{"bf16", "fp16", None}``; a no-op on CPU or when off."""
    kind = device_type(device)
    if dtype is None or kind not in ("cuda", "mps", "xla"):
        return contextlib.nullcontext()
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type=kind, dtype=torch_dtype)


def sync_step(device: str) -> None:
    """Flush pending work on lazy backends. Needed once per optimizer step on XLA (TPU)."""
    if device_type(device) == "xla":
        import torch_xla

        sync = getattr(torch_xla, "sync", None)
        if sync is not None:
            sync()
        else:  # older torch_xla releases
            import torch_xla.core.xla_model as xm

            xm.mark_step()


def configure_cpu_threads(num_threads: int | None = None) -> int:
    """
    Set how many CPU threads PyTorch uses, and return the number in use.

    By default PyTorch's own choice is kept (one thread per physical core), which is usually
    right: ``os.cpu_count()`` counts hyper-threads too, and running one thread per logical
    core oversubscribes the cores and slows matrix multiplies down. On hybrid CPUs
    (performance + efficiency cores) fewer threads, such as the number of performance cores,
    can be faster; try a few values with ``scripts/benchmark.py --threads N``.
    """
    if num_threads:
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()


def describe_device(device: str) -> str:
    """One line per fact about the runtime, for logs and bug reports."""
    lines = [f"PyTorch {torch.__version__} | device: {device}"]
    kind = device_type(device)
    if kind == "cuda" and torch.cuda.is_available():
        index = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(index)
        lines.append(
            f"GPU: {props.name} | capability {props.major}.{props.minor} | "
            f"{props.total_memory / 1024**3:.1f} GiB | CUDA {torch.version.cuda}"
        )
    elif kind == "mps":
        lines.append("GPU: Apple Silicon (MPS)")
    elif kind == "xla":
        lines.append("Accelerator: XLA (TPU), experimental support")
    else:
        lines.append(f"CPU threads: {torch.get_num_threads()} (of {os.cpu_count()} logical cores)")
    return "\n".join(lines)

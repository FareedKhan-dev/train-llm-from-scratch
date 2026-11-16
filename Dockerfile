# syntax=docker/dockerfile:1
#
# GPU image that reproduces the CI environment with a real GPU. The torch cu130
# wheel bundles the CUDA runtime libs (nvidia-* deps), so the base image only
# needs Python; the host needs the NVIDIA driver + nvidia-container-toolkit,
# and runs use --gpus all.
#
#   docker build -t train-llm:gpu .
#   docker run --rm --gpus all train-llm:gpu        # default CMD: pytest -q
#
# The test suite targets the CPU path (as CI does); run it on CPU even inside
# the GPU image — see README -> "Reproduce the CI checks with Docker":
#   docker run --rm -e CUDA_VISIBLE_DEVICES="" train-llm:gpu pytest -q

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SHAPE_CHECKS=1

# Dependency layer (cached separately from source): torch cu130 matches a host
# CUDA 13.0 driver; the wheel bundles the runtime, so no CUDA toolkit is needed.
# Dev tools mirror `pip install -e . pytest beartype ruff mypy` (CONTRIBUTING.md).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir ruff mypy pytest beartype

WORKDIR /app

# Editable install: resolves runtime deps (numpy, h5py, tiktoken, ...) from
# pyproject.toml; torch is already present, so this layer is fast.
COPY . .
RUN pip install --no-cache-dir -e .

# Default to the test suite; any args override it, e.g. `docker run <img> ruff check .`.
CMD ["pytest", "-q"]

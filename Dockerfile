# syntax=docker/dockerfile:1
#
# A GPU image for training and for running the CI checks on a real GPU. The torch
# CUDA wheel bundles the CUDA runtime as nvidia-* pip packages, so the base image
# only needs Python. The host needs the NVIDIA driver and nvidia-container-toolkit,
# and runs use --gpus all.
#
#   docker build -t train-llm:gpu .
#   docker run --rm --gpus all train-llm:gpu          # default CMD: pytest -q
#
# cu130 needs a driver for CUDA 13 (580 or newer). On an older driver, build the
# CUDA 12.6 variant instead:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 -t train-llm:gpu .

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SHAPE_CHECKS=1

# Dependency layer, cached separately from the source. The dev tools mirror
# `pip install -e . pytest beartype ruff mypy` from CONTRIBUTING.md.
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir ruff mypy pytest beartype

WORKDIR /app

# Editable install: resolves the runtime deps (numpy, h5py, tiktoken, ...) from
# pyproject.toml. torch is already present, so this layer is fast.
COPY . .
RUN pip install --no-cache-dir -e ".[train]"

# Default to the test suite; any arguments override it, e.g. `docker run <img> ruff check .`.
CMD ["pytest", "-q"]

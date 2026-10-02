"""
Default locations of everything that is too big for git.

All paths are relative to the repository root, which is where every command in the README
is run from:

    data/     datasets (HDF5 token streams and JSONL files)
    models/   checkpoints of every stage
    logs/     metrics (one JSONL file per run)

To keep them on another disk, point the paths in ``configs/*.json`` somewhere else
(``$VARS`` are expanded, e.g. ``"$SCRATCH/models/sft.pt"``), pass ``--out_ckpt`` and friends
on the command line, or simply make ``data``/``models`` symlinks.
"""

DATA_DIR = "data"
CKPT_DIR = "models"
LOG_DIR = "logs"

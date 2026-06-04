"""Read the newest ``/ephemeral/logs/<prefix>_*.jsonl`` metrics file into a DataFrame."""

from __future__ import annotations

import glob
import json
import os

from ui.stages import LOG_DIR


def latest_log(prefix: str) -> str | None:
    files = glob.glob(os.path.join(LOG_DIR, f"{prefix}_*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def load_metrics(path: str):
    import pandas as pd

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(rows)


def metric_columns(df) -> list[str]:
    """Numeric columns worth plotting (drop bookkeeping)."""
    skip = {"step", "wall"}
    return [c for c in df.columns if c not in skip and df[c].dtype.kind in "fi"]

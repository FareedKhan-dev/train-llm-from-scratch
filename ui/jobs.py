"""
Background job manager for the control panel.

Launches the real training / data-prep scripts as detached subprocesses, captures their
output to a logfile, and records a tiny JSON registry under /ephemeral/ui_jobs/ so jobs
survive Streamlit reruns and page navigation. Includes a GPU-busy guard so we never start a
second multi-GPU job on top of a running one (which would OOM the H100s).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

from ui.stages import REPO_ROOT

JOB_DIR = "/ephemeral/ui_jobs"
_TORCHRUN = os.path.join(os.path.dirname(sys.executable), "torchrun")


def _ensure_job_dir() -> None:
    os.makedirs(JOB_DIR, exist_ok=True)


def _reg(job_id: str) -> str:
    return os.path.join(JOB_DIR, f"{job_id}.json")


def _log(job_id: str) -> str:
    return os.path.join(JOB_DIR, f"{job_id}.log")


def _write_registry(job_id: str, data: dict) -> None:
    with open(_reg(job_id), "w") as f:
        json.dump(data, f)


def read_registry(job_id: str) -> dict | None:
    p = _reg(job_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A finished detached child lingers as a zombie (parent = this process) and still
    # passes kill(pid, 0). Treat zombies as dead and best-effort reap them.
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(") ", 1)[1].split(" ", 1)[0]
        if state == "Z":
            try:
                os.waitpid(pid, os.WNOHANG)
            except OSError:
                pass
            return False
    except (FileNotFoundError, IndexError):
        return False
    return True


def build_argv(script: str, config_json: str, nproc: int, multi_gpu: bool, extra: list[str] | None = None) -> list[str]:
    """Construct the exact command (torchrun for multi-GPU, else python)."""
    base = [script, "--config", config_json] + (extra or [])
    if multi_gpu and nproc > 1:
        return [_TORCHRUN, "--standalone", f"--nproc_per_node={nproc}", *base]
    return [sys.executable, *base]


def launch(job_id: str, argv: list[str], *, kind: str = "cpu") -> dict:
    """Start ``argv`` as a detached background job. ``kind`` is 'gpu' or 'cpu' (for the guard)."""
    _ensure_job_dir()
    log_path = _log(job_id)
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HOME", "/ephemeral/hf_cache")
    logf = open(log_path, "wb")
    proc = subprocess.Popen(
        argv, cwd=REPO_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,   # own process group -> killpg reaches all torchrun ranks
    )
    rec = dict(job_id=job_id, pid=proc.pid, cmd=argv, log=log_path, kind=kind,
               started=time.time(), status="running")
    _write_registry(job_id, rec)
    return rec


def status(job_id: str) -> str:
    """Return 'running' | 'finished' | 'failed' | 'none'."""
    rec = read_registry(job_id)
    if not rec:
        return "none"
    if _alive(rec["pid"]):
        return "running"
    tail = tail_log(job_id, 4000).lower()
    if "traceback" in tail or "error:" in tail or "aborted" in tail:
        return "failed"
    return "finished"


def stop(job_id: str) -> bool:
    """Terminate a running job and all its workers (process group)."""
    rec = read_registry(job_id)
    if not rec:
        return False
    try:
        os.killpg(os.getpgid(rec["pid"]), signal.SIGTERM)
    except OSError:
        return False
    rec["status"] = "stopped"
    _write_registry(job_id, rec)
    return True


def tail_log(job_id: str, max_bytes: int = 16000) -> str:
    p = _log(job_id)
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", errors="replace")


def gpu_busy() -> str | None:
    """Return the job_id of a *running* GPU job, if any (for the launch guard)."""
    if not os.path.isdir(JOB_DIR):
        return None
    for fn in os.listdir(JOB_DIR):
        if not fn.endswith(".json"):
            continue
        rec = read_registry(fn[:-5])
        if rec and rec.get("kind") == "gpu" and _alive(rec["pid"]):
            return rec["job_id"]
    return None


def active_jobs() -> list[dict]:
    out = []
    if not os.path.isdir(JOB_DIR):
        return out
    for fn in sorted(os.listdir(JOB_DIR)):
        if fn.endswith(".json"):
            rec = read_registry(fn[:-5])
            if rec:
                rec["live"] = _alive(rec["pid"])
                out.append(rec)
    return out

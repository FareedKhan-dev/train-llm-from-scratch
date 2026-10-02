"""
Background job manager for the control panel.

Launches the real training / data-prep scripts as detached subprocesses, captures their
output to a logfile, and records a tiny JSON registry under logs/ui_jobs/ so jobs
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

from ui.stages import LOG_DIR, REPO_ROOT

JOB_DIR = os.path.join(LOG_DIR, "ui_jobs")
IS_WINDOWS = os.name == "nt"


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


def _alive_windows(pid: int) -> bool:
    """Ask Windows whether ``pid`` is still running (never os.kill: on Windows it terminates)."""
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _alive(pid: int) -> bool:
    if IS_WINDOWS:
        return _alive_windows(pid)
    try:
        os.kill(pid, 0)  # signal 0 only checks that the process exists (POSIX)
    except OSError:
        return False
    # A finished child of this process lingers as a zombie and still passes kill(pid, 0).
    # waitpid reaps it and tells us it is done. Jobs started by an earlier Streamlit process
    # are not our children, and kill(pid, 0) above is the answer for them.
    try:
        done_pid, _ = os.waitpid(pid, os.WNOHANG)
        return done_pid != pid
    except ChildProcessError:
        return True


def build_argv(script: str, config_json: str, nproc: int, multi_gpu: bool, extra: list[str] | None = None) -> list[str]:
    """Construct the exact command (torchrun for multi-GPU, else python)."""
    base = [script, "--config", config_json] + (extra or [])
    if multi_gpu and nproc > 1:
        # `python -m torch.distributed.run` is torchrun, and works on every OS and venv layout.
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", *base]
    return [sys.executable, *base]


def launch(job_id: str, argv: list[str], *, kind: str = "cpu") -> dict:
    """Start ``argv`` as a detached background job. ``kind`` is 'gpu' or 'cpu' (for the guard)."""
    _ensure_job_dir()
    log_path = _log(job_id)
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logf = open(log_path, "wb")
    # Own process group, so stop() reaches every torchrun rank.
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS
             else {"start_new_session": True})
    proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT, **group)
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
    if IS_WINDOWS:
        # /T stops the whole process tree (every torchrun rank). Its exit code can report a
        # child that already exited, so success means "the job is gone", checked below.
        subprocess.run(["taskkill", "/PID", str(rec["pid"]), "/T", "/F"], capture_output=True)
        for _ in range(20):
            if not _alive(rec["pid"]):
                break
            time.sleep(0.1)
        else:
            return False
    else:
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

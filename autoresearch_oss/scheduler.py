"""Simple interval scheduler for validate / train / monitor (platform background)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid

_schedules: dict[str, dict] = {}
_lock = threading.Lock()
_runner: threading.Thread | None = None
_stop = threading.Event()


def _parse_interval(spec: str) -> int:
    spec = (spec or "").strip().lower()
    if spec.startswith("every "):
        spec = spec[6:].strip()
    if spec.endswith("h"):
        return int(float(spec[:-1]) * 3600)
    if spec.endswith("m"):
        return int(float(spec[:-1]) * 60)
    if spec.endswith("s"):
        return int(float(spec[:-1]))
    return int(spec)


def add_schedule(
    *,
    action: str,
    interval: str,
    config: str,
    db_path: str,
    repo_root: str,
    tenant_id: str = "default",
    extra: dict | None = None,
) -> str:
    sid = uuid.uuid4().hex[:10]
    with _lock:
        _schedules[sid] = {
            "id": sid,
            "action": action,
            "interval_sec": _parse_interval(interval),
            "config": config,
            "db_path": db_path,
            "repo_root": repo_root,
            "tenant_id": tenant_id,
            "extra": extra or {},
            "last_run": None,
            "last_status": None,
        }
    return sid


def list_schedules() -> list[dict]:
    with _lock:
        return [dict(v) for v in _schedules.values()]


def remove_schedule(schedule_id: str) -> bool:
    with _lock:
        return _schedules.pop(schedule_id, None) is not None


def _run_action(job: dict) -> None:
    action = job["action"]
    cfg = job["config"]
    db = job["db_path"]
    env = {**os.environ, "PYTHONPATH": job["repo_root"]}
    cwd = os.path.dirname(os.path.abspath(cfg))
    base = [sys.executable, "-m", "autoresearch_oss"]
    if action == "validate":
        cmd = base + ["validate", "--config", cfg]
    elif action == "train":
        cmd = base + ["train", "--config", cfg, "--db", db, "--out", job["extra"].get("out", "artifacts")]
        if job["extra"].get("promote"):
            cmd.append("--promote")
    elif action == "monitor":
        live = job["extra"].get("live")
        if not live:
            job["last_status"] = "failed: monitor needs extra.live path"
            return
        cmd = base + ["monitor", "--config", cfg, "--live", live]
    else:
        job["last_status"] = f"unknown action {action}"
        return
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    job["last_status"] = "ok" if r.returncode == 0 else f"rc={r.returncode}"
    job["last_log"] = (r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]


def _loop():
    while not _stop.is_set():
        now = time.time()
        with _lock:
            jobs = list(_schedules.values())
        for job in jobs:
            iv = job["interval_sec"]
            last = job.get("last_run") or 0
            if now - last >= iv:
                _run_action(job)
                job["last_run"] = time.time()
        _stop.wait(5)


def start_scheduler():
    global _runner
    if _runner and _runner.is_alive():
        return
    _stop.clear()
    _runner = threading.Thread(target=_loop, daemon=True, name="autoresearch-scheduler")
    _runner.start()


def stop_scheduler():
    _stop.set()

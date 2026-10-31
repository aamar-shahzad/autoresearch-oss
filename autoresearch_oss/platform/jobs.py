"""In-memory background jobs for platform API (train, etc.)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def submit_train(
    *,
    repo_root: str,
    config: str,
    db_path: str,
    out: str,
    promote: bool,
    cwd: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"status": "running", "started": time.time(), "log": ""}

    def run():
        cmd = [
            sys.executable,
            "-m",
            "autoresearch_oss",
            "train",
            "--config",
            config,
            "--db",
            db_path,
            "--out",
            out,
        ]
        if promote:
            cmd.append("--promote")
        env = {**os.environ, "PYTHONPATH": repo_root}
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=cwd or os.path.dirname(os.path.abspath(config)),
            )
            with _lock:
                _jobs[job_id].update(
                    {
                        "status": "done" if r.returncode == 0 else "failed",
                        "returncode": r.returncode,
                        "log": (r.stdout or "") + (r.stderr or ""),
                        "finished": time.time(),
                    }
                )
        except Exception as e:
            with _lock:
                _jobs[job_id].update(
                    {"status": "failed", "error": str(e), "finished": time.time()}
                )

    threading.Thread(target=run, daemon=True).start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        return dict(_jobs.get(job_id) or {}) or None

#!/usr/bin/env python3
"""Complete maintenance-diagnostics use case (synthetic fleet data).

Runs the full AutoResearch loop and prints before/after benefits vs a single
default recipe that would fail promotion gates.

  python examples/run_maintenance_use_case.py
  python examples/run_maintenance_use_case.py --fresh
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UC = os.path.join(ROOT, "examples", "maintenance_use_case")

# Pump-like alarm: first search round often picks replace_filter; promoted model should inspect_pump.
DEMO_INPUTS = {"alarm": "E214", "flow": 4.0, "temp": 92.0, "ambient": 35.0, "vibration": 1.3}


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _cli(*args, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "autoresearch_oss", *args],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": ROOT},
        capture_output=True,
        text=True,
    )


def _ensure_episodes(episodes_src: str) -> None:
    if not os.path.isfile(episodes_src):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "examples", "make_example.py")], cwd=ROOT)


def _setup_uc(fresh: bool) -> tuple[str, str, str]:
    if fresh and os.path.isdir(UC):
        shutil.rmtree(UC)
    os.makedirs(os.path.join(UC, "data"), exist_ok=True)
    episodes_src = os.path.join(ROOT, "examples", "episodes.csv")
    _ensure_episodes(episodes_src)
    episodes_dst = os.path.join(UC, "data", "episodes.csv")
    if not os.path.isfile(episodes_dst):
        shutil.copy2(episodes_src, episodes_dst)

    from autoresearch_oss.project import init_project

    init_project(UC, "maintenance-diagnostics")
    cfg_path = os.path.join(UC, "autoresearch.json")
    cfg = json.load(open(cfg_path))
    cfg["data"]["path"] = episodes_dst
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    db = os.path.join(UC, "registry.db")
    out = os.path.join(UC, "artifacts")
    return cfg_path, db, out


def _metrics_from_report(report_text: str) -> dict:
    m = {}
    for key, pat in (
        ("promoted_id", r"\*\*Model:\*\* `(model/[^`]+)`"),
        ("holdout_fix", r"\| fix_rate \| [^|]+ \| ([0-9.]+) \|"),
        ("holdout_worst", r"\| worst_slice \| [^|]+ \| ([0-9.]+) \|"),
        ("first_full_fix", r"First full run: fix ([0-9.]+)"),
        ("best_full_fix", r"best full run: fix ([0-9.]+)"),
    ):
        hit = re.search(pat, report_text)
        if hit:
            m[key] = hit.group(1)
    fail = re.search(r"pump \(fix ([0-9.]+)", report_text)
    if fail:
        m["pump_failure_fix"] = fail.group(1)
    return m


def _demo_serve(cfg_path: str, db: str, log_path: str) -> None:
    from autoresearch_oss.config import load_config
    from autoresearch_oss.lifecycle import load_model, production
    from autoresearch_oss.ontology import Registry, Workflow
    from autoresearch_oss.serve import Service

    cfg = load_config(cfg_path)
    wf = Workflow.from_dict(cfg["workflow"])
    reg = Registry(db)
    prod = production(reg, wf.name)
    if not prod:
        print("(skip serve demo: no production model)")
        return
    model = load_model(reg, prod["id"], wf)
    svc = Service(model, prod["id"], wf, log_path=log_path)
    srv = svc.make_server("127.0.0.1", 0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)

    def post(path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    try:
        pred = post("/v1/predict", {"inputs": DEMO_INPUTS})
        print(f"  predict -> action={pred['action']!r} confidence={pred['confidence']} model={pred['model']}")
        fb = post(
            "/v1/feedback",
            {
                "request_id": pred["request_id"],
                "final_action": pred["action"],
                "outcome_success": True,
                "slice": "pump",
            },
        )
        print(f"  feedback -> status={fb.get('status')} (logged to {log_path})")
    finally:
        srv.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", action="store_true", help="Delete prior examples/maintenance_use_case outputs")
    args = ap.parse_args()

    _section("USE CASE: Fleet maintenance — pick the repair that clears the alarm")
    print(
        "Story: CNC lines log sensor readings + engineer actions + whether the alarm cleared.\n"
        "Goal: recommend replace_filter | inspect_pump | … with per-root-cause (slice) safety.\n"
        "Pain without AutoResearch: one sklearn fit on all successes; rare 'pump' cases fail in production."
    )

    cfg_path, db, out = _setup_uc(args.fresh)
    os.makedirs(out, exist_ok=True)

    _section("1) Validate data (leakage + thin slices before spend)")
    from autoresearch_oss.project import validate_data

    val = validate_data(json.load(open(cfg_path)))
    print(f"  rows: {val['cleaning']['output_rows']}  success_rate: {val['success_rate']:.1%}")
    for w in val.get("warnings", []):
        print(f"  warning: {w}")

    _section("2) Search recipes under budget → promote only if holdout passes")
    r = _cli("train", "--config", cfg_path, "--promote", "--db", db, "--out", out, cwd=UC)
    if r.returncode not in (0, 2):
        print(r.stderr or r.stdout)
        return r.returncode
    print(r.stdout.strip() or "(train finished)")

    report_path = os.path.join(out, "report.md")
    if not os.path.isfile(report_path):
        print("Missing report.md", file=sys.stderr)
        return 1
    metrics = _metrics_from_report(open(report_path).read())

    _section("3) Benefit — AutoResearch vs “train once and ship”")
    pump_fix = metrics.get("pump_failure_fix", "?")
    first_fix = metrics.get("first_full_fix", "?")
    best_fix = metrics.get("best_full_fix", metrics.get("holdout_fix", "?"))
    worst = metrics.get("holdout_worst", "?")
    holdout = metrics.get("holdout_fix", "?")
    promoted = metrics.get("promoted_id", "?")
    print(
        f"""
  | What | Naive first recipe | After search ({promoted}) |
  |------|--------------------|---------------------------|
  | val fix (approx) | {first_fix} | {best_fix} |
  | worst slice (pump was ~{pump_fix}) | fails gate (<0.8) | {worst} on holdout |
  | holdout fix | not promoted | {holdout} |
  | spend | one manual try | report tracks ~900 budget units |

  Why it matters:
  • Named failure mode (pump → wrong replace_filter) drove extra mixture + synth, not guesswork.
  • Holdout untouched until a candidate passes validation — avoids silent overfit.
  • Registry lineage (see {report_path}) ties model → recipe → dataset → workflow.
"""
    )

    _section("4) Production API — predict, human feedback, JSONL for retrain")
    log_path = os.path.join(UC, "live.jsonl")
    _demo_serve(cfg_path, db, log_path)

    _section("5) Monitor live CSV — drift / override alerts")
    live_csv = os.path.join(ROOT, "examples", "live.csv")
    mon = _cli("monitor", "--config", cfg_path, "--live", live_csv, cwd=UC)
    if mon.returncode == 0:
        body = json.loads(mon.stdout)
        print(f"  retrain recommended: {body['retrain']}  reasons: {body.get('reasons')}")
    else:
        print(mon.stderr or mon.stdout)

    _section("Done")
    print(
        f"Project dir: {UC}\n"
        f"  autoresearch serve --config {cfg_path} --db {db}\n"
        f"  autoresearch train --config {cfg_path} --live-log {log_path} --db {db} --out {out} --promote\n"
        f"Full write-up: docs/use-case-maintenance-diagnostics.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

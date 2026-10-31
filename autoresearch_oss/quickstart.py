"""Guided tour on SAMPLE data (the synthetic maintenance world). Imported only by `quickstart`."""
from __future__ import annotations

import json
import os

from .backends import TabularBackend
from .datasets import ScrubPolicy, export_dataset
from .evaluate import evaluate
from .gate import Deployment
from .lifecycle import load_model, production, promote_to_production
from .loop import AutoResearch, Budget
from .monitor import drift_report, performance_report, should_retrain
from .ontology import Registry
from .report import generate
from .sample import simulate_live
from .sample.cnc import WORKFLOW, gen_episodes, simulate_outcome, synth_from_procedures

SHIFT = {"filter": 0.42, "pump": 0.08, "bearing": 0.18, "fan": 0.32}   # a hot summer


def quickstart(n: int, db: str, out: str, seed: int, live_n: int):
    os.makedirs(out, exist_ok=True)
    reg = Registry(db)
    hist = gen_episodes(n, seed=seed)
    ar = AutoResearch(reg, WORKFLOW, hist, simulate_outcome, TabularBackend(seed=seed), synth_from_procedures,
                      budget=Budget(), seed=seed, artifacts_dir=out, unit_cost_usd=0.02)
    shift_eval = gen_episodes(2000, seed=seed + 99, family_probs=SHIFT, assets=(500, 600), source="shift")
    print("\n=== ROUND 1: historical data only (SAMPLE data) ===")
    s1 = ar.run()
    if not s1.promoted_model_id:
        print("round 1 promoted nothing"); return
    promote_to_production(reg, s1.promoted_model_id)
    m1 = load_model(reg, s1.promoted_model_id)
    v1 = evaluate(m1, shift_eval, WORKFLOW, simulate_outcome).metrics
    print("\n=== DEPLOY v1 behind the approval gate; learn from live operations ===")
    live = simulate_live(Deployment(m1, s1.promoted_model_id), simulate_outcome, live_n, seed + 5, SHIFT)
    print("live outcomes:", live["status"].value_counts().to_dict())
    again, why = should_retrain(drift_report(hist, live, WORKFLOW), performance_report(live, WORKFLOW))
    print("monitor -> retrain?", again, why)
    print("\n=== ROUND 2: history + live episodes (overrides up-weighted) ===")
    s2 = ar.run(extra_episodes=live)
    if s2.promoted_model_id:
        promote_to_production(reg, s2.promoted_model_id)
        v2 = evaluate(load_model(reg, s2.promoted_model_id), shift_eval, WORKFLOW, simulate_outcome).metrics
        print(f"\nShifted test set: v1 fix={v1['fix_rate']:.3f} worst_slice={v1['worst_slice']:.2f} "
              f"-> v2 fix={v2['fix_rate']:.3f} worst_slice={v2['worst_slice']:.2f}")
    pol = ScrubPolicy(hash_cols=["asset"], round_numeric={"flow": 1, "temp": 1, "ambient": 1, "vibration": 2},
                      salt="demo-salt", quasi_identifiers=["site", "alarm"], k=5)
    pub = export_dataset(reg, ar.splits["train"], WORKFLOW, out, "sample-diagnostics", pol, parents=[ar.base_ds_id])
    generate(reg, WORKFLOW.name, 0.02, f"{out}/report.md")
    json.dump(reg.export(), open(f"{out}/registry_export.json", "w"), indent=1)
    print(f"\nProduction model: {production(reg, WORKFLOW.name)['id']} (rollback available)")
    print(f"Published dataset: {pub['path']} ({pub['card']['rows']} rows)   Report: {out}/report.md")

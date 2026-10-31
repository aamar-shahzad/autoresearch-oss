"""Batch eval runs: score production (or staging) model on a fixed case set — not a chat host."""

from __future__ import annotations

import pandas as pd

from .evaluate import check_thresholds, evaluate
from .lifecycle import load_model, production


def run_eval(
    reg,
    wf,
    outcome_fn,
    df: pd.DataFrame,
    *,
    model_id: str | None = None,
    abstain_below: float = 0.0,
) -> dict:
    """One eval run with aggregate metrics + per-row case results."""
    if model_id:
        mid = model_id
        model = load_model(reg, mid, wf)
    else:
        m = production(reg, wf.name)
        if not m:
            raise RuntimeError("no production model; train --promote or pass model_id")
        mid = m["id"]
        model = load_model(reg, mid, wf)

    ev = evaluate(model, df, wf, outcome_fn, abstain_below=abstain_below)
    passed = check_thresholds(ev.metrics, wf.thresholds)
    pred = model.predict(df)
    cases = []
    for j, (_, row) in enumerate(df.iterrows()):
        cases.append(
            {
                "case_id": str(row.get("episode_id", j)),
                "slice": row.get(wf.slice_by),
                "expected_action": str(row.get("action", "")),
                "predicted_action": str(pred["action"].iloc[j]),
                "confidence": float(pred["confidence"].iloc[j]),
            }
        )

    payload = {
        "model_id": mid,
        "workflow": wf.name,
        "n_cases": len(df),
        "metrics": ev.metrics,
        "slices": ev.slices,
        "failure_modes": ev.failure_modes,
        "passed_thresholds": passed,
        "cases": cases,
    }
    aid = reg.put("eval_run", wf.name, payload, parents=[mid])
    return {"id": aid, **payload}

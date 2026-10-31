"""Step 10, quantified: is production still like training, and are people overriding the model?"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate, >0.25 major shift."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    b = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((b - a) * np.log(b / a)))


def drift_report(ref: pd.DataFrame, live: pd.DataFrame, wf, psi_threshold: float = 0.25,
                 mix_threshold: float = 0.10) -> dict:
    feats = {c: psi(ref[c].dropna().to_numpy(float), live[c].dropna().to_numpy(float))
             for c in wf.inputs if pd.api.types.is_numeric_dtype(ref[c])}
    a = ref[wf.slice_by].value_counts(normalize=True)
    b = live[wf.slice_by].value_counts(normalize=True)
    keys = a.index.union(b.index)
    shifts = {k: float(b.get(k, 0) - a.get(k, 0)) for k in keys}
    tvd = 0.5 * sum(abs(v) for v in shifts.values())
    return {"feature_psi": feats, "slice_mix_shift": shifts, "slice_mix_tvd": float(tvd),
            "drifted_features": [c for c, v in feats.items() if v > psi_threshold],
            "mix_shifted": bool(tvd > mix_threshold)}


def performance_report(live: pd.DataFrame, wf, max_override=0.10, max_fail=0.05) -> dict:
    n = len(live)
    rates = live["status"].value_counts(normalize=True).to_dict()
    by_slice = (live.assign(bad=live["status"].isin(["failed", "overridden"]))
                .groupby(wf.slice_by)["bad"].agg(["mean", "size"]).round(4).to_dict("index"))
    return {"n": n, "rates": rates, "bad_by_slice": by_slice,
            "override_alert": rates.get("overridden", 0) > max_override,
            "failure_alert": rates.get("failed", 0) > max_fail}


def should_retrain(drift: dict, perf: dict) -> tuple:
    reasons = []
    if drift["drifted_features"]:
        reasons.append(f"feature drift: {drift['drifted_features']}")
    if drift["mix_shifted"]:
        reasons.append(f"slice mix shifted (TVD={drift['slice_mix_tvd']:.2f})")
    if perf["override_alert"]:
        reasons.append("override rate above limit")
    if perf["failure_alert"]:
        reasons.append("failure rate above limit")
    return bool(reasons), reasons

"""Step 7: score a candidate on held-back episodes against thresholds that were
set in advance. Failure modes are named so the loop knows what to change."""
from __future__ import annotations

import operator
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

OPS = {">=": operator.ge, "<=": operator.le}


@dataclass
class EvalResult:
    metrics: dict
    slices: dict
    failure_modes: list = field(default_factory=list)


def evaluate(model, df, wf, outcome_fn, abstain_below: float = 0.0, min_slice: int = 20) -> EvalResult:
    pred = model.predict(df)
    act, conf = pred["action"].to_numpy(), pred["confidence"].to_numpy()
    esc = conf < abstain_below
    # outcome_fn may return bools (twin) or probabilities in [0,1] (offline estimates)
    ok = np.asarray(outcome_fn(df, act), dtype=float) * (~esc)
    unsafe = np.isin(act, wf.unsafe_actions)
    if wf.unsafe_patterns:
        rx = [re.compile(p, re.I | re.S) for p in wf.unsafe_patterns]
        unsafe = unsafe | np.array([any(r.search(str(a)) for r in rx) for a in act])
    unsafe = unsafe & ~esc
    fam = df[wf.slice_by].to_numpy()
    slices = {f: float(ok[fam == f].mean()) for f in np.unique(fam) if (fam == f).sum() >= min_slice}
    succ = df["outcome_success"].to_numpy(dtype=bool)
    required = list(wf.slice_values) or list(np.unique(fam))
    uncovered = [f for f in required if f not in slices]   # a slice we cannot measure is a blind spot
    metrics = {
        "uncovered_slices": len(uncovered),
        "fix_rate": float(ok.mean()),
        "unsafe_rate": float(unsafe.mean()),
        "costly_error_rate": float((np.isin(act, wf.costly_actions) * (1 - ok) * (~esc)).mean()),
        "escalation_rate": float(esc.mean()),
        "worst_slice": float(min(slices.values())) if slices else float("nan"),
        # weak proxy usable without a twin: agree with the human on episodes that worked
        "agreement_on_successes": float((act[succ] == df["action"].to_numpy()[succ]).mean()) if succ.any() else float("nan"),
        "n": int(len(df)),
    }
    floor = wf.thresholds.get("worst_slice", [">=", 0.8])[1]
    modes = []
    for f, r in sorted(slices.items(), key=lambda kv: kv[1]):
        if r < floor:
            bad = (fam == f) & (ok < 0.5)
            top = Counter(act[bad]).most_common(1)
            modes.append({"slice": f, "fix_rate": r, "n": int((fam == f).sum()),
                          "most_common_wrong_action": top[0][0] if top else None})
    return EvalResult(metrics, slices, modes)


def check_thresholds(metrics: dict, thresholds: dict) -> dict:
    return {m: bool(OPS[op](metrics[m], v)) for m, (op, v) in thresholds.items()}


def utility(metrics: dict) -> float:
    return metrics["fix_rate"] - 5 * metrics["unsafe_rate"] + 0.3 * metrics["worst_slice"]

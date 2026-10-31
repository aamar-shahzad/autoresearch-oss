"""A simulated CNC fleet: the paper's worked example (alarm E214, coolant filter)
made runnable. Doubles as (a) a source of historical 'episodes', (b) a digital
twin that scores ANY proposed action (needed to evaluate a model's
counterfactual recommendations), and (c) a procedure-driven synthetic generator
standing in for 'manuals -> synthetic tasks'. Swap these for your real
connectors / twin / LLM generator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ontology import Workflow

ACTIONS = ["replace_filter", "inspect_pump", "replace_bearing", "clean_fan", "restart_no_check"]
FAMILIES = ["filter", "pump", "bearing", "fan"]
REPLACE = {"replace_filter", "replace_bearing"}
DEFAULT_FAMILY_PROBS = {"filter": 0.50, "pump": 0.004, "bearing": 0.20, "fan": 0.296}

WORKFLOW = Workflow(
    name="cnc-spindle-overheat",
    description="Recommend the action that clears a spindle-overheating alarm.",
    inputs=["alarm", "flow", "temp", "ambient", "vibration"],
    outcome_fields=["outcome_success", "flow_after", "temp_after"],
    action_space=ACTIONS,
    unsafe_actions=["restart_no_check"],
    slice_by="fault_family",
    thresholds={"fix_rate": [">=", 0.90], "unsafe_rate": ["<=", 0.01], "worst_slice": [">=", 0.80],
                "uncovered_slices": ["<=", 0]},
    slice_values=FAMILIES,
    group_by="asset",
    costly_actions=sorted(REPLACE),
)


def correct_action(df: pd.DataFrame) -> np.ndarray:
    """Ground truth ('the procedure manual'). High-ambient low-flow is a pump issue,
    NOT a filter issue -- the rare case a naive model will miss."""
    return np.where(df["vibration"] > 4, "replace_bearing",
           np.where(df["flow"] < 6,
                    np.where(df["ambient"] >= 30, "inspect_pump", "replace_filter"),
                    "clean_fan"))


def simulate_outcome(df: pd.DataFrame, actions) -> np.ndarray:
    """Digital twin: did this action clear the alarm?"""
    return np.asarray(actions) == correct_action(df)


def gen_episodes(n: int, seed: int = 0, family_probs: dict | None = None, human_acc: float = 0.72,
                 label_noise: float = 0.03, assets: tuple = (1, 300), source: str = "history",
                 id_prefix: str = "H") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = family_probs or DEFAULT_FAMILY_PROBS
    probs = np.array([p.get(f, 0.0) for f in FAMILIES], dtype=float)
    fam = rng.choice(FAMILIES, size=n, p=probs / probs.sum())
    m = {f: fam == f for f in FAMILIES}
    u = lambda lo, hi: rng.uniform(lo, hi, n)
    flow = np.select([m["filter"], m["pump"], m["bearing"], m["fan"]],
                     [u(3, 5.8), u(3, 5.8), u(6.5, 12), u(6.5, 12)]) + rng.normal(0, .15, n)
    temp = np.select([m["filter"], m["pump"], m["bearing"], m["fan"]],
                     [u(80, 95), u(85, 95), u(60, 85), u(80, 95)]) + rng.normal(0, .5, n)
    ambient = np.where(m["pump"], u(30, 38), np.clip(rng.normal(22, 4, n), 10, 29))
    vib = np.where(m["bearing"], u(4.5, 8), u(.5, 3.5))
    aid = rng.integers(assets[0], assets[1], n)
    df = pd.DataFrame({
        "episode_id": [f"{id_prefix}{seed}-{i}" for i in range(n)],
        "asset": [f"CNC-{a:03d}" for a in aid], "site": aid % 6,
        "period": rng.integers(0, 12, n),
        "alarm": np.where(vib > 4, "E330", "E214"),
        "flow": flow, "temp": temp, "ambient": ambient, "vibration": vib,
    })
    corr = correct_action(df)
    idx = np.array([ACTIONS.index(c) for c in corr])
    wrong = np.array(ACTIONS)[(idx + rng.integers(1, len(ACTIONS), n)) % len(ACTIONS)]
    df["action"] = np.where(rng.random(n) < human_acc, corr, wrong)
    ok = (df["action"].values == corr) ^ (rng.random(n) < label_noise)
    df["outcome_success"] = ok
    df["flow_after"] = np.where(ok, 10.0, flow)
    df["temp_after"] = np.where(ok, 72.0, temp)
    df["fault_family"] = fam
    df["source"] = source
    df["status"] = "historical"
    return df


def synth_from_procedures(family: str, n: int, seed: int = 0) -> pd.DataFrame:
    """Targeted synthetic episodes for a family, generated from the procedure
    (here: the rule table; in production: an LLM prompted with manuals + faults)."""
    probs = {f: (1.0 if f == family else 0.0) for f in FAMILIES}
    d = gen_episodes(n, seed=10_000 + seed, family_probs=probs, human_acc=1.0, label_noise=0.0,
                     assets=(0, 1), source="synthetic", id_prefix="S")
    d["asset"] = "SYNTH"
    d["status"] = "synthetic"
    return d

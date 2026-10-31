"""Writes examples/episodes.csv, examples/live.csv and examples/config.json so you can try the
bring-your-own-data path:  python examples/make_example.py &&
python -m autoresearch_oss run --config examples/config.json --out artifacts
Replace the CSV with your own export and edit the workflow block in config.json."""
import json
import os
from dataclasses import asdict

import numpy as np

from autoresearch_oss.sample.cnc import WORKFLOW, gen_episodes

here = os.path.dirname(os.path.abspath(__file__))
gen_episodes(8000, seed=0).drop(columns=["flow_after", "temp_after"]).to_csv(f"{here}/episodes.csv", index=False)

live = gen_episodes(1500, seed=7, family_probs={"filter": .42, "pump": .08, "bearing": .18, "fan": .32},
                    assets=(300, 400), source="live").drop(columns=["flow_after", "temp_after"])
rng = np.random.default_rng(1)
live["status"] = rng.choice(["accepted", "overridden", "failed"], len(live), p=[.80, .12, .08])  # illustrative
live.to_csv(f"{here}/live.csv", index=False)

wf = asdict(WORKFLOW)
wf["outcome_fields"] = ["outcome_success"]          # post-action readings were dropped above
wf["thresholds"] = {"fix_rate": [">=", 0.85], "unsafe_rate": ["<=", 0.02],
                    "worst_slice": [">=", 0.70], "uncovered_slices": ["<=", 0]}
cfg = {
    "workflow": wf,
    "data": {"path": f"{here}/episodes.csv"},
    "outcome": {"type": "replay", "k": 15},          # no twin: offline, pessimistic estimate
    "synth": {"type": "resample", "noise": 0.05},
    "backend": {"type": "tabular", "bases": ["hgb", "rf"]},
    "budget": {"max_compute": 900, "max_rounds": 4},
    "unit_cost_usd": 0.02,
    "export": {"hash_cols": ["asset"], "round_numeric": {"flow": 1, "temp": 1}, "salt": "change-me",
               "quasi_identifiers": ["site", "alarm"], "k": 5},
}
json.dump(cfg, open(f"{here}/config.json", "w"), indent=1)
print("wrote examples/{episodes.csv,live.csv,config.json}")

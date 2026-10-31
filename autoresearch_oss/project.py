"""Project scaffolding and data validation."""
from __future__ import annotations

import json
import os

import pandas as pd

from .connectors import load_episodes, validate_episodes
from .ontology import Workflow
from .quality import clean

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def list_templates() -> dict:
    return {f[:-5]: json.load(open(os.path.join(TEMPLATES, f)))["description"]
            for f in sorted(os.listdir(TEMPLATES)) if f.endswith(".json")}


def init_project(path: str, template: str) -> list:
    t = json.load(open(os.path.join(TEMPLATES, f"{template}.json")))
    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    cfg = {k: v for k, v in t["config"].items() if not k.startswith("_")}
    files = [os.path.join(path, "autoresearch.json")]
    json.dump(cfg, open(files[0], "w"), indent=1)
    ex = os.path.join(path, "data", "episodes.example.csv")
    pd.DataFrame(t["example_rows"]).to_csv(ex, index=False)
    doc = os.path.join(path, "data", "README.md")
    with open(doc, "w") as f:
        f.write(f"# Data for `{template}`\n\nPut your episodes in `data/episodes.csv` (one row per case + action + measured outcome).\n\n"
                "| column | meaning |\n|---|---|\n" + "\n".join(f"| `{k}` | {v} |" for k, v in t["columns"].items()) +
                "\n\nCheck it with: `autoresearch validate --config autoresearch.json`\n")
    return files + [ex, doc]


def validate_data(cfg: dict) -> dict:
    wf = Workflow.from_dict(cfg["workflow"])
    df = load_episodes(cfg["data"], wf)
    clean_df, rep = clean(df, wf)
    warn = []
    if len(clean_df) < 300:
        warn.append(f"only {len(clean_df)} usable episodes: expect noisy search results below a few hundred")
    ok = clean_df["outcome_success"]
    if ok.mean() > 0.97 or ok.mean() < 0.03:
        warn.append(f"outcome_success is {ok.mean():.0%} true: little signal about what fails")
    sl = clean_df.groupby(wf.slice_by).agg(n=("episode_id", "size"), success=("outcome_success", "mean")).round(3)
    thin = sl[sl["n"] < 30].index.tolist()
    if thin:
        warn.append(f"thin slices (<30 episodes): {thin}; the loop will need synthetic data to evaluate them")
    declared = set(wf.slice_values) - set(sl.index)
    if declared:
        warn.append(f"declared slices with ZERO episodes: {sorted(declared)} (cannot be learned or evaluated)")
    groups = clean_df[wf.group_by].nunique()
    if groups < 10:
        warn.append(f"only {groups} distinct {wf.group_by} values: group-wise splits will be lumpy")
    return {"cleaning": rep, "success_rate": float(ok.mean()), "slices": sl.to_dict("index"),
            "groups": int(groups), "problems": validate_episodes(df, wf), "warnings": warn}

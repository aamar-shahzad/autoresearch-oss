"""Step 3: isolate contradictory, incomplete and leaked records."""
from __future__ import annotations

from .ontology import Workflow


def clean(df, wf: Workflow):
    wf.validate()  # raises on input/outcome leakage
    rep = {"input_rows": len(df)}
    d = df.dropna(subset=wf.inputs + ["action", "outcome_success"])
    rep["incomplete"] = len(df) - len(d)
    nun = d.groupby("episode_id")["outcome_success"].nunique()
    bad = nun[nun > 1].index
    d = d[~d["episode_id"].isin(bad)]
    rep["contradictory"] = int(len(bad))
    n0 = len(d)
    d = d.drop_duplicates("episode_id")
    rep["duplicates"] = n0 - len(d)
    rep["output_rows"] = len(d)
    return d.reset_index(drop=True), rep

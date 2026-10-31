"""Eval run stores per-case predictions on a split."""

import joblib

from autoresearch_oss.config import build_parts
from autoresearch_oss.eval_run import run_eval
from autoresearch_oss.lifecycle import promote_to_production
from autoresearch_oss.ontology import Registry, Workflow
from autoresearch_oss.sample.cnc import WORKFLOW, gen_episodes


class _StubModel:
    def predict(self, df):
        import pandas as pd

        return pd.DataFrame(
            {"action": df["action"].values, "confidence": [0.9] * len(df)}
        )


def test_eval_run_on_holdout(tmp_path):
    db = str(tmp_path / "r.db")
    reg = Registry(db)
    reg.put("workflow", WORKFLOW.name, WORKFLOW.__dict__)
    wpath = tmp_path / "m.joblib"
    joblib.dump(_StubModel(), wpath)
    mid = reg.put(
        "model",
        WORKFLOW.name,
        {
            "stage": "staging",
            "recipe": "test",
            "holdout_metrics": {"fix_rate": 0.9},
            "weights": str(wpath),
        },
    )
    promote_to_production(reg, mid)
    df = gen_episodes(200, seed=2).drop(columns=["flow_after", "temp_after"])
    wf = Workflow.from_dict(WORKFLOW.__dict__)
    _, outcome, _, _ = build_parts(
        {
            "workflow": WORKFLOW.__dict__,
            "outcome": {"type": "replay"},
            "backend": {"type": "tabular"},
        }
    )
    outcome.fit(df, wf)
    r = run_eval(reg, wf, outcome, df.head(50), model_id=mid)
    assert r["n_cases"] == 50
    assert len(r["cases"]) == 50
    assert "metrics" in r
    assert reg.list("eval_run")

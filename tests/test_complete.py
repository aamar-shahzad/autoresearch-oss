import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from autoresearch_oss.connectors import load_episodes, validate_episodes
from autoresearch_oss.datasets import ScrubPolicy, export_dataset, scrub, verify_export
from autoresearch_oss.feedback import tag_status
from autoresearch_oss.lifecycle import production, promote_to_production, rollback
from autoresearch_oss.monitor import drift_report, performance_report, psi, should_retrain
from autoresearch_oss.ontology import Registry, Workflow
from autoresearch_oss.outcomes import ReplayOutcome
from autoresearch_oss.report import generate
from autoresearch_oss.synth import LLMSynth, ResampleSynth
from autoresearch_oss.sample.cnc import WORKFLOW, correct_action, gen_episodes, simulate_outcome

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _train(n=6000, seed=0):
    return gen_episodes(n, seed=seed)


def test_replay_outcome_semantics():
    d = _train()
    r = ReplayOutcome(k=15).fit(d, WORKFLOW)
    # same action as logged -> observed outcome, exactly
    assert (r(d.iloc[:200], d.iloc[:200]["action"]) == d.iloc[:200]["outcome_success"].astype(float)).all()
    # a never-logged action label is unsupported -> pessimistic 0
    assert (r(d.iloc[:50], ["nonexistent"] * 50) == 0).all()
    # proposing the true best action should beat proposing a wrong one, in estimate
    good = r(d.iloc[:500], correct_action(d.iloc[:500])).mean()
    bad = r(d.iloc[:500], ["restart_no_check"] * 500).mean()
    assert good > bad + 0.3


def test_resample_synth_and_empty_slice():
    d = _train()
    s = ResampleSynth().fit(d, WORKFLOW)
    out = s("pump", 40, 1)
    assert len(out) == 40 and (out["source"] == "synthetic").all() and (out["ambient"] > 25).all()
    assert len(s("does-not-exist", 10, 1)) == 0      # honest: cannot invent what it has never seen


def test_llm_synth_validates_and_verifies():
    d = _train()
    good = {"alarm": "E214", "flow": 4.0, "temp": 90.0, "ambient": 33.0, "vibration": 1.0, "action": "inspect_pump"}
    wrong = dict(good, action="replace_filter")           # valid schema, but does not work
    out_of_range = dict(good, flow=999)
    bad_action = dict(good, action="explode")
    fake = lambda prompt: "sure! " + json.dumps([good, wrong, out_of_range, bad_action]) + " done"
    s = LLMSynth(fake, verify=simulate_outcome, max_calls=1).fit(d, WORKFLOW)
    out = s("pump", 4, 0)
    assert len(out) == 1 and out.iloc[0]["action"] == "inspect_pump" and out["verified"].all()
    junk = LLMSynth(lambda p: "no json here", max_calls=2).fit(d, WORKFLOW)
    assert len(junk("pump", 3, 0)) == 0


def test_connector_validation(tmp_path):
    d = _train(300)
    p = tmp_path / "e.csv"
    d.drop(columns=["alarm"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing column 'alarm'"):
        load_episodes({"path": str(p)}, WORKFLOW)
    d2 = d.copy(); d2["action"] = "bogus"
    assert any("action_space" in x for x in validate_episodes(d2, WORKFLOW))
    q = tmp_path / "r.csv"
    d.rename(columns={"outcome_success": "result"}).assign(result=lambda x: np.where(x["result"], "PASS", "FAIL")).to_csv(q, index=False)
    ok = load_episodes({"path": str(q), "columns": {"result": "outcome_success"}, "success_values": ["PASS"]}, WORKFLOW)
    assert ok["outcome_success"].dtype == bool


def test_sqlite_connector(tmp_path):
    import sqlite3
    d = _train(300)
    db = tmp_path / "x.db"
    with sqlite3.connect(db) as c:
        d.to_sql("eps", c, index=False)
    out = load_episodes({"path": str(db), "format": "sqlite", "query": "select * from eps"}, WORKFLOW)
    assert len(out) == 300


def test_scrub_and_export(tmp_path):
    d = _train(2000)
    d["note"] = "call bob@example.com or +1 (555) 123-4567 from 10.0.0.1"
    pol = ScrubPolicy(hash_cols=["asset"], text_cols=["note"], round_numeric={"flow": 1},
                      drop=["period"], quasi_identifiers=["site", "alarm"], k=5, salt="s")
    out, rep = scrub(d, pol)
    assert "period" not in out and not out["asset"].str.startswith("CNC").any()
    assert "example.com" not in " ".join(out["note"]) and "<email>" in out["note"].iloc[0]
    assert (out["flow"] == out["flow"].round(1)).all()
    reg = Registry()
    r = export_dataset(reg, d, WORKFLOW, str(tmp_path), "t", pol, markings=("Restricted",))
    assert verify_export(r["path"], r["card"])
    with gzip.open(r["path"], "rt") as f:
        assert len(pd.read_csv(f)) == r["card"]["rows"]
    with open(r["path"], "ab") as f:
        f.write(b"tamper")
    assert not verify_export(r["path"], r["card"])
    assert reg.get(r["id"])["payload"]["markings"] == ["Restricted"]


def test_monitor_detects_shift():
    ref = _train(4000, 0)
    same = gen_episodes(2000, seed=9)
    shifted = gen_episodes(2000, seed=9, family_probs={"filter": .1, "pump": .4, "bearing": .1, "fan": .4})
    assert psi(ref["flow"].to_numpy(), same["flow"].to_numpy()) < 0.1
    assert not should_retrain(drift_report(ref, same, WORKFLOW), {"override_alert": False, "failure_alert": False})[0]
    again, why = should_retrain(drift_report(ref, shifted, WORKFLOW), {"override_alert": False, "failure_alert": False})
    assert again and why
    live = shifted.assign(status=["overridden"] * 500 + ["accepted"] * 1500)
    assert performance_report(live, WORKFLOW)["override_alert"]


def test_tag_status():
    logs = pd.DataFrame({"model_rec": ["a", "a", "a"], "action": ["a", "b", "a"],
                         "outcome_success": [True, True, False]})
    assert tag_status(logs)["status"].tolist() == ["accepted", "overridden", "failed"]


def test_lifecycle_promote_and_rollback():
    reg = Registry()
    m1 = reg.put("model", "w", {"weights": "x.joblib", "stage": "staging"})
    m2 = reg.put("model", "w", {"weights": "y.joblib", "stage": "staging"})
    promote_to_production(reg, m1)
    promote_to_production(reg, m2)
    assert production(reg, "w")["id"] == m2 and reg.get(m1)["payload"]["stage"] == "archived"
    assert rollback(reg, "w") == m1 and production(reg, "w")["id"] == m1
    with pytest.raises(RuntimeError):
        rollback(reg, "w")


def test_end_to_end_cli_on_csv(tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT)
    subprocess.run([sys.executable, f"{ROOT}/examples/make_example.py"], check=True, env=env, capture_output=True)
    db, out = str(tmp_path / "r.db"), str(tmp_path / "out")
    r = subprocess.run([sys.executable, "-m", "autoresearch_oss", "train", "--config", f"{ROOT}/examples/config.json",
                        "--db", db, "--out", out, "--promote"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "PROMOTED" in r.stdout
    rep = open(f"{out}/report.md").read()
    assert "Model card" in rep and "| metric |" in rep
    reg = Registry(db)
    assert production(reg, WORKFLOW.name) is not None

"""Enterprise-style features: ontology materialize, studies, tenant, audit."""

import os

import pandas as pd
import pytest

from autoresearch_oss.episode_materialize import (
    materialize_episodes_from_entities,
    sync_case_entities_from_episodes,
)
from autoresearch_oss.ontology import Registry, Workflow
from autoresearch_oss.project import init_project
from autoresearch_oss.studies import experiment_digest, propose_study_recipes


def test_tenant_isolation(tmp_path):
    db = str(tmp_path / "r.db")
    r1 = Registry(db, tenant_id="t1")
    r2 = Registry(db, tenant_id="t2")
    r1.put("workflow", "wf", {"name": "wf"})
    assert r2.list("workflow") == []
    assert len(r1.list("workflow")) == 1


def test_case_materialize_roundtrip(tmp_path):
    db = str(tmp_path / "r.db")
    proj = tmp_path / "p"
    init_project(str(proj), "maintenance-diagnostics")
    cfg_path = proj / "autoresearch.json"
    import shutil

    shutil.copy(proj / "data" / "episodes.example.csv", proj / "data" / "episodes.csv")
    from autoresearch_oss.config import load_config
    from autoresearch_oss.connectors import load_episodes

    cfg = load_config(str(cfg_path))
    wf = Workflow.from_dict(cfg["workflow"])
    prev = os.getcwd()
    try:
        os.chdir(proj)
        df = load_episodes(cfg["data"], wf)
    finally:
        os.chdir(prev)
    reg = Registry(db)
    sync_case_entities_from_episodes(reg, df.head(4), wf)
    back = materialize_episodes_from_entities(reg, wf, ["Case"])
    assert len(back) == 4
    assert set(wf.inputs).issubset(back.columns)


def test_study_proposes_from_prior_experiment(tmp_path):
    db = str(tmp_path / "r.db")
    reg = Registry(db)
    wf = Workflow.from_dict(
        {
            "name": "wf",
            "description": "d",
            "inputs": ["x"],
            "outcome_fields": ["outcome_success"],
            "action_space": ["a"],
            "unsafe_actions": [],
            "slice_by": "x",
            "thresholds": {"fix_rate": [">=", 0.9]},
            "slice_values": ["s1"],
        }
    )
    reg.put(
        "experiment",
        "e1",
        {
            "stage": "full",
            "recipe": "test",
            "val_metrics": {"fix_rate": 0.5, "worst_slice": 0.4, "unsafe_rate": 0},
            "failure_modes": [{"slice": "s1", "fix_rate": 0.3, "n": 10, "most_common_wrong_action": "a"}],
        },
    )

    class _B:
        name = "tabular"
        bases = ["cpu"]

    recipes, digest = propose_study_recipes(reg, wf, _B(), ["imitate"], use_llm=False)
    assert digest["count"] >= 1
    assert len(recipes) >= 1


def test_audit_table(tmp_path):
    db = str(tmp_path / "r.db")
    from autoresearch_oss.governance import list_audit, record_audit
    from autoresearch_oss.registry_store import RegistryStore

    store = RegistryStore(str(db), "default")
    record_audit(store.reg.db, tenant_id="default", actor="u", action="test", detail={"x": 1})
    rows = list_audit(store.reg.db, "default")
    assert rows[0]["action"] == "test"

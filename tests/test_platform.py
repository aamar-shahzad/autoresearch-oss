"""Platform API, ontology entities, dashboard static assets."""

import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from autoresearch_oss import builtins  # noqa: F401
from autoresearch_oss.entities import export_ontology, sync_entities_from_episodes
from autoresearch_oss.lifecycle import promote_to_production
from autoresearch_oss.ontology import Registry
from autoresearch_oss.platform.api import create_app
from autoresearch_oss.sample.cnc import WORKFLOW, gen_episodes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_entity_sync_and_export(tmp_path):
    reg = Registry(str(tmp_path / "r.db"))
    df = gen_episodes(100, seed=1).drop(columns=["flow_after", "temp_after"])
    stats = sync_entities_from_episodes(
        reg, df, {"asset": {"type": "Asset"}, "site": {"type": "Site"}}
    )
    assert stats["links"]["asset"] > 0
    ont = export_ontology(reg)
    assert len(ont["entity_types"]) >= 2
    assert len(ont["entities"]) >= stats["links"]["asset"]


def test_platform_api_models_promote_rollback(tmp_path):
    import joblib

    db = str(tmp_path / "r.db")
    wpath = tmp_path / "m.joblib"
    joblib.dump({"stub": True}, wpath)
    reg = Registry(db)
    reg.put("workflow", WORKFLOW.name, WORKFLOW.__dict__)
    mid = reg.put(
        "model",
        WORKFLOW.name,
        {
            "stage": "staging",
            "recipe": "test",
            "holdout_metrics": {"fix_rate": 0.9, "worst_slice": 0.85, "unsafe_rate": 0},
            "weights": str(wpath),
        },
    )
    promote_to_production(reg, mid)

    app = create_app(db)
    client = TestClient(app)
    assert client.get("/health").json()["ok"] is True
    models = client.get(f"/api/v1/models?workflow={WORKFLOW.name}").json()
    assert any(m["id"] == mid for m in models)
    prod = client.get(f"/api/v1/models/production?workflow={WORKFLOW.name}").json()
    assert prod["id"] == mid
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_platform_api_lineage(tmp_path):
    db = str(tmp_path / "r.db")
    reg = Registry(db)
    w = reg.put("workflow", "wf", {"name": "wf"})
    m = reg.put("model", "wf", {"stage": "staging"}, parents=[w])
    app = create_app(db)
    client = TestClient(app)
    lin = client.get(f"/api/v1/lineage/{m}").json()
    assert any(x["id"] == m for x in lin)


def test_platform_api_validate_doctor_templates(tmp_path):
    db = str(tmp_path / "r.db")
    proj = tmp_path / "proj"
    from autoresearch_oss.project import init_project

    init_project(str(proj), "maintenance-diagnostics")
    import shutil

    ex = proj / "data" / "episodes.example.csv"
    shutil.copy(ex, proj / "data" / "episodes.csv")
    cfg = str(proj / "autoresearch.json")
    app = create_app(db)
    client = TestClient(app)
    assert client.get("/api/v1/doctor").json().get("cpu_model")
    assert "maintenance-diagnostics" in client.get("/api/v1/templates").json()
    prev = os.getcwd()
    try:
        os.chdir(proj)
        v = client.post("/api/v1/validate", json={"config": cfg}).json()
    finally:
        os.chdir(prev)
    assert "problems" in v

"""Model lifecycle on top of the registry: staging -> production -> archived, with rollback."""
from __future__ import annotations


def load_model(reg, model_id: str, workflow=None):
    """Tabular models load from .joblib; LLM adapters need the workflow (prompt format) and torch."""
    w = reg.get(model_id)["payload"]["weights"]
    if w.endswith(".joblib"):
        import joblib
        return joblib.load(w)
    if workflow is None:
        raise ValueError("loading an LLM adapter needs the workflow: load_model(reg, id, workflow)")
    from .backends.llm import HFModel
    return HFModel.load(w, workflow)


def production(reg, workflow_name: str):
    for m in reversed(reg.list("model")):
        if m["name"] == workflow_name and m["payload"].get("stage") == "production":
            return m
    return None


def promote_to_production(reg, model_id: str) -> str:
    m = reg.get(model_id)
    cur = production(reg, m["name"])
    if cur and cur["id"] != model_id:
        reg.update(cur["id"], {"stage": "archived"})
    reg.update(model_id, {"stage": "production", "replaces": cur["id"] if cur else None})
    return model_id


def rollback(reg, workflow_name: str) -> str:
    cur = production(reg, workflow_name)
    if not cur or not cur["payload"].get("replaces"):
        raise RuntimeError("nothing to roll back to")
    prev = cur["payload"]["replaces"]
    reg.update(cur["id"], {"stage": "rolled_back"})
    reg.update(prev, {"stage": "production"})
    return prev

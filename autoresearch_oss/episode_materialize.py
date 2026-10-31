"""Ontology-driven episodes: Case entities ↔ training DataFrame."""

from __future__ import annotations

import uuid

import pandas as pd

from .entities import ensure_entity_type, upsert_entity
from .ontology import Registry, Workflow

CASE_TYPE = "Case"


def default_episode_field_map(wf: Workflow) -> dict:
    cols = ["episode_id", wf.group_by, wf.slice_by, "action", "outcome_success"]
    cols += list(wf.inputs)
    if wf.task_type == "generation":
        cols = [c for c in cols if c != "action"] + ["action"]
    return {"fields": {c: c for c in cols}, "workflow_key": "_workflow"}


def load_episode_field_map(cfg: dict, wf: Workflow) -> dict:
    m = cfg.get("entity_episode_map") or {}
    if not m.get("fields"):
        m = {**default_episode_field_map(wf), **m}
    return m


def sync_case_entities_from_episodes(
    reg: Registry, df: pd.DataFrame, wf: Workflow, *, tenant_id: str | None = None
) -> dict:
    """Write each episode row as a Case entity (reverse of materialize)."""
    ensure_entity_type(
        reg,
        CASE_TYPE,
        {"description": "Workflow episode stored in the ontology", "key": "episode_id"},
    )
    tid = tenant_id or getattr(reg, "tenant_id", "default")
    n = 0
    for _, row in df.iterrows():
        eid = str(row.get("episode_id") or uuid.uuid4().hex[:12])
        props = {c: row[c] for c in df.columns if c in row.index}
        props["_workflow"] = wf.name
        props["_tenant_id"] = tid
        upsert_entity(reg, CASE_TYPE, eid, props)
        n += 1
    return {"cases_upserted": n, "workflow": wf.name}


def materialize_episodes_from_entities(
    reg: Registry,
    wf: Workflow,
    entity_types: list[str] | None = None,
    field_map: dict | None = None,
    *,
    tenant_id: str | None = None,
) -> pd.DataFrame:
    """Build episode DataFrame from ontology entities (UI: pick Asset/Case types → trainable rows)."""
    types = set(entity_types or [CASE_TYPE])
    fmap = field_map or default_episode_field_map(wf)
    fields = fmap.get("fields") or {}
    wf_key = fmap.get("workflow_key", "_workflow")
    tid = tenant_id or getattr(reg, "tenant_id", "default")
    rows: list[dict] = []
    for ent in reg.list("entity"):
        p = ent["payload"]
        if p.get("type") not in types:
            continue
        props = p.get("properties") or {}
        if props.get(wf_key) and props.get(wf_key) != wf.name:
            continue
        if props.get("_tenant_id") and props.get("_tenant_id") != tid:
            continue
        row: dict = {}
        for ep_col, prop_key in fields.items():
            if prop_key in props:
                row[ep_col] = props[prop_key]
            elif ep_col in props:
                row[ep_col] = props[ep_col]
        if "action" not in row and "action" in props:
            row["action"] = props["action"]
        if "outcome_success" not in row and "outcome_success" in props:
            row["outcome_success"] = props["outcome_success"]
        if not row.get("episode_id"):
            row["episode_id"] = p.get("external_id") or ent["id"]
        missing_in = [c for c in wf.inputs if c not in row]
        if missing_in:
            continue
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(fields.keys()))
    return pd.DataFrame(rows)


def link_entity_to_workflow(reg: Registry, entity_id: str, workflow_name: str) -> None:
    ent = reg.get(entity_id)
    props = ent["payload"].get("properties") or {}
    wfs = set(props.get("workflows") or [])
    wfs.add(workflow_name)
    props["workflows"] = sorted(wfs)
    reg.update(entity_id, {"properties": props})

"""Ontology-lite: shared entity types and registry-backed entities linked from episode columns."""

from __future__ import annotations

from .ontology import Registry

DEFAULT_ENTITY_TYPES = {
    "Asset": {"description": "Physical machine or equipment id", "key": "asset_id"},
    "Site": {"description": "Plant or location", "key": "site_id"},
    "Customer": {"description": "Customer account", "key": "customer_id"},
}


def ensure_entity_type(reg: Registry, name: str, meta: dict | None = None) -> str:
    existing = [e for e in reg.list("entity_type") if e["name"] == name]
    if existing:
        return existing[-1]["id"]
    payload = meta or DEFAULT_ENTITY_TYPES.get(
        name, {"description": name, "key": f"{name.lower()}_id"}
    )
    return reg.put("entity_type", name, payload)


def upsert_entity(
    reg: Registry, type_name: str, external_id: str, properties: dict | None = None
) -> str:
    ensure_entity_type(reg, type_name)
    ext = str(external_id)
    for e in reg.list("entity"):
        p = e["payload"]
        if p.get("type") == type_name and p.get("external_id") == ext:
            if properties:
                reg.update(
                    e["id"], {"properties": {**p.get("properties", {}), **properties}}
                )
            return e["id"]
    return reg.put(
        "entity",
        f"{type_name}:{ext}",
        {"type": type_name, "external_id": ext, "properties": properties or {}},
    )


def sync_entities_from_episodes(reg: Registry, df, entity_links: dict) -> dict:
    """entity_links: column -> {type, id_column?} — column value becomes external_id."""
    stats = {"created": 0, "links": {}}
    for col, spec in entity_links.items():
        if col not in df.columns:
            continue
        type_name = spec.get("type") if isinstance(spec, dict) else spec
        if not type_name:
            continue
        ensure_entity_type(reg, type_name)
        ids = df[col].dropna().astype(str).unique()
        stats["links"][col] = len(ids)
        for ext in ids:
            upsert_entity(reg, type_name, ext)
            stats["created"] += 1
    return stats


def export_ontology(reg: Registry) -> dict:
    return {
        "entity_types": [
            {"id": e["id"], "name": e["name"], **e["payload"]}
            for e in reg.list("entity_type")
        ],
        "entities": [
            {"id": e["id"], "name": e["name"], **e["payload"]}
            for e in reg.list("entity")
        ],
        "workflows": [
            {"id": w["id"], "name": w["name"], "payload": w["payload"]}
            for w in reg.list("workflow")
        ],
    }


def load_entity_links(cfg: dict) -> dict:
    return cfg.get("entity_links") or {}

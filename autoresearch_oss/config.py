"""Build a full run from a JSON config. Every `type` is looked up in the plugin registry."""
from __future__ import annotations

import json

from . import builtins  # noqa: F401  (registers built-ins)
from .connectors import load_episodes
from .loop import AutoResearch, Budget
from .ontology import Registry, Workflow
from .plugins import get


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_parts(cfg: dict):
    wf = Workflow.from_dict(cfg["workflow"])
    ospec = cfg.get("outcome", {"type": "replay"})
    outcome = get("outcome", ospec.get("type", "replay"))(ospec)
    sspec = cfg.get("synth", {"type": "resample"})
    synth = get("synth", sspec.get("type", "resample"))(sspec, outcome)
    bspec = cfg.get("backend", {"type": "tabular"})
    backend = get("backend", bspec.get("type", "tabular"))(bspec, outcome, wf)
    return wf, outcome, synth, backend


def build(cfg: dict, db: str, out: str, log=print):
    wf, outcome, synth, backend = build_parts(cfg)
    episodes = load_episodes(cfg["data"], wf)
    tenant = cfg.get("tenant_id", "default")
    ar = AutoResearch(
        Registry(db, tenant_id=tenant),
        wf,
        episodes,
        outcome,
        backend,
        synth,
        techniques=cfg.get("techniques"),
        budget=Budget(**cfg.get("budget", {})),
        seed=cfg.get("seed", 0),
        artifacts_dir=out,
        log=log,
        generic_synth=cfg.get("generic_synth", False),
        unit_cost_usd=cfg.get("unit_cost_usd", 0.0),
        study=cfg.get("study"),
    )
    return ar, episodes, wf

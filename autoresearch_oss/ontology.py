"""The 'Fine-Tuning Ontology': a versioned, lineage-tracked registry of the
three asset classes in the paper -- datasets, techniques/recipes, models --
plus experiments. Backed by SQLite so it is a single portable file you own."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Workflow:
    """Step 1 of the paper: fix inputs, outputs and the rule for 'correct'."""
    name: str
    description: str
    inputs: list
    outcome_fields: list          # measured AFTER the action; never model inputs
    action_space: list
    unsafe_actions: list
    slice_by: str                 # column used for per-slice (failure-mode) analysis
    thresholds: dict              # metric -> [op, value]
    slice_values: list = field(default_factory=list)  # every slice that MUST be evaluated
    group_by: str = "asset"                           # splits never share a group (e.g. machine)
    costly_actions: list = field(default_factory=list)  # expensive actions; wrong ones are tracked
    task_type: str = "choice"          # choice: pick from action_space | generation: free text + verifier
    prompt_template: str = ""          # e.g. "Ticket: {subject}\n{body}"; default renders inputs as key=value
    unsafe_patterns: list = field(default_factory=list)  # regexes a generated answer must never match

    @classmethod
    def from_dict(cls, d: dict) -> "Workflow":
        names = {f for f in cls.__dataclass_fields__}
        w = cls(**{k: v for k, v in d.items() if k in names})
        w.validate()
        return w

    def validate(self) -> None:
        if self.task_type not in ("choice", "generation"):
            raise ValueError("task_type must be 'choice' or 'generation'")
        if self.task_type == "choice" and not self.action_space:
            raise ValueError("choice workflows need an action_space")
        leaked = set(self.inputs) & set(self.outcome_fields)
        if leaked:
            raise ValueError(f"Leakage: outcome fields used as inputs: {sorted(leaked)}")


def _clean(o: Any):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class Registry:
    """Generic artifact store. kinds: workflow, dataset, evalset, episodes,
    recipe, experiment, model."""

    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY, kind TEXT, name TEXT,"
            " version INT, parents TEXT, payload TEXT, created REAL)"
        )
        self.db.commit()

    def put(self, kind: str, name: str, payload: dict, parents=()) -> str:
        n = self.db.execute(
            "SELECT COUNT(*) FROM artifacts WHERE kind=? AND name=?", (kind, name)
        ).fetchone()[0]
        version = n + 1
        aid = f"{kind}/{name}@v{version}"
        self.db.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
            (aid, kind, name, version, json.dumps(list(parents)),
             json.dumps(_clean(payload)), time.time()),
        )
        self.db.commit()
        return aid

    def get(self, aid: str) -> dict:
        r = self.db.execute("SELECT * FROM artifacts WHERE id=?", (aid,)).fetchone()
        if not r:
            raise KeyError(aid)
        return {"id": r[0], "kind": r[1], "name": r[2], "version": r[3],
                "parents": json.loads(r[4]), "payload": json.loads(r[5]), "created": r[6]}

    def update(self, aid: str, patch: dict) -> None:
        cur = self.get(aid)["payload"]
        cur.update(_clean(patch))
        self.db.execute("UPDATE artifacts SET payload=? WHERE id=?", (json.dumps(cur), aid))
        self.db.commit()

    def list(self, kind: str) -> list:
        rows = self.db.execute(
            "SELECT id FROM artifacts WHERE kind=? ORDER BY rowid", (kind,)
        ).fetchall()
        return [self.get(r[0]) for r in rows]

    def lineage(self, aid: str) -> list:
        seen, stack, out = set(), [aid], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            a = self.get(cur)
            out.append(a)
            stack.extend(a["parents"])
        return out

    def export(self) -> list:
        return [self.get(r[0]) for r in self.db.execute("SELECT id FROM artifacts ORDER BY rowid")]

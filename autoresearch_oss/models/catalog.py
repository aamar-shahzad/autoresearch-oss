"""Curated catalog of open-weight base models. Static data ships in catalog.json; the user can
extend it with a local JSON file (same schema) or live Hub search. Extend by adding entries --
no code changes needed."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
USER_CATALOG = os.path.expanduser("~/.autoresearch/catalog.json")


@dataclass
class Entry:
    id: str                       # Hugging Face repo id
    family: str
    params_b: float               # TOTAL parameters (billions): this is what memory follows
    vocab: int = 150_000
    active_params_b: float | None = None   # MoE only
    license_id: str = "unknown"
    license_class: str = "custom"  # permissive | custom | research
    gated: bool = False
    tags: list = field(default_factory=list)
    notes: str = ""

    @property
    def commercial_ok(self) -> str:
        return {"permissive": "yes", "custom": "with conditions", "research": "no"}.get(self.license_class, "check")

    @property
    def short(self) -> str:
        return self.id.split("/")[-1]


def load_catalog(extra_paths=()) -> list:
    entries = {}
    for p in [os.path.join(_HERE, "catalog.json"), USER_CATALOG, *extra_paths]:
        if p and os.path.exists(p):
            for m in json.load(open(p))["models"]:
                entries[m["id"]] = Entry(**{k: v for k, v in m.items() if k in Entry.__dataclass_fields__})
    return list(entries.values())


def filter_entries(entries, license_class=None, family=None, max_params=None, min_params=None,
                   tag=None, query=None):
    out = []
    for e in entries:
        if license_class and e.license_class not in (license_class if isinstance(license_class, (list, tuple, set)) else [license_class]):
            continue
        if family and family.lower() not in e.family.lower():
            continue
        if max_params is not None and e.params_b > max_params:
            continue
        if min_params is not None and e.params_b < min_params:
            continue
        if tag and tag not in e.tags:
            continue
        if query and query.lower() not in (e.id + " " + e.family).lower():
            continue
        out.append(e)
    return out


def find(entries, repo: str) -> Entry | None:
    return next((e for e in entries if e.id == repo or e.short == repo), None)

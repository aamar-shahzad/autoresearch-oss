"""Extension points. Everything a team is likely to customise is looked up by name here, so you
extend the product by registering a function -- not by forking it.

  kind        factory signature
  ---------   ------------------------------------------------------------
  connector   fn(spec: dict, wf) -> DataFrame                  (data sources)
  outcome     fn(spec: dict) -> callable(df, actions)->scores  (twin / verifier / judge)
  verifier    fn(spec: dict) -> callable(row, completion)->0..1
  synth       fn(spec: dict, outcome) -> callable(slice, n, seed)->DataFrame
  backend     fn(spec: dict, outcome, wf) -> training backend
  hub         fn(spec: dict) -> object with .info/.search/.download  (model sources)

Register in your own package and expose it with an entry point:

    [project.entry-points."autoresearch_oss.plugins"]
    mycorp = "mycorp_autoresearch"          # importing this module runs its @register calls
"""
from __future__ import annotations

import importlib.metadata as md

KINDS = ("connector", "outcome", "verifier", "synth", "backend", "hub")
_REG: dict = {k: {} for k in KINDS}
_loaded = False


def register(kind: str, name: str):
    if kind not in _REG:
        raise ValueError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")

    def deco(fn):
        _REG[kind][name] = fn
        return fn
    return deco


def get(kind: str, name: str):
    load_entry_points()
    try:
        return _REG[kind][name]
    except KeyError:
        raise KeyError(f"no {kind} named {name!r}. available: {sorted(_REG[kind])}") from None


def find(kind: str, name: str):
    load_entry_points()
    return _REG[kind].get(name)


def available(kind: str | None = None) -> dict:
    load_entry_points()
    return {k: sorted(v) for k, v in _REG.items() if kind in (None, k)}


def load_entry_points() -> list:
    global _loaded
    errs = []
    if _loaded:
        return errs
    _loaded = True
    try:
        eps = md.entry_points(group="autoresearch_oss.plugins")
    except TypeError:
        eps = md.entry_points().get("autoresearch_oss.plugins", [])
    for ep in eps:
        try:
            ep.load()
        except Exception as e:       # a broken third-party plugin must not break the product
            errs.append(f"{ep.name}: {e}")
    return errs

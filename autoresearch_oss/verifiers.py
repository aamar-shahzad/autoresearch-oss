"""Verifiers turn 'is this generated answer good?' into a score in [0,1] -- the verifiable reward
that makes evaluation, GRPO and verified synthetic data possible for free-text tasks.

A verifier is fn(row, completion) -> float. `VerifierOutcome` adapts one to the loop's
outcome_fn(df, completions) interface. For tasks with a reference answer, `row["action"]` holds the
logged GOOD response (generation workflows evaluate on successful episodes only)."""
from __future__ import annotations

import json
import re

import numpy as np

from .plugins import register


def _json(text: str):
    t = text.strip()
    for a, b in (("{", "}"), ("[", "]")):
        if a in t and b in t:
            try:
                return json.loads(t[t.index(a): t.rindex(b) + 1])
            except json.JSONDecodeError:
                pass
    return None


@register("verifier", "exact")
def exact(spec):
    ci = spec.get("case_insensitive", True)
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower() if ci else str(s).strip()
    return lambda row, comp: float(norm(comp) == norm(row["action"]))


@register("verifier", "regex")
def regex(spec):
    rx = re.compile(spec["pattern"], re.I | re.S)
    return lambda row, comp: float(bool(rx.search(str(comp))))


@register("verifier", "json_fields")
def json_fields(spec):
    """Valid JSON; required keys present; optionally the listed fields equal the reference's."""
    req, match = spec.get("required", []), spec.get("match", [])

    def fn(row, comp):
        got = _json(comp)
        if not isinstance(got, dict) or any(k not in got for k in req):
            return 0.0
        if match:
            ref = _json(str(row["action"])) or {}
            hits = [str(got.get(k)).strip().lower() == str(ref.get(k)).strip().lower() for k in match]
            return float(np.mean(hits))
        return 1.0
    return fn


@register("verifier", "callable")
def callable_(spec):
    from .outcomes import load_callable
    return load_callable(spec["callable"])


class VerifierOutcome:
    verifiable = True

    def __init__(self, verifier):
        self.verifier = verifier

    def __call__(self, df, completions) -> np.ndarray:
        rows = df.to_dict("records")
        return np.array([float(self.verifier(r, c)) for r, c in zip(rows, completions)])


@register("outcome", "verifier")
def _outcome(spec):
    from .plugins import get
    return VerifierOutcome(get("verifier", spec["verifier"]["type"])(spec["verifier"]))

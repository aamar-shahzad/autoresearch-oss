"""Step 2: bring your own episodes. CSV / Parquet / SQLite or SQLAlchemy URL -> the
episode table the loop expects (episode_id, <group_by>, action, outcome_success,
<slice_by>, + workflow inputs)."""
from __future__ import annotations

import os
import sqlite3

import pandas as pd

from .plugins import available, find, register


@register("connector", "csv")
def _csv(spec, wf):
    return pd.read_csv(spec["path"])


@register("connector", "parquet")
def _parquet(spec, wf):
    return pd.read_parquet(spec["path"])           # needs pyarrow


@register("connector", "jsonl")
def _jsonl(spec, wf):
    return pd.read_json(spec["path"], lines=True)


@register("connector", "sqlite")
def _sqlite(spec, wf):
    with sqlite3.connect(spec["path"]) as con:
        return pd.read_sql_query(spec["query"], con)


@register("connector", "sql")
def _sql(spec, wf):
    from sqlalchemy import create_engine           # optional dependency
    return pd.read_sql_query(spec["query"], create_engine(spec["url"]))


def load_episodes(spec: dict, wf) -> pd.DataFrame:
    path = spec.get("path", "")
    fmt = spec.get("format") or (os.path.splitext(path)[1].lstrip(".") or "csv")
    fmt = {"db": "sqlite", "json": "jsonl"}.get(fmt, fmt)
    reader = find("connector", fmt)
    if reader is None:
        raise ValueError(f"no connector for format {fmt!r}; available: {available('connector')['connector']}")
    df = reader(spec, wf)
    df = df.rename(columns=spec.get("columns", {}))
    if "success_values" in spec:             # e.g. {"success_values": ["PASS", "resolved"]}
        df["outcome_success"] = df["outcome_success"].isin(spec["success_values"])
    df["outcome_success"] = df["outcome_success"].astype(bool)
    if "episode_id" not in df:
        df["episode_id"] = [f"E{i}" for i in range(len(df))]
    for k, v in {"source": "history", "status": "historical", **spec.get("defaults", {})}.items():
        if k not in df:
            df[k] = v
    problems = validate_episodes(df, wf)
    if problems:
        raise ValueError("episode table invalid:\n - " + "\n - ".join(problems))
    return df


def validate_episodes(df: pd.DataFrame, wf) -> list:
    need = set(wf.inputs) | {"episode_id", wf.group_by, wf.slice_by, "action", "outcome_success"}
    probs = [f"missing column {c!r}" for c in sorted(need - set(df.columns))]
    if not probs:
        bad = sorted(set(df["action"].dropna()) - set(wf.action_space))
        if bad:
            probs.append(f"actions not in action_space: {bad[:5]}")
        if df["outcome_success"].nunique() < 2:
            probs.append("outcome_success has one class only; cannot learn from failures")
    return probs

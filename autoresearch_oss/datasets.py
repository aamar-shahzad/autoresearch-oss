"""Publish a training dataset as a first-class asset: scrubbed, versioned, hashed, with a
data card, lineage and security markings -- the paper's 'distribute / license' path.
NOTE: scrubbing here is a practical baseline (drop, salted-hash, round, regex, k-anonymity
suppression). It is NOT a formal privacy guarantee (no differential privacy)."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd

PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
       (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "<phone>"),
       (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>")]


@dataclass
class ScrubPolicy:
    drop: list = field(default_factory=list)          # columns removed entirely
    hash_cols: list = field(default_factory=list)     # salted one-way pseudonyms (keeps joins)
    text_cols: list = field(default_factory=list)     # free text: regex PII redaction
    round_numeric: dict = field(default_factory=dict)  # col -> decimals
    quasi_identifiers: list = field(default_factory=list)
    k: int = 5                                         # suppress rows in groups smaller than k
    salt: str = ""


def scrub(df: pd.DataFrame, pol: ScrubPolicy):
    d, rep = df.copy(), {"rows_in": len(df)}
    d = d.drop(columns=[c for c in pol.drop if c in d])
    for c in pol.hash_cols:
        d[c] = d[c].astype(str).map(lambda v: hashlib.sha256((pol.salt + v).encode()).hexdigest()[:12])
    for c in pol.text_cols:
        for rx, rep_to in PII:
            d[c] = d[c].astype(str).str.replace(rx, rep_to, regex=True)
    for c, n in pol.round_numeric.items():
        d[c] = d[c].round(n)
    if pol.quasi_identifiers:
        sizes = d.groupby(pol.quasi_identifiers)[d.columns[0]].transform("size")
        rep["suppressed_k_anonymity"] = int((sizes < pol.k).sum())
        d = d[sizes >= pol.k]
    rep["rows_out"] = len(d)
    return d.reset_index(drop=True), rep


def export_dataset(reg, df: pd.DataFrame, wf, out_dir: str, name: str, policy: ScrubPolicy | None = None,
                   markings=("Restricted",), license: str = "proprietary", parents=()) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    clean_df, rep = scrub(df, policy or ScrubPolicy())
    data_path = os.path.join(out_dir, f"{name}.csv.gz")
    with gzip.open(data_path, "wt") as f:
        clean_df.to_csv(f, index=False)
    sha = hashlib.sha256(open(data_path, "rb").read()).hexdigest()
    card = {"name": name, "workflow": wf.name, "description": wf.description, "rows": len(clean_df),
            "columns": {c: str(t) for c, t in clean_df.dtypes.items()},
            "inputs": wf.inputs, "target": "action", "success_field": "outcome_success",
            "class_balance": clean_df["outcome_success"].value_counts(normalize=True).round(4).to_dict(),
            "synthetic_share": float((clean_df.get("source", pd.Series(dtype=str)) == "synthetic").mean()) if len(clean_df) else 0.0,
            "scrubbing": rep, "markings": list(markings), "license": license, "sha256": sha,
            "caveats": ["scrubbing is baseline, not a formal privacy guarantee",
                        "synthetic rows are flagged in the 'source' column"]}
    with open(os.path.join(out_dir, f"{name}.card.json"), "w") as f:
        json.dump(card, f, indent=1)
    aid = reg.put("dataset", f"published-{name}", card, parents)
    return {"id": aid, "path": data_path, "card": card}


def verify_export(path: str, card: dict) -> bool:
    return hashlib.sha256(open(path, "rb").read()).hexdigest() == card["sha256"]

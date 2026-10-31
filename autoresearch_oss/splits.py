"""Step 4: partition BEFORE anything is trained. Splitting by asset (machine)
means a recipe can't score by recognising a machine it has already seen."""
from __future__ import annotations

import hashlib

import numpy as np


def _bucket(key, seed) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_by_group(df, group: str = "asset", fracs=(0.8, 0.1, 0.1), seed: int = 0) -> dict:
    b = df[group].map(lambda k: _bucket(k, seed)).values
    tag = np.where(b < fracs[0], "train", np.where(b < fracs[0] + fracs[1], "val", "eval"))
    out = {s: df[tag == s].reset_index(drop=True) for s in ("train", "val", "eval")}
    sets = [set(out[s][group]) for s in out]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "group leakage across splits"
    return out

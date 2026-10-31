"""Step 5: a mixture is a weighted selection of real episodes plus synthetic
episodes for cases the record lacks. Each mixture is hashable/versionable."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class MixtureSpec:
    name: str
    use_failures: bool = False          # keep episodes whose action failed?
    synth_frac: float = 0.0             # share of final mixture that is synthetic
    synth_families: tuple = ()
    upweight: dict = field(default_factory=dict)   # family -> replication factor
    override_weight: int = 3            # replicate live episodes where a human overrode the model

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()[:8]


def build_mixture(train: pd.DataFrame, spec: MixtureSpec, synth_fn, seed: int = 0,
                  slice_col: str = "fault_family") -> pd.DataFrame:
    d = train if spec.use_failures else train[train["outcome_success"]]
    parts = [d]
    for fam, k in spec.upweight.items():
        sub = d[(d[slice_col] == fam) & d["outcome_success"]]
        if len(sub) and k > 1:
            parts.append(sub.sample(int(len(sub) * (k - 1)), replace=True, random_state=seed))
    ov = d[(d.get("status", pd.Series("", index=d.index)) == "overridden") & d["outcome_success"]]
    if len(ov) and spec.override_weight > 1:
        parts.append(ov.sample(len(ov) * (spec.override_weight - 1), replace=True, random_state=seed))
    base = pd.concat(parts, ignore_index=True)
    if spec.synth_frac > 0 and spec.synth_families:
        n = int(len(base) * spec.synth_frac / (1 - spec.synth_frac))
        per = max(1, n // len(spec.synth_families))
        syn = [synth_fn(f, per, seed + i) for i, f in enumerate(spec.synth_families)]
        syn = [x for x in syn if len(x)]
        base = pd.concat([base] + syn, ignore_index=True)
    return base.sample(frac=1.0, random_state=seed).reset_index(drop=True)

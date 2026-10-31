"""Synthetic-episode generators (step 5). All share one signature:
    synth(slice_value, n, seed) -> DataFrame of episodes (source='synthetic')
and may define fit(train, wf), which the loop calls with TRAIN data only.

* ResampleSynth  -- jittered bootstrap of real successful episodes in that slice (no LLM).
* LLMSynth       -- ask any LLM (you supply `complete(prompt)->str`) for new episodes,
                    then validate ranges/schema and (strongly recommended) VERIFY each
                    row with your twin so only actions that really work are kept."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _finish(rows: pd.DataFrame, wf, slice_value, seed: int, tag: str) -> pd.DataFrame:
    rows = rows.copy().reset_index(drop=True)
    rows["episode_id"] = [f"SYN{tag}{seed}-{i}" for i in range(len(rows))]
    rows[wf.group_by] = "SYNTH"
    rows[wf.slice_by] = slice_value
    rows["outcome_success"] = True
    rows["source"] = "synthetic"
    rows["status"] = "synthetic"
    return rows


class ResampleSynth:
    def __init__(self, noise: float = 0.05):
        self.noise = noise

    def fit(self, train, wf):
        self.train, self.wf = train, wf
        self.num = [c for c in wf.inputs if pd.api.types.is_numeric_dtype(train[c])]
        return self

    def __call__(self, slice_value, n: int, seed: int = 0) -> pd.DataFrame:
        wf = self.wf
        pool = self.train[(self.train[wf.slice_by] == slice_value) & self.train["outcome_success"]]
        if pool.empty:
            return self.train.iloc[0:0].copy()      # nothing to learn from: report honestly
        rng = np.random.default_rng(seed)
        rows = pool.sample(n, replace=True, random_state=seed).copy()
        for c in self.num:
            sd = pool[c].std() if len(pool) > 2 else self.train[c].std()
            rows[c] = rows[c] + rng.normal(0, self.noise * (sd if sd == sd else 0), n)
        return _finish(rows, wf, slice_value, seed, "R")


class LLMSynth:
    def __init__(self, complete, verify=None, examples: int = 3, max_calls: int = 6):
        self.complete, self.verify, self.examples, self.max_calls = complete, verify, examples, max_calls

    def fit(self, train, wf):
        self.train, self.wf = train, wf
        self.rng_ = {}
        for c in wf.inputs:
            if pd.api.types.is_numeric_dtype(train[c]):
                lo, hi = float(train[c].min()), float(train[c].max())
                self.rng_[c] = (lo - .25 * (hi - lo), hi + .25 * (hi - lo))
            else:
                self.rng_[c] = sorted(train[c].dropna().astype(str).unique())
        return self

    def _prompt(self, slice_value, n) -> str:
        wf = self.wf
        ex = self.train[(self.train[wf.slice_by] == slice_value) & self.train["outcome_success"]]
        ex = ex.head(self.examples)[list(wf.inputs) + ["action"]].to_dict("records")
        return (f"Task: {wf.description}\nGenerate {n} NEW, realistic, diverse episodes for the "
                f"case '{slice_value}' where the chosen action succeeds.\n"
                f"Input fields and allowed ranges/values: {json.dumps(self.rng_)}\n"
                f"Allowed actions: {wf.action_space}\nReal examples: {json.dumps(ex, default=str)}\n"
                f"Reply with ONLY a JSON array of objects having keys {list(wf.inputs)+['action']}.")

    def _valid(self, r: dict) -> bool:
        if r.get("action") not in self.wf.action_space:
            return False
        for c, rg in self.rng_.items():
            v = r.get(c)
            if isinstance(rg, tuple):
                if not isinstance(v, (int, float)) or not (rg[0] <= v <= rg[1]):
                    return False
            elif str(v) not in rg:
                return False
        return True

    def __call__(self, slice_value, n: int, seed: int = 0) -> pd.DataFrame:
        kept = []
        for _ in range(self.max_calls):
            if len(kept) >= n:
                break
            txt = self.complete(self._prompt(slice_value, n - len(kept)))
            try:
                arr = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
            except (ValueError, json.JSONDecodeError):
                continue
            rows = [r for r in arr if isinstance(r, dict) and self._valid(r)]
            if rows and self.verify is not None:
                df = pd.DataFrame(rows)
                rows = [r for r, ok in zip(rows, np.asarray(self.verify(df, df["action"])) >= 0.5) if ok]
            kept += rows
        if not kept:
            return self.train.iloc[0:0].copy()
        df = _finish(pd.DataFrame(kept[:n]), self.wf, slice_value, seed, "L")
        df["verified"] = self.verify is not None
        return df

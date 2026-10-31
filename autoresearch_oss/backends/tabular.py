"""CPU backend so the whole loop runs anywhere, on ANY tabular workflow. Stand-in for
LLM post-training: same interface, registry and search -- different model.
'imitate' ~ SFT on successes; 'contrast' ~ learning from failures (reward-model
flavour of preference learning)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


class Featurizer:
    """Numeric columns pass through (median-imputed); everything else is one-hot."""

    def __init__(self, inputs):
        self.inputs = list(inputs)

    def fit(self, df: pd.DataFrame) -> "Featurizer":
        self.num = [c for c in self.inputs if pd.api.types.is_numeric_dtype(df[c])]
        self.cat = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.inputs if c not in self.num}
        self.med = {c: float(df[c].median()) for c in self.num}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = [df[c].fillna(self.med[c]).to_numpy(dtype=float) for c in self.num]
        for c, vals in self.cat.items():
            s = df[c].astype(str)
            cols += [(s == v).to_numpy(dtype=float) for v in vals]
        return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def _est(base: str, hp: dict, seed: int):
    if base == "rf":
        return RandomForestClassifier(n_estimators=hp.get("max_iter", 100) // 2 or 50,
                                      max_depth=hp.get("max_depth", 8),
                                      min_samples_leaf=hp.get("min_samples_leaf", 30),
                                      n_jobs=-1, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=hp.get("max_iter", 100),
                                          max_depth=hp.get("max_depth", 8),
                                          min_samples_leaf=hp.get("min_samples_leaf", 30),
                                          learning_rate=0.1, random_state=seed)


@dataclass
class TabularModel:
    est: object
    technique: str
    actions: list
    feat: Featurizer

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.feat.transform(df)
        if self.technique == "imitate":
            P, acts = self.est.predict_proba(X), np.array(self.est.classes_)
        else:
            cols = []
            for a in self.actions:
                oh = np.tile([float(a == b) for b in self.actions], (len(X), 1))
                cols.append(self.est.predict_proba(np.hstack([X, oh]))[:, 1])
            P, acts = np.array(cols).T, np.array(self.actions)
        i = P.argmax(1)
        return pd.DataFrame({"action": acts[i], "confidence": P[np.arange(len(P)), i]})

    def save(self, path: str) -> None:
        joblib.dump(self, path)


@dataclass
class TrainResult:
    model: TabularModel
    compute_units: float
    seconds: float


class TabularBackend:
    name = "tabular"

    def __init__(self, bases=("hgb", "rf"), seed: int = 0):
        self.bases, self.seed = list(bases), seed

    def train(self, recipe, df: pd.DataFrame, workflow) -> TrainResult:
        t0, hp = time.time(), recipe.hp
        feat = Featurizer(workflow.inputs).fit(df)
        if recipe.technique == "imitate":
            d = df[df["outcome_success"]]
            est = _est(recipe.base, hp, self.seed).fit(feat.transform(d), d["action"])
        elif recipe.technique == "contrast":
            acts = workflow.action_space
            oh = np.array([[float(a == b) for b in acts] for a in df["action"]])
            est = _est(recipe.base, hp, self.seed).fit(np.hstack([feat.transform(df), oh]),
                                                       df["outcome_success"].astype(int))
        else:
            raise ValueError(f"tabular backend has no technique {recipe.technique!r}")
        cu = len(df) * hp.get("max_iter", 100) / 1e4   # abstract compute units
        return TrainResult(TabularModel(est, recipe.technique, list(workflow.action_space), feat),
                           cu, time.time() - t0)

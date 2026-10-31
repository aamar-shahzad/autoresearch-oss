"""Outcome functions answer: 'would this proposed action have worked?'

* a digital twin / simulator (best): any callable (df, actions) -> bool array
* ReplayOutcome (no twin): offline counterfactual ESTIMATE from logged episodes.

ReplayOutcome rules (direct-method, deliberately pessimistic):
  - proposed == logged action      -> the observed outcome (unbiased)
  - proposed != logged, supported  -> success rate of the k nearest logged cases that took
                                      that action (only if close enough to real data)
  - proposed != logged, unsupported-> 0  (we cannot claim it works where we have no evidence)
It cannot see confounding, so treat it as a screening estimate, not proof."""
from __future__ import annotations

import importlib

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .backends.tabular import Featurizer


def load_callable(path: str):
    mod, fn = path.split(":")
    return getattr(importlib.import_module(mod), fn)


class ReplayOutcome:
    def __init__(self, k: int = 15, radius_quantile: float = 0.95):
        self.k, self.q = k, radius_quantile
        self.support = {}

    def fit(self, train, wf):
        self.feat = Featurizer(wf.inputs).fit(train)
        X = self.feat.transform(train)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        self.per = {}
        for a, g in train.groupby("action"):
            if len(g) <= self.k:
                continue
            Z = (self.feat.transform(g) - self.mu) / self.sd
            nn = NearestNeighbors().fit(Z)
            d, _ = nn.kneighbors(Z, n_neighbors=self.k + 1)
            radius = float(np.quantile(d[:, 1:].mean(1), self.q))
            self.per[a] = (nn, g["outcome_success"].to_numpy(dtype=float), radius)
        return self

    def __call__(self, df, actions) -> np.ndarray:
        actions = np.asarray(actions)
        logged = df["action"].to_numpy()
        same = actions == logged
        out = np.zeros(len(df))
        out[same] = df["outcome_success"].to_numpy(dtype=float)[same]
        Z = (self.feat.transform(df) - self.mu) / self.sd
        supported = same.copy()
        for a in np.unique(actions[~same]):
            idx = np.where(~same & (actions == a))[0]
            if a not in self.per:
                continue
            nn, y, radius = self.per[a]
            d, i = nn.kneighbors(Z[idx], n_neighbors=self.k)
            ok = d.mean(1) <= radius
            out[idx] = np.where(ok, y[i].mean(1), 0.0)
            supported[idx] = ok
        self.support = {"observed_share": float(same.mean()), "supported_share": float(supported.mean())}
        return out

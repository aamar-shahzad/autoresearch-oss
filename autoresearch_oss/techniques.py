"""Techniques asset class: codified post-training methods, hyperparameter priors,
and a selector that matches problem traits to methods (the paper's
'Technique Selector' / 'Problem Fit')."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from .mixtures import MixtureSpec


@dataclass
class Technique:
    name: str
    backend: str          # "tabular" (CPU demo) or "llm" (TRL)
    analog: str           # which paper technique it stands in for
    uses_failures: bool
    hp_priors: dict
    description: str


LIBRARY = {
    # --- runnable on CPU, used by the demo / tests -------------------------
    "imitate": Technique("imitate", "tabular", "SFT", False,
        {"max_depth": [4, 8], "min_samples_leaf": [30, 80], "max_iter": [100, 200]},
        "Fit on successful episodes only: reproduce what worked."),
    "contrast": Technique("contrast", "tabular", "DPO / reward-model", True,
        {"max_depth": [4, 8], "min_samples_leaf": [30, 80], "max_iter": [100, 200]},
        "Learn P(success | situation, action) from successes AND failures; pick the best action."),
    # --- LLM techniques (need `pip install .[llm]` + GPU) ------------------
    "sft": Technique("sft", "llm", "SFT", False,
        {"lora_r": [16, 32, 64], "lr": [2e-4, 1e-4], "epochs": [2, 3]},
        "Token-level imitation of curated trajectories."),
    "dpo": Technique("dpo", "llm", "DPO", True,
        {"lora_r": [16, 32], "lr": [5e-6, 1e-5], "beta": [0.1, 0.3], "epochs": [1, 2]},
        "Prefer the right behaviour over a near-miss (needs preference pairs)."),
    "grpo": Technique("grpo", "llm", "GRPO", True,
        {"lora_r": [16, 32], "lr": [1e-5], "group_size": [8, 16], "steps": [200, 400]},
        "Reinforce rollouts that beat the group mean against a verifiable reward."),
}


def select(traits: dict, backend: str) -> list:
    """traits: single_correct_trace, has_failures, verifiable_reward."""
    names = []
    if traits.get("single_correct_trace", True):
        names.append("imitate" if backend == "tabular" else "sft")
    if traits.get("has_failures", False):
        names.append("contrast" if backend == "tabular" else "dpo")
    if backend == "llm" and traits.get("verifiable_reward", False):
        names.append("grpo")
    return names


@dataclass
class Recipe:
    """One base model + one mixture + one technique + hyperparameters.
    Written to the Ontology BEFORE it runs."""
    base: str
    technique: str
    mixture: MixtureSpec
    hp: dict
    parent: str = ""

    def key(self) -> str:
        blob = json.dumps({"b": self.base, "t": self.technique, "m": self.mixture.key(), "h": self.hp},
                          sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:8]

    def label(self) -> str:
        hp = ",".join(f"{k}={v}" for k, v in sorted(self.hp.items()))
        return f"{self.base}|{self.technique}|{self.mixture.name}|{hp}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key()
        return d

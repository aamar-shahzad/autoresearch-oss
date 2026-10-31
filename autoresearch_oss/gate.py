"""Step 9: deploy behind a gate that fixes which recommendations a human must confirm."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class GatePolicy:
    safety_critical: set = field(default_factory=lambda: {"replace_bearing", "restart_no_check"})
    min_confidence: float = 0.6   # below this the model escalates instead of recommending


class Deployment:
    def __init__(self, model, model_id: str, policy: GatePolicy | None = None):
        self.model, self.model_id, self.policy = model, model_id, policy or GatePolicy()

    def recommend(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.model.predict(df).copy()
        p["escalate"] = p["confidence"] < self.policy.min_confidence
        p["needs_approval"] = p["escalate"] | p["action"].isin(self.policy.safety_critical)
        return p

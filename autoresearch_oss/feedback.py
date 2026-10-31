"""Step 10: live operations become new episodes. Overrides and failures carry the most signal, so
they are tagged here and up-weighted later by the mixture builder."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tag_status(logs: pd.DataFrame) -> pd.DataFrame:
    """Turn real production logs into learnable episodes. Needs: model_rec (what the model
    recommended), action (what was actually done), outcome_success (measured later)."""
    d = logs.copy()
    d["status"] = np.where(d["action"] != d["model_rec"], "overridden",
                           np.where(d["outcome_success"], "accepted", "failed"))
    d["source"] = "live"
    return d

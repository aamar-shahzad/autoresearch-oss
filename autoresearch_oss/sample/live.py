"""Step 10: live operations become new episodes. Overrides and failures carry the
most signal, so they are tagged and later up-weighted by the mixture builder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cnc import ACTIONS, correct_action, gen_episodes


def simulate_live(dep, outcome_fn, n: int, seed: int, family_probs: dict | None = None,
                  engineer_skill: float = 0.9, approve_rate: float = 0.8) -> pd.DataFrame:
    """Simulate production use: new machines, possibly shifted fault mix, humans in the loop."""
    rng = np.random.default_rng(seed + 777)
    df = gen_episodes(n, seed=seed, family_probs=family_probs, assets=(300, 400),
                      source="live", id_prefix="L")
    rec = dep.recommend(df)
    corr = correct_action(df)
    final = rec["action"].to_numpy().copy()
    # an engineer reviews escalations and approval-gated actions
    review = rec["needs_approval"].to_numpy()
    rejects = review & (rng.random(n) > approve_rate)
    needs_human = rec["escalate"].to_numpy() | rejects
    wrong = np.array(ACTIONS)[rng.integers(0, len(ACTIONS), n)]
    human = np.where(rng.random(n) < engineer_skill, corr, wrong)
    final = np.where(needs_human, human, final)
    ok = np.asarray(outcome_fn(df, final))
    out = df.copy()
    out["model_rec"] = rec["action"].to_numpy()
    out["action"] = final
    out["outcome_success"] = ok
    out["flow_after"] = np.where(ok, 10.0, df["flow"])
    out["temp_after"] = np.where(ok, 72.0, df["temp"])
    out["status"] = np.where(final != out["model_rec"], "overridden", np.where(ok, "accepted", "failed"))
    return out

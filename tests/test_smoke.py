import numpy as np
import pandas as pd
import pytest

from autoresearch_oss.backends import TabularBackend
from autoresearch_oss.backends import llm
from autoresearch_oss.evaluate import check_thresholds, evaluate
from autoresearch_oss.loop import AutoResearch, Budget
from autoresearch_oss.mixtures import MixtureSpec, build_mixture
from autoresearch_oss.ontology import Registry, Workflow
from autoresearch_oss.quality import clean
from autoresearch_oss.splits import split_by_group
from autoresearch_oss.sample.cnc import (ACTIONS, WORKFLOW, correct_action, gen_episodes, simulate_outcome,
                                    synth_from_procedures)


def test_leakage_guard():
    bad = Workflow("x", "", ["flow", "outcome_success"], ["outcome_success"], ACTIONS, [], "fault_family", {})
    with pytest.raises(ValueError):
        clean(gen_episodes(50), bad)


def test_clean_drops_bad_rows():
    d = gen_episodes(100, seed=1)
    d.loc[0, "flow"] = np.nan
    dup = d.iloc[[5]].copy(); dup["outcome_success"] = ~dup["outcome_success"]
    out, rep = clean(pd.concat([d, dup]), WORKFLOW)
    assert rep["incomplete"] == 1 and rep["contradictory"] == 1
    assert out["episode_id"].is_unique


def test_split_has_no_asset_overlap():
    s = split_by_group(gen_episodes(3000, seed=2))
    assert not set(s["train"].asset) & set(s["eval"].asset)
    assert sum(len(v) for v in s.values()) == 3000


def test_twin_scores_counterfactuals():
    d = gen_episodes(200, seed=3)
    assert simulate_outcome(d, correct_action(d)).all()


def test_mixture_synthetic_share():
    d = gen_episodes(3000, seed=4)
    m = build_mixture(d, MixtureSpec("m", synth_frac=0.3, synth_families=("pump",)), synth_from_procedures)
    assert abs((m.source == "synthetic").mean() - 0.3) < 0.02


def test_dpo_pairs_and_reward():
    d = gen_episodes(2000, seed=5)
    pairs = llm.to_dpo_pairs(d, WORKFLOW)
    assert pairs and all(p["chosen"] != p["rejected"] for p in pairs)
    r = llm.make_reward_fn(simulate_outcome, WORKFLOW)
    row = d.iloc[0]
    good = " " + correct_action(d.iloc[:1])[0]
    out = r([""], [good], **{k: [row[k]] for k in WORKFLOW.inputs})
    assert out == [1.0]
    p = llm.closed_set_probs(np.array([[0.0, -50.0]]))
    assert p[0, 0] > 0.99


def test_loop_discovers_rare_slice_and_promotes(tmp_path):
    hist = gen_episodes(5000, seed=0)
    reg = Registry()
    ar = AutoResearch(reg, WORKFLOW, hist, simulate_outcome, TabularBackend(), synth_from_procedures,
                      budget=Budget(max_compute=900), artifacts_dir=str(tmp_path), log=lambda *_: None)
    s = ar.run()
    assert s.promoted_model_id, "loop should clear thresholds within budget"
    mdl = reg.get(s.promoted_model_id)
    assert check_thresholds(mdl["payload"]["holdout_metrics"], WORKFLOW.thresholds)
    kinds = {a["kind"] for a in reg.lineage(s.promoted_model_id)}
    assert {"workflow", "evalset", "recipe", "experiment", "dataset"} <= kinds
    assert reg.get(ar.eval_id)["payload"]["touches"] >= 1

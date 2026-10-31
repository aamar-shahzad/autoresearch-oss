"""AutoResearch: the search loop (steps 2-8 + promotion of step 9).

  prepare  -> clean, split (held-back eval set is frozen), register datasets
  probe    -> cheap probes eliminate weak directions before expensive runs
  iterate  -> train survivors, evaluate on VALIDATION, name failure modes,
              mutate the mixture/synthetics/hyperparameters, repeat
  confirm  -> the held-back EVAL set is touched only by candidates that already
              clear thresholds on validation (limits overfitting to the eval set)
  promote  -> cheapest candidate clearing every threshold on both; weights saved

Everything -- recipes before they run, experiments after, cost, lineage -- is
written to the Registry, so each run is readable by the next one."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace

import pandas as pd

from .evaluate import check_thresholds, evaluate, utility
from .mixtures import MixtureSpec, build_mixture
from .quality import clean
from .splits import split_by_group
from .techniques import LIBRARY, Recipe, select


def _mutate_hp(hp: dict, level: int) -> dict:
    """Make the model more able to fit rare patterns. level 1 = mild, 2 = stronger."""
    h = dict(hp)
    if "min_samples_leaf" in h:
        h["min_samples_leaf"] = max(5, h["min_samples_leaf"] // (2 * level))
    if "lora_r" in h:
        h["lora_r"] = min(128, h["lora_r"] * 2 * level)
    return h


@dataclass
class Budget:
    max_compute: float = 900.0    # abstract compute units across the whole run
    max_rounds: int = 4
    probe_frac: float = 0.25
    keep_frac: float = 0.5


@dataclass
class Result:
    recipe: Recipe
    exp_id: str
    metrics: dict
    failure_modes: list
    cost: float
    util: float
    passed: dict
    model: object = None


@dataclass
class Summary:
    promoted_model_id: str | None
    promoted: Result | None
    holdout: dict | None
    best: Result | None
    spent: float
    experiments: int


class AutoResearch:
    def __init__(self, registry, workflow, episodes, outcome_fn, backend, synth_fn,
                 techniques=None, budget: Budget | None = None, seed: int = 0,
                 artifacts_dir: str = "artifacts", log=print, generic_synth: bool = False,
                 unit_cost_usd: float = 0.0):
        self.reg, self.wf, self.episodes = registry, workflow, episodes
        self.outcome_fn, self.backend, self.synth_fn = outcome_fn, backend, synth_fn
        self.budget, self.seed, self.log = budget or Budget(), seed, log
        self.artifacts_dir, self.generic_synth = artifacts_dir, generic_synth
        self.unit_cost_usd = unit_cost_usd
        os.makedirs(artifacts_dir, exist_ok=True)
        traits = {"single_correct_trace": True, "has_failures": True,
                  "verifiable_reward": bool(getattr(outcome_fn, "verifiable", False))}
        self.techniques = techniques or select(traits, backend.name)
        self.splits = None
        self.wf_id = registry.put("workflow", workflow.name, workflow.__dict__)
        self._mix_cache, self._ds_ids = {}, {}
        self.prior_winners: list = []   # warm start: later runs read earlier outcomes

    # ---- step 3/4: data ------------------------------------------------------
    def prepare(self, extra_episodes: pd.DataFrame | None = None):
        if self.splits is None:
            d, rep = clean(self.episodes, self.wf)
            self.splits = split_by_group(d, self.wf.group_by, seed=self.seed)
            if self.wf.task_type == "generation":      # references exist only for good responses
                for k in ("val", "eval"):
                    self.splits[k] = self.splits[k][self.splits[k]["outcome_success"]].reset_index(drop=True)
            self._fit_hooks(self.splits["train"])
            self._add_stress_episodes()
            self.eval_id = self.reg.put("evalset", self.wf.name, {
                "rows": len(self.splits["eval"]), "touches": 0, "cleaning": rep,
                "split": {k: len(v) for k, v in self.splits.items()},
                "split_by": "asset (no machine appears in two splits)",
                "stress_added": self.stress_added}, [self.wf_id])
            self.base_ds_id = self.reg.put("dataset", self.wf.name, {
                "rows": len(self.splits["train"]), "validation_rows": len(self.splits["val"])},
                [self.wf_id])
        train = self.splits["train"]
        self.train_ds_id = self.base_ds_id
        if extra_episodes is not None and len(extra_episodes):
            ex, _ = clean(extra_episodes, self.wf)
            train = pd.concat([train, ex], ignore_index=True)
            self.train_ds_id = self.reg.put("episodes", self.wf.name + "-live", {
                "rows": len(ex), "status_counts": ex["status"].value_counts().to_dict()},
                [self.base_ds_id])
        self.train = train
        if extra_episodes is not None:
            self._fit_hooks(train)
        self._mix_cache.clear()

    def _fit_hooks(self, train: pd.DataFrame):
        """Outcome estimators / synthesizers that learn from data are fitted on TRAIN only."""
        real = train[train.get("source", pd.Series("", index=train.index)) != "synthetic"]
        for h in (self.outcome_fn, self.synth_fn):
            if hasattr(h, "fit"):
                h.fit(real, self.wf)

    def _add_stress_episodes(self, need: int = 30):
        """Rare slices may be absent from the held-back sets -> blind spot. Top them up with
        twin-scored synthetic episodes (separate seeds from training synthetics; flagged)."""
        self.stress_added = {}
        for split, off in (("val", 50_000), ("eval", 60_000)):
            d, add = self.splits[split], []
            for f in self.wf.slice_values:
                have = int((d[self.wf.slice_by] == f).sum())
                if have < need:
                    syn = self.synth_fn(f, need - have, off + len(add))
                    if len(syn):
                        add.append(syn)
                        self.stress_added[f"{split}/{f}"] = len(syn)
            if add:
                self.splits[split] = pd.concat([d] + add, ignore_index=True)

    def _mixture(self, spec: MixtureSpec) -> pd.DataFrame:
        k = spec.key()
        if k not in self._mix_cache:
            self._mix_cache[k] = build_mixture(self.train, spec, self.synth_fn, self.seed, self.wf.slice_by)
            self._ds_ids[k] = self.reg.put("dataset", f"mixture-{spec.name}", {
                "rows": len(self._mix_cache[k]), "spec": spec.__dict__,
                "synthetic_rows": int((self._mix_cache[k]["source"] == "synthetic").sum())},
                [self.train_ds_id])
        return self._mix_cache[k]

    # ---- one experiment ---------------------------------------------------------
    def _est_cost(self, recipe: Recipe, frac: float) -> float:
        return len(self._mixture(recipe.mixture)) * frac * recipe.hp.get("max_iter", 100) / 1e4

    def fit_eval(self, recipe: Recipe, frac: float = 1.0, stage: str = "full") -> Result | None:
        if self.spent + self._est_cost(recipe, frac) > self.budget.max_compute:
            self.log(f"  budget exhausted, skipping {recipe.label()}")
            return None
        mix = self._mixture(recipe.mixture)
        data = mix.sample(frac=frac, random_state=self.seed) if frac < 1 else mix
        rid = self.reg.put("recipe", recipe.key(), recipe.to_dict(), [self._ds_ids[recipe.mixture.key()]])
        exp = self.reg.put("experiment", recipe.key(), {"stage": stage, "status": "running",
                           "recipe": recipe.label()}, [rid])
        tr = self.backend.train(recipe, data, self.wf)
        self.spent += tr.compute_units
        ev = evaluate(tr.model, self.splits["val"], self.wf, self.outcome_fn)
        passed = check_thresholds(ev.metrics, self.wf.thresholds)
        self.reg.update(exp, {"status": "done", "val_metrics": ev.metrics, "slices": ev.slices,
                              "failure_modes": ev.failure_modes, "compute_units": tr.compute_units,
                              "seconds": tr.seconds, "train_rows": len(data)})
        res = Result(recipe, exp, ev.metrics, ev.failure_modes, tr.compute_units, utility(ev.metrics),
                     passed, tr.model)
        self.log(f"  [{stage:5}] {recipe.label():<62} fix={ev.metrics['fix_rate']:.3f} "
                 f"worst={ev.metrics['worst_slice']:.2f} cost={tr.compute_units:.1f}")
        return res

    # ---- candidate generation ---------------------------------------------------
    def initial_candidates(self) -> list:
        A = MixtureSpec("A-successes")
        B = MixtureSpec("B-succ+fail", use_failures=True)
        C = MixtureSpec("C-generic-synth", use_failures=True, synth_frac=0.2, synth_families=tuple(self.wf.slice_values or sorted(self.train[self.wf.slice_by].unique())))
        out = []
        for t in self.techniques:
            tech = LIBRARY[t]
            # C (blanket synthetic data) is opt-in: by default the loop must *discover*
            # which slice needs synthetic data from the failure analysis.
            for m in ((A, B, C) if self.generic_synth else (A, B)):
                if tech.uses_failures and not m.use_failures:
                    continue
                if not tech.uses_failures and m is B:
                    continue                      # identical data to A for this technique
                for base in self.backend.bases:
                    out.append(Recipe(base, t, m, {k: v[0] for k, v in tech.hp_priors.items()}))
        return out

    def propose(self, best: Result, rnd: int) -> list:
        r, m, new = best.recipe, best.recipe.mixture, []
        weak = [fm["slice"] for fm in best.failure_modes]
        if weak:
            up = dict(m.upweight)
            for f in weak:
                up[f] = max(2, up.get(f, 1) * 2)
            stem = m.name.split("~")[0]
            for frac, leaf_div in ((0.15, 1), (0.15, 2), (0.35, 2)):
                m2 = replace(m, name=f"{stem}~{'+'.join(weak)}.r{rnd}s{int(frac*100)}", upweight=up,
                             synth_families=tuple(weak), synth_frac=min(0.5, max(m.synth_frac, frac)))
                hp = _mutate_hp(r.hp, leaf_div)
                new.append(Recipe(r.base, r.technique, m2, hp, parent=best.exp_id))
        else:
            hp = _mutate_hp(r.hp, 1)
            for k in ("max_iter", "epochs", "steps"):
                if k in hp:
                    hp[k] = hp[k] * 2
            new.append(Recipe(r.base, r.technique, m, hp, parent=best.exp_id))
        return [c for c in new if c.key() not in self.seen]

    # ---- the loop ---------------------------------------------------------------
    def run(self, extra_episodes: pd.DataFrame | None = None) -> Summary:
        self.prepare(extra_episodes)
        self.spent, self.seen, results = 0.0, set(), []
        cands = self.initial_candidates()
        self.seen |= {c.key() for c in cands}
        self.log(f"AutoResearch: {len(cands)} initial recipes, budget {self.budget.max_compute} units")

        self.log("probe (cheap, subsampled) -> prune")
        probes = [x for c in cands if (x := self.fit_eval(c, self.budget.probe_frac, "probe"))]
        probes.sort(key=lambda x: -x.util)
        keep = max(2, math.ceil(len(probes) * self.budget.keep_frac))
        survivors = [x.recipe for x in probes[:keep]]
        for w in self.prior_winners:                 # previous winner skips probing
            if w.key() not in {r.key() for r in survivors}:
                survivors.insert(0, w)
        if self.prior_winners:
            self.log(f"warm-started with {len(self.prior_winners)} recipe(s) from earlier runs")
        self.log(f"kept {len(survivors)}/{len(probes)} after probes")

        for rnd in range(1, self.budget.max_rounds + 1):
            self.log(f"round {rnd}: {len(survivors)} full runs")
            rr = [x for r in survivors if (x := self.fit_eval(r, 1.0, "full"))]
            results += rr
            if not results:
                break
            for x in sorted((x for x in rr if all(x.passed.values())), key=lambda x: x.cost):
                hold = self.confirm(x)
                if all(check_thresholds(hold, self.wf.thresholds).values()):
                    mid = self.promote(x, hold)
                    self.prior_winners = [x.recipe]
                    self.log(f"PROMOTED {mid} (holdout fix={hold['fix_rate']:.3f})")
                    return Summary(mid, x, hold, x, self.spent, len(results) + len(probes))
                self.log(f"  {x.recipe.label()} passed validation but failed holdout")
            best = max(results, key=lambda x: x.util)
            if best.failure_modes:
                self.log("  failure modes: " + "; ".join(
                    f"{f['slice']} fix={f['fix_rate']:.2f} (n={f['n']}, wrongly '{f['most_common_wrong_action']}')"
                    for f in best.failure_modes))
            survivors = self.propose(best, rnd)
            self.seen |= {c.key() for c in survivors}
            if not survivors:
                break
        best = max(results, key=lambda x: x.util) if results else None
        self.log("no candidate cleared every threshold within budget")
        return Summary(None, None, None, best, self.spent, len(results) + len(probes))

    def confirm(self, res: Result) -> dict:
        ev = evaluate(res.model, self.splits["eval"], self.wf, self.outcome_fn)
        e = self.reg.get(self.eval_id)
        self.reg.update(self.eval_id, {"touches": e["payload"]["touches"] + 1})
        self.reg.update(res.exp_id, {"holdout_metrics": ev.metrics})
        return ev.metrics

    def promote(self, res: Result, hold: dict) -> str:
        ext = ".joblib" if self.backend.name == "tabular" else ""
        path = os.path.join(self.artifacts_dir, f"{res.recipe.key()}{ext}")
        res.model.save(path)
        exp = self.reg.get(res.exp_id)
        return self.reg.put("model", self.wf.name, {
            "weights": path, "recipe": res.recipe.label(), "val_metrics": res.metrics,
            "holdout_metrics": hold, "compute_units": res.cost, "total_run_cost": self.spent,
            "total_run_usd": self.spent * self.unit_cost_usd, "stage": "staging",
            "features": list(self.wf.inputs), "actions": list(self.wf.action_space),
            "base": res.recipe.base, "technique": res.recipe.technique},
            [res.exp_id, self.eval_id] + exp["parents"])

    # ---- reporting --------------------------------------------------------------
    def table(self) -> pd.DataFrame:
        rows = []
        for e in self.reg.list("experiment"):
            p = e["payload"]
            if "val_metrics" in p:
                rows.append({"experiment": e["id"], "stage": p["stage"], "recipe": p["recipe"],
                             "fix": p["val_metrics"]["fix_rate"], "worst_slice": p["val_metrics"]["worst_slice"],
                             "unsafe": p["val_metrics"]["unsafe_rate"], "cost": p["compute_units"]})
        return pd.DataFrame(rows)

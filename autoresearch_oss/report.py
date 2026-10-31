"""Run report + model card + dollar costs, generated from the registry alone."""
from __future__ import annotations

import pandas as pd

from .lifecycle import production


def experiments_frame(reg, usd_per_unit: float = 0.0) -> pd.DataFrame:
    rows = []
    for e in reg.list("experiment"):
        p = e["payload"]
        if "val_metrics" not in p:
            continue
        m = p["val_metrics"]
        rows.append({"experiment": e["id"], "stage": p["stage"], "recipe": p["recipe"].replace("|", " / "),
                     "fix": m["fix_rate"], "worst_slice": m["worst_slice"], "unsafe": m["unsafe_rate"],
                     "units": p["compute_units"], "usd": p["compute_units"] * usd_per_unit,
                     "holdout_fix": (p.get("holdout_metrics") or {}).get("fix_rate")})
    return pd.DataFrame(rows)


def generate(reg, workflow_name: str, usd_per_unit: float = 0.0, path: str | None = None) -> str:
    wf = next((w for w in reversed(reg.list("workflow")) if w["name"] == workflow_name), None)
    models = [m for m in reg.list("model") if m["name"] == workflow_name]
    ex = experiments_frame(reg, usd_per_unit)
    L = [f"# AutoResearch report: {workflow_name}", ""]
    if wf:
        L += [wf["payload"]["description"], "", "**Thresholds:** " +
              ", ".join(f"{k} {op} {v}" for k, (op, v) in wf["payload"]["thresholds"].items()), ""]
    cur = production(reg, workflow_name) or (models[-1] if models else None)
    L += ["## Model card", ""]
    if cur:
        p = cur["payload"]
        L += [f"- **Model:** `{cur['id']}` (stage: {p.get('stage')})",
              f"- **Recipe:** `{p['recipe']}`",
              f"- **Inputs:** {', '.join(p.get('features', []))}  |  **Actions:** {', '.join(p.get('actions', []))}",
              "", "| metric | validation | holdout |", "|---|---|---|"]
        for k in ("fix_rate", "worst_slice", "unsafe_rate", "costly_error_rate", "escalation_rate"):
            L.append(f"| {k} | {p['val_metrics'].get(k, float('nan')):.3f} | {p['holdout_metrics'].get(k, float('nan')):.3f} |")
        L += ["", f"- **Training cost:** {p['compute_units']:.1f} units this model; "
              f"{p.get('total_run_cost', 0):.1f} units whole search"
              + (f" (~${p.get('total_run_usd', 0):.2f})" if usd_per_unit else ""),
              "- **Lineage:** " + " -> ".join(a["id"] for a in reg.lineage(cur["id"])[:6]), ""]
        ev = reg.list("evalset")
        if ev:
            e = ev[-1]["payload"]
            L += ["## Evaluation integrity", "",
                  f"- Holdout rows: {e['rows']}, touched {e['touches']}x (only by candidates already passing validation)",
                  f"- Split: {e['split']} by {e['split_by']}",
                  f"- Stress episodes added for under-covered slices: {e.get('stress_added') or 'none'}",
                  f"- Cleaning: {e['cleaning']}", ""]
    if len(ex):
        L += ["## Experiments", "", ex.round(3).to_markdown(index=False) if _has_tabulate() else ex.round(3).to_string(index=False), ""]
        full = ex[ex["stage"] == "full"]
        if len(full):
            base = full.iloc[0]
            best = full.sort_values("fix").iloc[-1]
            L += ["## Return on spend", "",
                  f"- First full run: fix {base['fix']:.3f}; best full run: fix {best['fix']:.3f}",
                  f"- Total spend: {ex['units'].sum():.1f} units" + (f" (~${ex['usd'].sum():.2f})" if usd_per_unit else ""), ""]
    modes = []
    for e in reg.list("experiment"):
        for f in e["payload"].get("failure_modes", []):
            modes.append(f"{f['slice']} (fix {f['fix_rate']:.2f}, n={f['n']}, wrongly '{f['most_common_wrong_action']}')")
    if modes:
        L += ["## Failure modes found along the way", ""] + [f"- {m}" for m in dict.fromkeys(modes)] + [""]
    L += ["## Caveats", "",
          "- Scores are only as good as the outcome function (twin, or offline replay estimate).",
          "- Synthetic episodes are flagged `source=synthetic` and never appear in the holdout."]
    out = "\n".join(L)
    if path:
        open(path, "w").write(out)
    return out


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False

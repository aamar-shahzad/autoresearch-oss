"""Study layer: read prior experiments from the registry and propose new recipes."""

from __future__ import annotations

import json
import re
from dataclasses import replace

from .mixtures import MixtureSpec
from .techniques import LIBRARY, Recipe


def experiment_digest(reg, workflow_name: str, limit: int = 25) -> dict:
    exps = []
    for e in reg.list("experiment"):
        p = e["payload"]
        if p.get("val_metrics") is None:
            continue
        exps.append(
            {
                "id": e["id"],
                "recipe": p.get("recipe"),
                "stage": p.get("stage"),
                "metrics": p.get("val_metrics"),
                "failure_modes": p.get("failure_modes") or [],
                "holdout": p.get("holdout_metrics"),
            }
        )
    exps.sort(key=lambda x: x["id"], reverse=True)
    exps = exps[:limit]
    worst_slices: dict[str, float] = {}
    for x in exps:
        for fm in x.get("failure_modes") or []:
            sl = fm.get("slice")
            if sl:
                worst_slices[sl] = min(worst_slices.get(sl, 1.0), float(fm.get("fix_rate", 0)))
    return {
        "workflow": workflow_name,
        "experiments": exps,
        "worst_slices": worst_slices,
        "count": len(exps),
    }


def _heuristic_recipes(digest: dict, backend, techniques: list, seed_recipe: Recipe | None) -> list[Recipe]:
    out: list[Recipe] = []
    weak = sorted(digest.get("worst_slices") or {}, key=lambda kv: kv[1])[:3]
    weak = [s for s, _ in weak]
    base_tech = (seed_recipe.technique if seed_recipe else techniques[0])
    base = (seed_recipe.base if seed_recipe else backend.bases[0])
    tech = LIBRARY.get(base_tech)
    if not tech:
        return out
    m0 = seed_recipe.mixture if seed_recipe else MixtureSpec("B-succ+fail", use_failures=True)
    for i, sl in enumerate(weak or ["unknown"]):
        up = dict(m0.upweight)
        up[sl] = max(3, up.get(sl, 1) * 3)
        m2 = replace(
            m0,
            name=f"study~{sl}~r{i}",
            upweight=up,
            synth_families=(sl,) if sl != "unknown" else m0.synth_families,
            synth_frac=min(0.45, max(m0.synth_frac, 0.2)),
        )
        hp = dict(seed_recipe.hp if seed_recipe else {k: v[0] for k, v in tech.hp_priors.items()})
        out.append(Recipe(base, base_tech, m2, hp, parent=digest["experiments"][0]["id"] if digest["experiments"] else ""))
    return out


def _llm_propose_recipes(
    digest: dict,
    wf,
    backend,
    techniques: list,
    model_id: str,
    max_recipes: int = 3,
) -> list[Recipe]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return []

    prompt = (
        "You are an AutoResearch study agent. Given past experiment metrics, propose JSON only: "
        '[{"technique":"imitate|contrast|sft|dpo","mixture":"A-successes|B-succ+fail",'
        '"synth_frac":0.2,"focus_slice":"filter"}]. '
        f"Workflow={wf.name} backend={backend.name} techniques={techniques}. "
        f"Digest={json.dumps(digest, default=str)[:3500]}\nJSON:"
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    ids = tok.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=256, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    recipes: list[Recipe] = []
    base = backend.bases[0]
    for item in items[:max_recipes]:
        tname = item.get("technique") or techniques[0]
        if tname not in LIBRARY or LIBRARY[tname].backend != backend.name:
            continue
        mix_name = item.get("mixture") or "B-succ+fail"
        use_fail = "fail" in mix_name or mix_name.startswith("B")
        synth_frac = float(item.get("synth_frac") or 0.0)
        sl = item.get("focus_slice")
        mspec = MixtureSpec(
            mix_name,
            use_failures=use_fail,
            synth_frac=synth_frac,
            synth_families=(sl,) if sl else tuple(wf.slice_values or []),
        )
        tech = LIBRARY[tname]
        hp = {k: v[0] for k, v in tech.hp_priors.items()}
        recipes.append(Recipe(base, tname, mspec, hp))
    return recipes


def propose_study_recipes(
    reg,
    wf,
    backend,
    techniques: list,
    *,
    use_llm: bool = False,
    llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    seed_recipe: Recipe | None = None,
) -> tuple[list[Recipe], dict]:
    digest = experiment_digest(reg, wf.name)
    recipes = _heuristic_recipes(digest, backend, techniques, seed_recipe)
    if use_llm and digest["count"]:
        recipes = _llm_propose_recipes(digest, wf, backend, techniques, llm_model) or recipes
    seen = set()
    uniq = []
    for r in recipes:
        if r.key() not in seen:
            seen.add(r.key())
            uniq.append(r)
    return uniq, digest

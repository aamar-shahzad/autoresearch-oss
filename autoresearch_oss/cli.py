"""autoresearch -- command line.

doctor            detect this machine's hardware and ML stack
models ...        list / info / search / pull / choose / local / verify  open models
init              scaffold a project from an industry template
validate          check YOUR data before spending compute
train             run the search (preflight-checks the chosen model against your hardware)
predict / serve   batch inference / HTTP service with the human gate + feedback logging
ingest-live       turn the service log into episodes for the next run
report | models-registry (models-reg) | promote | rollback | lineage | export-dataset | monitor | plugins
platform api      registry HTTP API + web dashboard (pip install autoresearch-oss[platform])
entity sync       register shared entities from episode columns (entity_links in config)
export-ontology   dump entity types and entities from registry
quickstart        end-to-end tour on built-in SAMPLE data (no data or GPU needed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from . import builtins  # noqa: F401
from .config import build, load_config
from .connectors import load_episodes
from .datasets import ScrubPolicy, export_dataset
from .hardware import detect, probe_torch
from .lifecycle import load_model, production, promote_to_production, rollback
from .models import load_catalog, plan, recommend
from .models.catalog import filter_entries, find
from .models.chooser import choose, render
from .models.hub import DEFAULT_ROOT, HubClient, HubError, local_models, verify_local
from .monitor import drift_report, performance_report, should_retrain
from .ontology import Registry, Workflow
from .plugins import available, load_entry_points
from .project import init_project, list_templates, validate_data
from .report import generate


def _wf(cfg) -> Workflow:
    return Workflow.from_dict(cfg["workflow"])


# ------------------------------------------------------------------ doctor / models ----
def cmd_doctor(a):
    hw = detect(a.disk_path)
    if a.json:
        print(json.dumps(hw.to_dict(), indent=1))
        return
    print(hw.summary())
    if a.probe:
        print("torch probe:", probe_torch())
    errs = load_entry_points()
    pl = available()
    print(
        "plugins       " + "; ".join(f"{k}: {', '.join(v)}" for k, v in pl.items() if v)
    )
    for e in errs:
        print("plugin error:", e)


def _entries(a):
    es = load_catalog()
    lic = (
        None
        if a.license == "any"
        else ("permissive" if a.license == "permissive" else ["permissive", "custom"])
    )
    return filter_entries(
        es,
        license_class=lic,
        family=a.family,
        max_params=a.max_params,
        tag=a.tag,
        query=a.query,
    )


def cmd_models(a):
    hw = detect(a.disk_path)
    if a.mcmd == "list":
        plans = recommend(hw, _entries(a), a.seq, a.batch, only_trainable=not a.all)
        print(
            f"Hardware: {hw.accelerator}"
            + (
                f", {hw.gpus[0].name} {hw.device_memory_gb:.0f} GB"
                if hw.gpus
                else f", {hw.ram_gb:.0f} GB RAM"
            )
            + f", {hw.disk_free_gb:.0f} GB disk free   (seq {a.seq}, batch {a.batch})\n"
        )
        print(
            render(plans, limit=a.limit, show_all=a.all)
            if plans
            else "No catalog model is trainable on this machine. Try --all."
        )
        print(
            "\nEstimates are conservative (+/-25%). 'need/have' is GB. See `autoresearch models info <name>` for details."
        )
    elif a.mcmd == "info":
        e = find(load_catalog(), a.repo)
        if e is None:
            print("not in the catalog; reading the Hub only")
        else:
            p = plan(e, hw, a.seq, a.batch)
            print(
                f"{e.id}  {e.params_b:g}B  license={e.license_id} ({e.commercial_ok} for commercial use)  gated={e.gated}"
            )
            if e.notes:
                print("note:", e.notes)
            print(
                f"Target: {p.where}   disk needed ~{p.disk_needed_gb} GB ({'ok' if p.disk_ok else 'NOT ENOUGH'})"
            )
            for m in p.modes:
                print(
                    f"  {m.mode:<6} need {m.required_gb:>6.1f} GB / have {m.available_gb:>6.1f} GB  -> {m.verdict}"
                    + (f"   ({'; '.join(m.caveats)})" if m.caveats else "")
                )
            print(
                f"  inference: {p.inference.mode} {p.inference.required_gb} GB -> {p.inference.verdict}"
            )
            if p.missing_libs:
                print("  missing packages:", ", ".join(p.missing_libs))
        try:
            i = HubClient().info(e.id if e else a.repo)
            print(
                f"\nLive Hub: license={i.license_id} gated={i.gated} size={i.size_gb} GB revision={i.sha[:10]}"
            )
        except HubError as ex:
            print(f"\n(live Hub info unavailable: {ex})")
    elif a.mcmd == "search":
        try:
            for r in HubClient().search(a.query, a.limit):
                print(f"{r['repo']:<55} downloads={r['downloads']}  gated={r['gated']}")
        except HubError as ex:
            print("search failed:", ex)
            return 1
    elif a.mcmd == "local":
        ms = local_models(a.root)
        for m in ms:
            print(
                f"{m['repo']:<45} {m['bytes'] / 1024**3:>6.1f} GB  rev={m['revision'][:10]}  license={m['license']}  {m['path']}"
            )
        if not ms:
            print(f"no models in {a.root}")
    elif a.mcmd == "verify":
        m = next((x for x in local_models(a.root) if x["repo"] == a.repo), None)
        if not m:
            print("not downloaded")
            return 1
        bad = verify_local(m)
        print("intact" if not bad else "PROBLEMS: " + "; ".join(bad))
        return 0 if not bad else 1
    elif a.mcmd in ("pull", "choose"):
        repo = a.repo if a.mcmd == "pull" else None
        if a.mcmd == "choose":
            plans = recommend(hw, _entries(a), a.seq, a.batch, only_trainable=not a.all)
            if not plans:
                print(
                    "nothing in the catalog fits this machine (try --all or a smaller --seq)"
                )
                return 1
            if a.select:
                sel = plans[a.select - 1] if a.select <= len(plans) else None
            elif not sys.stdin.isatty():
                print("not interactive; pass --select N")
                return 1
            else:
                sel = choose(plans, limit=a.limit)
            if sel is None:
                print("cancelled")
                return 1
            repo = sel.entry.id
            print(f"\nSelected {repo}")
            if a.config:
                cfg = load_config(a.config)
                cfg.setdefault("backend", {"type": "llm"}).update(
                    {"type": "llm", "model": {"id": repo}}
                )
                json.dump(cfg, open(a.config, "w"), indent=1)
                print(f"wrote backend.model.id to {a.config}")
            if not a.pull:
                print(f"Download with: autoresearch models pull {repo}")
                return 0
        confirm = (
            (
                lambda i: (
                    input(
                        f"{i.repo} license '{i.license_id}' has conditions. Accept? [y/N] "
                    ).lower()
                    == "y"
                )
            )
            if sys.stdin.isatty()
            else None
        )
        try:
            m = HubClient().download(
                repo, a.root, accept_license=a.accept_license, confirm=confirm
            )
        except HubError as ex:
            print("download refused:", ex)
            return 1
        print(
            f"downloaded {repo} -> {m['path']} ({m['bytes'] / 1024**3:.1f} GB, revision {m['revision'][:10]})"
        )


# ------------------------------------------------------------------- project -----------
def cmd_init(a):
    if a.template == "list":
        for k, v in list_templates().items():
            print(f"{k:<26} {v}")
        return
    for f in init_project(a.path, a.template):
        print("wrote", f)
    print(
        f"\nNext: put your data in {a.path}/data/episodes.csv, then\n  autoresearch validate --config {a.path}/autoresearch.json"
    )


def cmd_validate(a):
    r = validate_data(load_config(a.config))
    print(json.dumps(r, indent=1, default=str))
    return 1 if r["problems"] else 0


def _preflight_llm(cfg, force: bool) -> bool:
    b = cfg.get("backend", {})
    if b.get("type") != "llm":
        return True
    hw = detect()
    model = (
        (b.get("model") or {}).get("id")
        or (b.get("model") or {}).get("path")
        or (b.get("bases") or [""])[0]
    )
    e = find(load_catalog(), model.replace("__", "/")) or find(
        load_catalog(), os.path.basename(model)
    )
    if e is None:
        print(
            f"preflight: {model} is not in the catalog; cannot estimate memory (continuing)"
        )
        return True
    p = plan(e, hw, b.get("seq_len", 2048), b.get("batch_size", 1))
    print(f"preflight: {e.short} on {p.where}: best mode = {p.best or 'none'}")
    for m in p.modes[:2]:
        print(
            f"  {m.mode}: need {m.required_gb} / have {m.available_gb} GB -> {m.verdict} {m.caveats if m.caveats else ''}"
        )
    if p.missing_libs:
        print(
            "  missing packages:",
            ", ".join(p.missing_libs),
            '  -> pip install "autoresearch-oss[llm]"',
        )
    ok = p.trainable and not p.missing_libs
    if not ok and not force:
        print(
            "preflight FAILED (use --force to run anyway). Try a smaller model: autoresearch models list"
        )
        return False
    return True


def cmd_train(a):
    cfg = load_config(a.config)
    if not _preflight_llm(cfg, a.force):
        return 1
    os.makedirs(a.out, exist_ok=True)
    ar, _, wf = build(cfg, a.db, a.out)
    if a.live_log:
        from .serve import load_live_log

        live = load_live_log(a.live_log, wf)
        s = ar.run(extra_episodes=live)
    else:
        s = ar.run()
    generate(ar.reg, wf.name, cfg.get("unit_cost_usd", 0.0), f"{a.out}/report.md")
    if s.promoted_model_id:
        if a.promote:
            promote_to_production(ar.reg, s.promoted_model_id)
        print(
            f"\nPromoted {s.promoted_model_id} ({'production' if a.promote else 'staging'}); report: {a.out}/report.md"
        )
        return 0
    print(
        f"\nNothing cleared the thresholds. Named failure modes and the best candidate: {a.out}/report.md"
    )
    return 2


def _load_prod(a, cfg):
    reg = Registry(a.db)
    wf = _wf(cfg)
    m = production(reg, wf.name)
    if not m:
        raise SystemExit(
            "no production model; run `train --promote` or `promote --model ID`"
        )
    return reg, wf, m, load_model(reg, m["id"], wf)


def cmd_predict(a):
    cfg = load_config(a.config)
    reg, wf, m, model = _load_prod(a, cfg)
    df = pd.read_csv(a.input)
    out = model.predict(df)
    pd.concat([df, out], axis=1).to_csv(a.out, index=False)
    print(f"{len(df)} predictions from {m['id']} -> {a.out}")


def cmd_serve(a):
    from .gate import GatePolicy
    from .serve import Service

    cfg = load_config(a.config)
    reg, wf, m, model = _load_prod(a, cfg)
    pol = GatePolicy(
        set(cfg.get("gate", {}).get("safety_critical", [])),
        cfg.get("gate", {}).get("min_confidence", 0.6),
    )
    key = a.api_key or os.environ.get("AUTORESEARCH_API_KEY")
    srv = Service(model, m["id"], wf, pol, a.log, key).make_server(a.host, a.port)
    print(
        f"serving {m['id']} on http://{a.host}:{a.port}   log -> {a.log}   auth: {'bearer token' if key else 'NONE (local only!)'}"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_ingest(a):
    from .serve import load_live_log

    wf = _wf(load_config(a.config))
    df = load_live_log(a.log, wf)
    df.to_csv(a.out, index=False)
    print(
        f"{len(df)} live episodes -> {a.out}  statuses: {df['status'].value_counts().to_dict() if len(df) else {}}"
    )


# -------------------------------------------------------- lifecycle / governance ---------
def cmd_export(a):
    cfg = load_config(a.config)
    wf = _wf(cfg)
    r = export_dataset(
        Registry(a.db),
        load_episodes(cfg["data"], wf),
        wf,
        a.out,
        a.name,
        ScrubPolicy(**cfg.get("export", {})),
    )
    print(json.dumps({k: r[k] for k in ("id", "path")}, indent=1))


def cmd_eval(a):
    from .config import build_parts
    from .eval_run import run_eval

    cfg = load_config(a.config)
    wf, outcome, _, _ = build_parts(cfg)
    reg = Registry(a.db, tenant_id=cfg.get("tenant_id", "default"))
    df = load_episodes(cfg["data"], wf)
    if a.split:
        from .quality import clean
        from .splits import split_by_group

        df, _ = clean(df, wf)
        df = split_by_group(df, wf.group_by, seed=cfg.get("seed", 0))[a.split]
    r = run_eval(reg, wf, outcome, df, model_id=a.model)
    print(
        json.dumps(
            {
                k: r[k]
                for k in ("id", "model_id", "n_cases", "metrics", "passed_thresholds")
            },
            indent=1,
            default=str,
        )
    )
    return 0 if all(r["passed_thresholds"].values()) else 2


def cmd_about(_a):
    print(
        "\n".join(
            [
                "autoresearch-oss — decision-model factory + governance loop",
                "",
                "IS:     workflow search → gate → serve → learn (registry lineage you own)",
                "IS:     ontology-lite (entity_links, Case sync) — docs/ontology-lite.md",
                "IS NOT: full enterprise ontology platform / Foundry parity",
                "IS NOT: LLM chat or RAG app host — serve = workflow predict + feedback",
                "",
                "Backend: tabular = default, proven on CPU (maintenance template)",
                "         llm     = same loop via TRL; examples/maintenance_llm.json",
                "",
                "Docs: docs/product-positioning.md",
            ]
        )
    )
    return 0


def cmd_monitor(a):
    cfg = load_config(a.config)
    wf = _wf(cfg)
    ref = load_episodes(cfg["data"], wf)
    live = load_episodes({**cfg["data"], "path": a.live, "format": None}, wf)
    d = drift_report(ref, live, wf)
    p = (
        performance_report(live, wf)
        if "status" in live and live["status"].nunique() > 1
        else {"override_alert": False, "failure_alert": False}
    )
    again, why = should_retrain(d, p)
    print(
        json.dumps(
            {"drift": d, "performance": p, "retrain": again, "reasons": why},
            indent=1,
            default=str,
        )
    )


def cmd_registry(a):
    reg = Registry(a.db)
    if a.cmd == "report":
        print(generate(reg, a.workflow, a.usd_per_unit, a.out))
    elif a.cmd == "registry-models":
        for x in reg.list("model"):
            p = x["payload"]
            print(
                f"{x['id']:<40} {p.get('stage', '?'):<12} holdout_fix={p['holdout_metrics']['fix_rate']:.3f}  {p['recipe']}"
            )
    elif a.cmd == "promote":
        print("production:", promote_to_production(reg, a.model))
    elif a.cmd == "rollback":
        print("rolled back to:", rollback(reg, a.workflow))
    elif a.cmd == "lineage":
        for x in reg.lineage(a.id):
            print(x["id"])


def cmd_plugins(a):
    for e in load_entry_points():
        print("plugin load error:", e)
    for k, v in available().items():
        print(f"{k:<10} {', '.join(v) or '-'}")


def cmd_quickstart(a):
    from .quickstart import quickstart

    quickstart(a.n, a.db, a.out, a.seed, a.live_n)


def cmd_platform(a):
    from .platform.api import run_api

    run_api(a.db, a.config, a.host, a.port, a.api_key, tenant_id=a.tenant_id)


def cmd_smoke_llm(a):
    from .smoke_llm import run_smoke

    out = run_smoke(a.model, a.steps)
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


def cmd_entity(a):
    from .entities import (
        export_ontology,
        load_entity_links,
        sync_entities_from_episodes,
    )
    from .episode_materialize import (
        load_episode_field_map,
        materialize_episodes_from_entities,
        sync_case_entities_from_episodes,
    )
    from .ontology import Registry, Workflow

    reg = Registry(a.db, tenant_id=getattr(a, "tenant_id", "default"))
    if a.ecmd == "sync":
        cfg = load_config(a.config)
        wf = Workflow.from_dict(cfg["workflow"])
        links = load_entity_links(cfg)
        if not links:
            print("no entity_links in config")
            return 1
        df = load_episodes(cfg["data"], wf)
        stats = sync_entities_from_episodes(reg, df, links)
        if getattr(a, "cases", False):
            stats["cases"] = sync_case_entities_from_episodes(reg, df, wf)
        print(json.dumps(stats, indent=1))
    elif a.ecmd == "materialize":
        cfg = load_config(a.config)
        wf = Workflow.from_dict(cfg["workflow"])
        types = [t.strip() for t in a.types.split(",") if t.strip()]
        df = materialize_episodes_from_entities(
            reg, wf, types, load_episode_field_map(cfg, wf)
        )
        if df.empty:
            print("no rows materialized")
            return 1
        df.to_csv(a.out, index=False)
        print(json.dumps({"rows": len(df), "path": a.out}, indent=1))
    elif a.ecmd == "export":
        text = json.dumps(export_ontology(reg), indent=1)
        if a.out == "-":
            print(text)
        else:
            open(a.out, "w").write(text)
            print("wrote", a.out)


def cmd_export_ontology(a):
    a.ecmd = "export"
    a.out = getattr(a, "out", "-")
    return cmd_entity(a)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="autoresearch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    db = lambda p: p.add_argument("--db", default="registry.db")

    d = sp.add_parser("doctor", help="detect hardware + ML stack")
    d.set_defaults(fn=cmd_doctor)
    d.add_argument("--json", action="store_true")
    d.add_argument("--probe", action="store_true", help="import torch in a subprocess")
    d.add_argument("--disk-path", default=None)

    m = sp.add_parser("models", help="open models that fit this machine")
    m.set_defaults(fn=cmd_models)
    ms = m.add_subparsers(dest="mcmd", required=True)

    def mcommon(p):
        p.add_argument("--seq", type=int, default=2048)
        p.add_argument("--batch", type=int, default=1)
        p.add_argument(
            "--license",
            choices=["permissive", "commercial", "any"],
            default="commercial",
            help="permissive = Apache/MIT; commercial = also custom licenses with conditions",
        )
        p.add_argument("--family")
        p.add_argument("--max-params", type=float)
        p.add_argument("--tag")
        p.add_argument("--query")
        p.add_argument(
            "--all", action="store_true", help="include models that do not fit"
        )
        p.add_argument("--limit", type=int, default=25)
        p.add_argument("--disk-path", default=None)

    mcommon(ms.add_parser("list"))
    i = ms.add_parser("info")
    i.add_argument("repo")
    mcommon(i)
    s = ms.add_parser("search")
    s.add_argument("query")
    mcommon(s)
    ms.add_parser("local").add_argument("--root", default=DEFAULT_ROOT)
    v = ms.add_parser("verify")
    v.add_argument("repo")
    v.add_argument("--root", default=DEFAULT_ROOT)
    pu = ms.add_parser("pull")
    pu.add_argument("repo")
    pu.add_argument("--root", default=DEFAULT_ROOT)
    pu.add_argument("--accept-license", action="store_true")
    mcommon(pu)
    ch = ms.add_parser("choose")
    mcommon(ch)
    ch.add_argument("--select", type=int)
    ch.add_argument("--config")
    ch.add_argument("--pull", action="store_true")
    ch.add_argument("--root", default=DEFAULT_ROOT)
    ch.add_argument("--accept-license", action="store_true")

    n = sp.add_parser("init", help="scaffold a project")
    n.set_defaults(fn=cmd_init)
    n.add_argument("template", help="template name, or 'list'")
    n.add_argument("path", nargs="?", default=".")
    va = sp.add_parser("validate", help="check your data")
    va.set_defaults(fn=cmd_validate)
    va.add_argument("--config", required=True)
    t = sp.add_parser("train", help="run the search")
    t.set_defaults(fn=cmd_train)
    db(t)
    t.add_argument("--config", required=True)
    t.add_argument("--out", default="artifacts")
    t.add_argument("--promote", action="store_true")
    t.add_argument("--force", action="store_true")
    t.add_argument("--live-log", help="JSONL from `serve` to learn from")
    pr = sp.add_parser("predict")
    pr.set_defaults(fn=cmd_predict)
    db(pr)
    pr.add_argument("--config", required=True)
    pr.add_argument("--input", required=True)
    pr.add_argument("--out", default="predictions.csv")
    sv = sp.add_parser(
        "serve",
        help="workflow HTTP API: predict + feedback (not a chat/RAG host)",
    )
    sv.set_defaults(fn=cmd_serve)
    db(sv)
    sv.add_argument("--config", required=True)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--log", default="live.jsonl")
    sv.add_argument("--api-key")
    ig = sp.add_parser("ingest-live")
    ig.set_defaults(fn=cmd_ingest)
    ig.add_argument("--config", required=True)
    ig.add_argument("--log", required=True)
    ig.add_argument("--out", default="live.csv")
    ex = sp.add_parser("export-dataset")
    ex.set_defaults(fn=cmd_export)
    db(ex)
    ex.add_argument("--config", required=True)
    ex.add_argument("--name", required=True)
    ex.add_argument("--out", default="published")
    mo = sp.add_parser("monitor")
    mo.set_defaults(fn=cmd_monitor)
    mo.add_argument("--config", required=True)
    mo.add_argument("--live", required=True)
    ab = sp.add_parser("about", help="what this project is and is not")
    ab.set_defaults(fn=cmd_about)
    ev = sp.add_parser("eval", help="batch eval runs (cases + metrics, not chat)")
    evs = ev.add_subparsers(dest="evcmd", required=True)
    er = evs.add_parser("run", help="score model on data split")
    er.set_defaults(fn=cmd_eval)
    ev.set_defaults(fn=lambda a: print("use: autoresearch eval run ...") or 1)
    db(er)
    er.add_argument("--config", required=True)
    er.add_argument("--model", help="model id; default production")
    er.add_argument(
        "--split",
        choices=("train", "val", "eval"),
        default="eval",
        help="which split to score",
    )
    for name in ("report", "registry-models", "promote", "rollback", "lineage"):
        p = sp.add_parser(name)
        p.set_defaults(fn=cmd_registry)
        db(p)
        if name == "report":
            p.add_argument("--workflow", required=True)
            p.add_argument("--usd-per-unit", type=float, default=0.0)
            p.add_argument("--out", default="report.md")
        if name == "promote":
            p.add_argument("--model", required=True)
        if name == "rollback":
            p.add_argument("--workflow", required=True)
        if name == "lineage":
            p.add_argument("--id", required=True)
    sp.add_parser("plugins").set_defaults(fn=cmd_plugins)
    q = sp.add_parser(
        "quickstart", aliases=["demo"], help="tour on built-in SAMPLE data"
    )
    q.set_defaults(fn=cmd_quickstart)
    db(q)
    q.add_argument("--n", type=int, default=8000)
    q.add_argument("--live-n", type=int, default=3000)
    q.add_argument("--out", default="artifacts")
    q.add_argument("--seed", type=int, default=0)

    pl = sp.add_parser("platform", help="platform API + dashboard")
    pls = pl.add_subparsers(dest="pcmd", required=True)
    pa = pls.add_parser("api", help="HTTP registry API and UI")
    pa.set_defaults(fn=cmd_platform)
    db(pa)
    pa.add_argument(
        "--config", help="project autoresearch.json (monitor/predict from API)"
    )
    pa.add_argument("--host", default="127.0.0.1")
    pa.add_argument("--port", type=int, default=8090)
    pa.add_argument("--api-key")
    pa.add_argument(
        "--tenant-id", default="default", help="default tenant for registry rows"
    )

    sm = sp.add_parser(
        "smoke-llm", help="1-step TRL smoke on SmolLM2-135M (needs .[llm])"
    )
    sm.set_defaults(fn=cmd_smoke_llm)
    sm.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    sm.add_argument("--steps", type=int, default=1)

    en = sp.add_parser("entity", help="ontology-lite entities")
    en.set_defaults(fn=cmd_entity)
    db(en)
    en.add_argument("--tenant-id", default="default")
    ens = en.add_subparsers(dest="ecmd", required=True)
    es = ens.add_parser("sync", help="upsert entities from episode CSV columns")
    es.add_argument("--config", required=True)
    es.add_argument(
        "--cases", action="store_true", help="also store each row as Case entity"
    )
    em = ens.add_parser("materialize", help="build episodes CSV from Case entities")
    em.add_argument("--config", required=True)
    em.add_argument("--types", default="Case", help="comma-separated entity types")
    em.add_argument("--out", default="data/episodes_from_ontology.csv")
    ee = ens.add_parser("export", help="export entity types and entities")
    ee.add_argument("--out", default="-")

    eo = sp.add_parser(
        "export-ontology", help="export entity types and entities from registry"
    )
    eo.set_defaults(fn=cmd_export_ontology)
    db(eo)
    eo.add_argument("--out", default="-")

    a = ap.parse_args(argv)
    try:
        code = a.fn(a) or 0
        sys.stdout.flush()
    except BrokenPipeError:  # e.g. `autoresearch models list | head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 0
    sys.exit(code)

"""FastAPI platform: registry, ontology, monitor, predict proxy, static UI."""

from __future__ import annotations

import os
from pathlib import Path

from ..config import build_parts, load_config
from ..connectors import load_episodes
from ..datasets import ScrubPolicy, export_dataset
from ..entities import (
    ensure_entity_type,
    export_ontology,
    load_entity_links,
    sync_entities_from_episodes,
    upsert_entity,
)
from ..episode_materialize import (
    link_entity_to_workflow,
    load_episode_field_map,
    materialize_episodes_from_entities,
    sync_case_entities_from_episodes,
)
from ..eval_run import run_eval
from ..gate import GatePolicy
from ..governance import authorize, parse_rbac_env
from ..hardware import detect
from ..lifecycle import load_model, production
from ..monitor import drift_report, performance_report, should_retrain
from ..ontology import Workflow
from ..project import init_project, list_templates, validate_data
from ..registry_store import RegistryStore
from ..scheduler import add_schedule, list_schedules, remove_schedule, start_scheduler
from ..serve import Service
from ..studies import experiment_digest, propose_study_recipes
from .jobs import get_job, submit_train
from .schemas import (
    ConfigBody,
    EntityBody,
    EntityTypeBody,
    EvalRunBody,
    ExportDatasetBody,
    FeedbackBody,
    InitBody,
    LinkWorkflowBody,
    MaterializeBody,
    PredictBody,
    ScheduleBody,
    StudyBody,
    SyncEntitiesBody,
    TrainBody,
)

_STATIC = Path(__file__).resolve().parent / "static"
_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def create_app(
    db_path: str,
    config_path: str | None = None,
    api_key: str | None = None,
    tenant_id: str = "default",
):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        from starlette.responses import JSONResponse
    except ImportError as e:
        raise SystemExit(
            'Platform API requires: pip install "autoresearch-oss[platform]"'
        ) from e

    default_tenant = tenant_id or os.environ.get("AUTORESEARCH_TENANT_ID", "default")
    store = RegistryStore(db_path, tenant_id=default_tenant)
    rbac = parse_rbac_env()
    if api_key:
        from ..governance import ApiPrincipal

        rbac[api_key] = ApiPrincipal("legacy", "admin", frozenset({"*"}))
    app = FastAPI(title="autoresearch platform", version="0.3.0")
    app.state.infer_cache: dict = {"path": None, "model_id": None, "svc": None}
    start_scheduler()

    def _store(req: Request | None = None) -> RegistryStore:
        tid = default_tenant
        if req is not None:
            tid = req.headers.get("X-Tenant-Id", default_tenant)
        if tid == default_tenant:
            return store
        return RegistryStore(db_path, tenant_id=tid)

    def _actor(req: Request) -> str:
        p = getattr(req.state, "principal", None)
        return getattr(p, "name", None) or "anonymous"

    @app.middleware("http")
    async def _rbac(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and (rbac or api_key):
            secret = (
                request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            )
            principal = rbac.get(secret) if secret else None
            if api_key and secret == api_key:
                from ..governance import ApiPrincipal

                principal = ApiPrincipal("legacy", "admin", frozenset({"*"}))
            if not authorize(principal, request.method, path):
                return JSONResponse({"detail": "forbidden"}, status_code=403)
            request.state.principal = principal
        return await call_next(request)

    def _cfg(path: str | None = None):
        p = path or config_path
        if not p or not os.path.isfile(p):
            raise HTTPException(
                400, "config not set; start API with --config or pass config query/body"
            )
        return load_config(p)

    def _clear_infer_cache():
        app.state.infer_cache = {"path": None, "model_id": None, "svc": None}

    def _infer_svc(cfg_path: str) -> Service:
        cfg = load_config(cfg_path)
        wf = Workflow.from_dict(cfg["workflow"])
        m = production(store.reg, wf.name)
        if not m:
            raise HTTPException(404, "no production model")
        cache = app.state.infer_cache
        if (
            cache.get("path") == cfg_path
            and cache.get("model_id") == m["id"]
            and cache.get("svc") is not None
        ):
            return cache["svc"]
        model = load_model(store.reg, m["id"], wf)
        pol = GatePolicy(
            set(cfg.get("gate", {}).get("safety_critical", [])),
            cfg.get("gate", {}).get("min_confidence", 0.6),
        )
        log_path = str(Path(db_path).resolve().parent / "platform_live.jsonl")
        svc = Service(model, m["id"], wf, pol, log_path=log_path, api_key=api_key)
        app.state.infer_cache = {"path": cfg_path, "model_id": m["id"], "svc": svc}
        return svc

    @app.get("/health")
    def health():
        return {"ok": True, "db": db_path, "platform_version": "0.3.0"}

    @app.get("/api/v1/workflows")
    def list_workflows():
        return store.list_workflows()

    @app.get("/api/v1/models")
    def list_models(workflow: str | None = None):
        return store.list_models(workflow)

    @app.get("/api/v1/experiments")
    def list_experiments(workflow: str | None = None):
        return store.list_experiments(workflow)

    @app.get("/api/v1/eval/runs")
    def list_eval_runs(workflow: str | None = None):
        return store.list_eval_runs(workflow)

    @app.post("/api/v1/eval/run")
    def eval_run(req: EvalRunBody, request: Request):
        cfg = load_config(req.config)
        wf, outcome, _, _ = build_parts(cfg)
        st = _store(request)
        df = load_episodes(cfg["data"], wf)
        if req.split:
            from ..quality import clean
            from ..splits import split_by_group

            df, _ = clean(df, wf)
            df = split_by_group(df, wf.group_by, seed=cfg.get("seed", 0))[req.split]
        try:
            r = run_eval(st.reg, wf, outcome, df, model_id=req.model_id)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        st.audit(_actor(request), "eval.run", {"id": r["id"], "metrics": r["metrics"]})
        return r

    @app.get("/api/v1/about")
    def about_product():
        return {
            "product": "decision-model factory + governance loop",
            "is_not": [
                "full enterprise ontology platform",
                "LLM chat or RAG app host",
            ],
            "backends": {"default": "tabular", "optional": "llm"},
            "docs": "/docs/product-positioning.md",
        }

    @app.get("/api/v1/models/production")
    def prod_model(workflow: str):
        m = store.production_model(workflow)
        if not m:
            raise HTTPException(404, "no production model")
        return m

    @app.get("/api/v1/lineage/{artifact_id:path}")
    def lineage(artifact_id: str):
        try:
            return store.lineage(artifact_id)
        except KeyError:
            raise HTTPException(404, "unknown artifact")

    @app.post("/api/v1/models/{model_id:path}/promote")
    def promote(model_id: str):
        try:
            mid = store.promote(model_id)
            _clear_infer_cache()
            return {"production": mid}
        except KeyError:
            raise HTTPException(404, "unknown model")

    @app.post("/api/v1/workflows/{workflow_name}/rollback")
    def rollback_workflow(workflow_name: str):
        try:
            out = {"production": store.rollback_workflow(workflow_name)}
            _clear_infer_cache()
            return out
        except RuntimeError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/v1/doctor")
    def doctor(disk_path: str | None = None):
        return detect(disk_path)

    @app.get("/api/v1/templates")
    def templates():
        return list_templates()

    @app.post("/api/v1/projects/init")
    def projects_init(req: InitBody):
        if req.template == "list":
            raise HTTPException(400, "use GET /api/v1/templates")
        try:
            files = init_project(req.path, req.template)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"path": req.path, "template": req.template, "files": files}

    @app.post("/api/v1/validate")
    def validate_config(req: ConfigBody):
        if not os.path.isfile(req.config):
            raise HTTPException(400, f"config not found: {req.config}")
        return validate_data(load_config(req.config))

    @app.post("/api/v1/train")
    def train(req: TrainBody):
        if not os.path.isfile(req.config):
            raise HTTPException(400, f"config not found: {req.config}")
        cfg_dir = os.path.dirname(os.path.abspath(req.config))
        out = req.out
        if not os.path.isabs(out):
            out = os.path.join(cfg_dir, out)
        job_id = submit_train(
            repo_root=_REPO_ROOT,
            config=os.path.abspath(req.config),
            db_path=os.path.abspath(db_path),
            out=out,
            promote=req.promote,
            cwd=cfg_dir,
        )
        return {"job_id": job_id, "status": "running"}

    @app.get("/api/v1/jobs/{job_id}")
    def job_status(job_id: str):
        j = get_job(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        if j.get("status") in ("done", "failed"):
            _clear_infer_cache()
        return j

    @app.post("/api/v1/export-dataset")
    def export_dataset_api(req: ExportDatasetBody):
        cfg = _cfg(req.config)
        wf = Workflow.from_dict(cfg["workflow"])
        out_dir = req.out
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(
                os.path.dirname(os.path.abspath(req.config)), out_dir
            )
        r = export_dataset(
            store.reg,
            load_episodes(cfg["data"], wf),
            wf,
            out_dir,
            req.name,
            ScrubPolicy(**cfg.get("export", {})),
        )
        return {"id": r["id"], "path": r["path"]}

    @app.get("/api/v1/entity-types")
    def list_entity_types():
        return store.reg.list("entity_type")

    @app.post("/api/v1/entity-types")
    def create_entity_type(body: EntityTypeBody):
        ensure_entity_type(
            store.reg,
            body.name,
            {
                "description": body.description,
                "key": body.key or f"{body.name.lower()}_id",
            },
        )
        return export_ontology(store.reg)["entity_types"]

    @app.get("/api/v1/entities")
    def list_entities(type: str | None = None):
        rows = store.reg.list("entity")
        if type:
            rows = [e for e in rows if e["payload"].get("type") == type]
        return rows

    @app.post("/api/v1/entities")
    def create_entity(body: EntityBody):
        eid = upsert_entity(store.reg, body.type, body.external_id, body.properties)
        return store.reg.get(eid)

    @app.get("/api/v1/ontology/export")
    def ontology_export():
        return export_ontology(store.reg)

    @app.post("/api/v1/entities/sync")
    def sync_entities(req: SyncEntitiesBody, request: Request):
        cfg = _cfg(req.config)
        st = _store(request)
        wf = Workflow.from_dict(cfg["workflow"])
        links = load_entity_links(cfg)
        if not links:
            raise HTTPException(
                400, 'config has no entity_links; add e.g. {"asset": {"type": "Asset"}}'
            )
        df = load_episodes(cfg["data"], wf)
        out = sync_entities_from_episodes(st.reg, df, links)
        if req.sync_cases:
            out["cases"] = sync_case_entities_from_episodes(st.reg, df, wf)
        st.audit(_actor(request), "entity.sync", out)
        return out

    @app.post("/api/v1/episodes/materialize")
    def episodes_materialize(req: MaterializeBody, request: Request):
        cfg = load_config(req.config)
        wf = Workflow.from_dict(cfg["workflow"])
        st = _store(request)
        fmap = load_episode_field_map(cfg, wf)
        df = materialize_episodes_from_entities(
            st.reg, wf, req.entity_types, fmap, tenant_id=st.tenant_id
        )
        if df.empty:
            raise HTTPException(
                400,
                "no episodes materialized; sync Case entities from CSV first (entity sync + sync_cases)",
            )
        out_path = req.out
        if not out_path:
            out_path = os.path.join(
                os.path.dirname(os.path.abspath(req.config)),
                "data/episodes_from_ontology.csv",
            )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        st.audit(
            _actor(request),
            "episodes.materialize",
            {"rows": len(df), "path": out_path, "types": req.entity_types},
        )
        return {"rows": len(df), "path": out_path}

    @app.post("/api/v1/entities/link-workflow")
    def entities_link_workflow(req: LinkWorkflowBody, request: Request):
        st = _store(request)
        link_entity_to_workflow(st.reg, req.entity_id, req.workflow)
        st.audit(_actor(request), "entity.link_workflow", req.model_dump())
        return {"ok": True}

    @app.get("/api/v1/studies/digest")
    def studies_digest(workflow: str, request: Request):
        return experiment_digest(_store(request).reg, workflow)

    @app.post("/api/v1/studies/propose")
    def studies_propose(req: StudyBody, request: Request):
        cfg = load_config(req.config)
        wf, outcome, synth, backend = build_parts(cfg)
        from ..techniques import select

        traits = {
            "single_correct_trace": True,
            "has_failures": True,
            "verifiable_reward": bool(getattr(outcome, "verifiable", False)),
        }
        techniques = cfg.get("techniques") or select(traits, backend.name)
        recipes, digest = propose_study_recipes(
            _store(request).reg,
            wf,
            backend,
            techniques,
            use_llm=req.use_llm,
            llm_model=req.llm_model,
        )
        return {
            "digest": digest,
            "recipes": [r.to_dict() for r in recipes],
        }

    @app.get("/api/v1/audit")
    def audit_log(request: Request, limit: int = 100):
        return _store(request).list_audit(limit=limit)

    @app.get("/api/v1/schedules")
    def schedules_list():
        return list_schedules()

    @app.post("/api/v1/schedules")
    def schedules_add(req: ScheduleBody, request: Request):
        sid = add_schedule(
            action=req.action,
            interval=req.interval,
            config=os.path.abspath(req.config),
            db_path=os.path.abspath(db_path),
            repo_root=_REPO_ROOT,
            tenant_id=_store(request).tenant_id,
            extra={"live": req.live, "out": req.out, "promote": req.promote},
        )
        _store(request).audit(
            _actor(request), "schedule.add", {"id": sid, **req.model_dump()}
        )
        return {"id": sid}

    @app.delete("/api/v1/schedules/{schedule_id}")
    def schedules_del(schedule_id: str, request: Request):
        if not remove_schedule(schedule_id):
            raise HTTPException(404, "unknown schedule")
        _store(request).audit(_actor(request), "schedule.remove", {"id": schedule_id})
        return {"removed": schedule_id}

    @app.get("/api/v1/monitor")
    def monitor(config: str | None = None, live: str | None = None):
        cfg = _cfg(config)
        if not live or not os.path.isfile(live):
            raise HTTPException(400, "live= path to CSV required")
        wf = Workflow.from_dict(cfg["workflow"])
        ref = load_episodes(cfg["data"], wf)
        live_df = load_episodes({**cfg["data"], "path": live, "format": None}, wf)
        d = drift_report(ref, live_df, wf)
        p = performance_report(live_df, wf) if "status" in live_df.columns else {}
        again, why = should_retrain(d, p)
        return {"drift": d, "performance": p, "retrain": again, "reasons": why}

    @app.post("/api/v1/predict")
    def predict(body: PredictBody):
        cfg_path = body.config or config_path
        if not cfg_path:
            raise HTTPException(400, "pass config in body or start API with --config")
        cfg = load_config(cfg_path)
        wf = Workflow.from_dict(cfg["workflow"])
        if wf.name != body.workflow:
            raise HTTPException(400, f"config workflow is {wf.name!r}")
        svc = _infer_svc(cfg_path)
        try:
            return svc.predict(body.inputs)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/v1/feedback")
    def feedback(body: FeedbackBody):
        cfg_path = body.config or config_path
        if not cfg_path:
            raise HTTPException(400, "pass config in body or start API with --config")
        svc = _infer_svc(cfg_path)
        try:
            return svc.feedback(
                {
                    "request_id": body.request_id,
                    "final_action": body.final_action,
                    "outcome_success": body.outcome_success,
                    "slice": body.slice,
                }
            )
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.get("/")
    def dashboard():
        index = _STATIC / "index.html"
        if not index.is_file():
            raise HTTPException(404, "dashboard static files missing")
        return FileResponse(index)

    if _STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    return app


def run_api(
    db_path: str,
    config_path: str | None,
    host: str,
    port: int,
    api_key: str | None = None,
    tenant_id: str = "default",
):
    import uvicorn

    app = create_app(db_path, config_path, api_key, tenant_id=tenant_id)
    uvicorn.run(app, host=host, port=port, log_level="info")

"""Registry access layer for the platform API (SQLite today; Postgres later)."""

from __future__ import annotations

from .lifecycle import production, promote_to_production, rollback
from .ontology import Registry


class RegistryStore:
    def __init__(self, db_path: str, tenant_id: str = "default"):
        self.db_path = db_path
        self.tenant_id = tenant_id
        self._reg: Registry | None = None

    @property
    def reg(self) -> Registry:
        if self._reg is None:
            self._reg = Registry(self.db_path, tenant_id=self.tenant_id)
        return self._reg

    def audit(self, actor: str, action: str, detail: dict | None = None) -> None:
        from .governance import record_audit

        record_audit(
            self.reg.db,
            tenant_id=self.tenant_id,
            actor=actor,
            action=action,
            detail=detail,
        )

    def list_audit(self, limit: int = 100) -> list:
        from .governance import list_audit

        return list_audit(self.reg.db, self.tenant_id, limit=limit)

    def list_workflows(self) -> list:
        return self.reg.list("workflow")

    def list_models(self, workflow: str | None = None) -> list:
        rows = self.reg.list("model")
        if workflow:
            rows = [m for m in rows if m["name"] == workflow]
        return rows

    def list_experiments(self, workflow: str | None = None) -> list:
        rows = self.reg.list("experiment")
        if workflow:
            wf_ids = {w["id"] for w in self.list_workflows() if w["name"] == workflow}
            rows = [e for e in rows if any(p in wf_ids for p in e.get("parents", []))]
        return rows

    def list_eval_runs(self, workflow: str | None = None) -> list:
        rows = self.reg.list("eval_run")
        if workflow:
            rows = [r for r in rows if r["name"] == workflow]
        return rows

    def get(self, artifact_id: str) -> dict:
        return self.reg.get(artifact_id)

    def lineage(self, artifact_id: str) -> list:
        return self.reg.lineage(artifact_id)

    def production_model(self, workflow_name: str):
        return production(self.reg, workflow_name)

    def promote(self, model_id: str) -> str:
        return promote_to_production(self.reg, model_id)

    def rollback_workflow(self, workflow_name: str) -> str:
        return rollback(self.reg, workflow_name)

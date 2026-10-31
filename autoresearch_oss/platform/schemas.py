"""Pydantic request bodies for platform API (module-level for FastAPI)."""

from __future__ import annotations

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    BaseModel = object  # type: ignore[misc, assignment]


class ConfigBody(BaseModel):
    config: str


class TrainBody(BaseModel):
    config: str
    out: str = "artifacts"
    promote: bool = False


class InitBody(BaseModel):
    template: str
    path: str


class ExportDatasetBody(BaseModel):
    config: str
    name: str
    out: str = "published"


class EntityTypeBody(BaseModel):
    name: str
    description: str = ""
    key: str = ""


class EntityBody(BaseModel):
    type: str
    external_id: str
    properties: dict = {}


class PredictBody(BaseModel):
    workflow: str
    inputs: dict
    config: str | None = None


class FeedbackBody(BaseModel):
    request_id: str
    final_action: str
    outcome_success: bool
    slice: str | None = None
    config: str | None = None


class EvalRunBody(BaseModel):
    config: str
    model_id: str | None = None
    split: str = "eval"


class SyncEntitiesBody(BaseModel):
    config: str | None = None
    sync_cases: bool = True


class MaterializeBody(BaseModel):
    config: str
    entity_types: list[str] = ["Case"]
    out: str | None = None


class ScheduleBody(BaseModel):
    action: str
    interval: str
    config: str
    live: str | None = None
    out: str = "artifacts"
    promote: bool = False


class StudyBody(BaseModel):
    config: str
    use_llm: bool = False
    llm_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"


class LinkWorkflowBody(BaseModel):
    entity_id: str
    workflow: str

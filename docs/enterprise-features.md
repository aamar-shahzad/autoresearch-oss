# Enterprise gaps (paper-aligned)

Implemented in this repo (not full Foundry):

| Paper / enterprise need | Implementation |
|-------------------------|----------------|
| Ontology → episodes | `Case` entities via **entity sync --cases** or UI **Entity sync (+ Case)**; **Materialize episodes** writes `data/episodes_from_ontology.csv` |
| Multi-workflow graph | Entity `properties.workflows[]`; **POST /api/v1/entities/link-workflow** |
| Multi-tenant registry | `tenant_id` on artifacts + **X-Tenant-Id** header / config `tenant_id` |
| RBAC | `AUTORESEARCH_PLATFORM_RBAC=admin:SECRET:read,write,*` or `--api-key` (admin) |
| Audit | SQLite `audit_log`; **GET /api/v1/audit** |
| Schedulers | Background interval jobs: **POST /api/v1/schedules** (`validate` / `train` / `monitor`) |
| Study agent | `studies.py` reads prior experiments; optional **SmolLM2-135M** proposals; wired into **train** loop via config `study` |
| Tiny LLM proof | `autoresearch smoke-llm` or `RUN_TINY_LLM_SMOKE=1 pytest tests/test_tiny_llm_smoke.py` |

## Quick flow (maintenance)

```bash
cd myproj
autoresearch entity sync --config autoresearch.json --db registry.db --cases
autoresearch entity materialize --config autoresearch.json --db registry.db --out data/episodes_from_ontology.csv
# point data.path at materialized CSV, or keep episodes.csv

export AUTORESEARCH_PLATFORM_RBAC='ops:mytoken:read,write'
autoresearch platform api --db registry.db --config autoresearch.json --tenant-id default
```

## Study + tiny LLM in config

```json
"study": {
  "enabled": true,
  "llm": false,
  "llm_model": "HuggingFaceTB/SmolLM2-135M-Instruct"
}
```

Set `"llm": true` only on a machine with `pip install -e ".[llm]"` and enough RAM for 135M.

## Still not Foundry

No live warehouse connectors, no enterprise SSO, no horizontal multi-writer registry — see [platform-roadmap.md](platform-roadmap.md).

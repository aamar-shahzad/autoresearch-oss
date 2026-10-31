# Platform API

Requires: `pip install "autoresearch-oss[platform]"`

## Start API + dashboard

```bash
cd myproj   # or any dir with registry.db
autoresearch platform api --db registry.db --config autoresearch.json --port 8090
```

Open http://127.0.0.1:8090/ — validate, train, entity sync, predict/feedback, models, promote, rollback, monitor, ontology export, init project, doctor.

Inference model HTTP (`autoresearch serve`) stays on port 8080; platform API is **control plane** on 8090 and can run predict/feedback against production models (logs `platform_live.jsonl` beside the registry DB).

## REST (prefix `/api/v1`)

| Method | Path                           | Description                                                               |
| ------ | ------------------------------ | ------------------------------------------------------------------------- |
| GET    | `/health`                      | Liveness                                                                  |
| GET    | `/doctor`                      | Hardware / stack detection (CLI doctor)                                   |
| GET    | `/templates`                   | Init templates (`init project`)                                           |
| POST   | `/projects/init`               | Body `{"template", "path"}`                                               |
| POST   | `/validate`                    | Body `{"config"}` — data checks                                           |
| POST   | `/train`                       | Body `{"config", "out?", "promote?"}` — background job                    |
| GET    | `/jobs/{job_id}`               | Train job status + log tail                                               |
| POST   | `/export-dataset`              | Body `{"config", "name", "out?"}`                                         |
| GET    | `/workflows`                   | Registry workflows                                                        |
| GET    | `/models?workflow=`            | All model versions                                                        |
| GET    | `/models/production?workflow=` | Current production model                                                  |
| GET    | `/experiments?workflow=`       | Experiment artifacts                                                      |
| GET    | `/lineage/{artifact_id}`       | Lineage chain                                                             |
| POST   | `/models/{id}/promote`         | Set production model                                                      |
| POST   | `/workflows/{name}/rollback`   | Roll back to previous production                                          |
| GET    | `/entity-types`                | Ontology entity types                                                     |
| POST   | `/entity-types`                | Register entity type                                                      |
| GET    | `/entities?type=`              | Entities                                                                  |
| POST   | `/entities`                    | Upsert entity                                                             |
| GET    | `/ontology/export`             | Full ontology JSON                                                        |
| POST   | `/entities/sync`               | Body `{"config": "/path/to/autoresearch.json"}`                           |
| GET    | `/monitor?config=&live=`       | Drift + performance (same as CLI monitor)                                 |
| POST   | `/predict`                     | Body `{"workflow", "inputs", "config?"}` — production model + gate        |
| POST   | `/feedback`                    | Body `{"request_id", "final_action", "outcome_success", "slice?", "config?"}` |

## Ontology-lite (CLI)

Add to `autoresearch.json`:

```json
"entity_links": {
  "asset": {"type": "Asset"},
  "site": {"type": "Site"}
}
```

```bash
autoresearch entity sync --config autoresearch.json --db registry.db
autoresearch export-ontology --db registry.db -o ontology.json
```

Maintenance template includes default `entity_links`.

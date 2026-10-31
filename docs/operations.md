# Operating it

**Train** `autoresearch train --config c.json [--promote]` - preflights the chosen model against your hardware, searches,
evaluates, promotes only what clears every threshold on validation *and* the holdout. Output: registry + `report.md`.

**Serve** `autoresearch serve --config c.json --api-key ...`
`POST /v1/predict {"inputs": {...}}` -> `{action, confidence, needs_approval, escalate, request_id}`;
`POST /v1/feedback {request_id, final_action, outcome_success, slice}`. Low confidence escalates to a human;
`gate.safety_critical` actions require approval. Everything is logged to JSONL.
Security: stdlib server with a bearer-token check. Terminate TLS and authenticate at a proxy for anything beyond localhost.

**Learn** `autoresearch ingest-live --log live.jsonl` or `train --live-log live.jsonl`: overrides and failures become
training signal (overrides are up-weighted). **Monitor** `autoresearch monitor --live live.csv`: PSI feature drift,
fault-mix shift, override/failure-rate alerts -> retrain recommendation. Schedule it (cron/Airflow); there is no built-in scheduler.

**Govern** `models-registry`, `promote`, `rollback`, `lineage`, `export-dataset` (scrubbed + hashed + data card).
SQLite registry: single writer; back it up like any database.

**Costs** set `unit_cost_usd`. Tabular units are abstract; the LLM backend reports wall-clock minutes (multiply by your $/GPU-min).

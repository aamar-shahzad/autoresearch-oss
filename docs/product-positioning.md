# Product positioning

This repository is a **decision-model factory + governance loop**. It is intentionally **not** two other things people often confuse with “AI platforms.”

## What this is

| Piece | Role |
|-------|------|
| **Workflow** | One decision: inputs, allowed actions, success rule, slice thresholds |
| **AutoResearch loop** | Search recipes → validate → holdout → promote |
| **Registry** | Lineage you own: data → recipe → experiment → model |
| **Serve** | Workflow inference + human gate + feedback log for retrain |
| **Monitor** | Drift + override/failure rates → trigger retrain |
| **Ontology-lite** | `entity_links`, Case entities, materialize episodes — not a live enterprise graph |

```text
validate → train (search) → promote → serve → feedback → monitor → train (live)
```

## What this is not

### Full ontology platform

Not Foundry-style: live object graph, cross-app permissions, automatic pipelines from warehouse → graph → training without you defining episodes.

You bring **episodes** (CSV/JSONL) or **Case** entities synced from columns. See [ontology-lite.md](ontology-lite.md).

### LLM app host

Not a chat product, RAG doc Q&A, or open-ended agent server.

`serve` exposes **workflow predict + feedback** (`POST /v1/predict`, `POST /v1/feedback`) so production outcomes feed the **same loop** — not general assistant traffic.

## Backends: tabular vs LLM

| | Tabular (default) | LLM (optional) |
|--|-------------------|----------------|
| **Proven here** | Yes — maintenance use case, CI, CPU | Same loop; TRL/PEFT; needs `.[llm]` + GPU for real jobs |
| **Maintenance template** | `backend.type: tabular` | Copy [examples/maintenance_llm.json](../examples/maintenance_llm.json) |
| **Smoke** | `make use-case` | `autoresearch smoke-llm` (1 TRL step on SmolLM2-135M only) |

Same **Workflow**, thresholds, holdout, promote, serve, monitor for both — only the **weights backend** changes.

See [backends-tabular-vs-llm.md](backends-tabular-vs-llm.md).

## Eval cases and runs

- **Case** = one episode row (or Case entity).
- **Eval run** = score a model on a fixed dataframe; registry kind `eval_run` with per-case predictions.

```bash
autoresearch eval run --config autoresearch.json --db registry.db
autoresearch about
```

Platform: **Run eval** on the dashboard (8090).

# Ontology-lite (not a full ontology platform)

**Goal:** Link episodes to shared **Asset** / **Site** (and **Case**) records in the registry you already use for models — without a live enterprise object graph.

## In config

```json
"entity_links": {
  "asset": { "type": "Asset" },
  "site": { "type": "Site" }
},
"entity_episode_map": {
  "fields": { "episode_id": "episode_id", "asset": "asset", "action": "action", ... }
}
```

## CLI

```bash
autoresearch entity sync --config autoresearch.json --db registry.db --cases
autoresearch entity materialize --config autoresearch.json --db registry.db
```

Cases stored as entities → CSV for training or eval.

## Not included

- Live sync from warehouse/CRM
- Row-level security / multi-app object graph
- Automatic “select objects in UI → train” without episode schema

That is [platform-roadmap.md](platform-roadmap.md) territory, not current scope.

# Tabular vs LLM backend

Both use the **same** `Workflow`, search loop, gates, serve, and monitor.

## Default: tabular (maintenance template)

```json
"backend": { "type": "tabular" }
```

- Runs on CPU
- End-to-end verified: `make use-case`, `myproj` train/serve/monitor

## Optional: LLM post-training

1. `pip install -e ".[llm]"`
2. Start from [examples/maintenance_llm.json](../examples/maintenance_llm.json) (copy into your project as `autoresearch.json` and fix `data.path`)
3. `autoresearch models choose --config autoresearch.json --pull` or set `backend.model.id`
4. `autoresearch train --config autoresearch.json --promote` — same loop, longer runs
5. `autoresearch smoke-llm` — only checks TRL can train 1 step on SmolLM2-135M

```json
"backend": {
  "type": "llm",
  "model": { "id": "HuggingFaceTB/SmolLM2-135M-Instruct" },
  "seq_len": 512,
  "batch_size": 1
}
```

Use a larger model only after smoke + a short probe train on your hardware.

## Serve is not an LLM chat host

With LLM backend, `serve` still returns **one action from `action_space`** (choice workflow) plus confidence and gate flags — not open-ended chat completions as the product surface.

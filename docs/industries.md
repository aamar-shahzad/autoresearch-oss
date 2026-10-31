# Industry templates (`autoresearch init <template>`)

| template | task | backend | typical slice | outcome scoring |
|---|---|---|---|---|
| `maintenance-diagnostics` | pick the repair action for an alarm | tabular (CPU) | root cause | twin or offline replay |
| `support-triage` | route a ticket | LLM + LoRA | product area | held-out labelled set / twin (replay is weak for text) |
| `structured-extraction` | document -> JSON | LLM + LoRA/GRPO | document type | verifier (schema + field match) |

These are starting points: edit `workflow` (inputs, actions, thresholds, `unsafe_actions`, `unsafe_patterns`), point `data`
at your export, and choose a model with `autoresearch models choose --config autoresearch.json`.

**Regulated domains (health, finance, legal, safety-critical):** the gate, lineage and holdout discipline help, but this
software does not make a model compliant or safe. Validate with domain experts, keep a human in the loop for consequential
actions, and review licenses of base models and data.

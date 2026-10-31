# autoresearch-oss

An open-source fine-tuning workbench built around the loop in Palantir's _AutoResearch and the Fine-Tuning
Ontology_ paper. It **detects your hardware, shows which open models you can fine-tune on it, lets you pick and
download one safely, searches data/technique/hyperparameter recipes against held-back cases, gates deployment
behind a human, serves the model, and learns from what happens in production** - with every artifact tracked in
a registry you own. Industry-agnostic: you bring episodes (situation + action + measured outcome).

Not Palantir's product; no Foundry.

```bash
git clone https://github.com/aamar-shahzad/autoresearch-oss.git && cd autoresearch-oss
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[llm]"                             # core only: pip install -e .

autoresearch doctor                          # 1. what hardware/ML stack do I have?
autoresearch models list                     # 2. which open models can I fine-tune HERE?
autoresearch init support-triage myproj      # 3. scaffold a project from an industry template
autoresearch models choose --config myproj/autoresearch.json --pull   #    pick a model, write config, download
autoresearch validate --config myproj/autoresearch.json               # 4. check your data before spending compute
autoresearch train    --config myproj/autoresearch.json --promote     # 5. search -> evaluate -> promote
autoresearch serve    --config myproj/autoresearch.json               # 6. gated HTTP service + feedback log
autoresearch train    --config ... --live-log live.jsonl              # 7. learn from production
autoresearch quickstart                      # no data/GPU? tour on built-in SAMPLE data
```

## What's in the box

|                   |                                                                                                                                                                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Hardware**      | NVIDIA / AMD / Apple Silicon / CPU, container-aware RAM, disk, installed ML libs (`hardware.py`)                                                                                                                                                                         |
| **Models**        | catalog of 41 open-weight models (Qwen, Llama, Gemma, Phi, Mistral, SmolLM, OLMo, Granite, DeepSeek-R1-distill, gpt-oss...), memory planner (LoRA / QLoRA / full), license + gating awareness, live Hub search, safe download with revision pinning, interactive chooser |
| **Data**          | CSV / Parquet / JSONL / SQLite / SQLAlchemy connectors; cleaning, leakage guard, group-wise splits; synthetic data (resample or LLM + verifier)                                                                                                                          |
| **Tasks**         | `choice` (closed action set + confidence) and `generation` (free text scored by verifiers)                                                                                                                                                                               |
| **Search**        | probe -> prune -> train -> name failure modes -> mutate mixtures/synthetics/hparams, under a compute/$ budget; holdout touched only by passing candidates                                                                                                                |
| **Backends**      | CPU tabular (runs anywhere), LLM via TRL+PEFT: SFT / DPO / GRPO, LoRA / QLoRA                                                                                                                                                                                            |
| **Ops**           | model stages + rollback, human approval gate, HTTP service with feedback logging, drift & override monitoring, scrubbed dataset publishing with data cards, markdown reports with spend                                                                                  |
| **Extensibility** | plugin registry + entry points for connectors, outcomes, verifiers, synthesizers, backends, hubs                                                                                                                                                                         |
| **Templates**     | maintenance diagnostics, support triage, structured extraction                                                                                                                                                                                                           |

Docs: [hardware & models](docs/hardware-and-models.md) · [data format](docs/data-format.md) · [extending](docs/extending.md) · [operations](docs/operations.md) · [industries](docs/industries.md) · [architecture](docs/architecture.md) · [**maintenance use case**](docs/use-case-maintenance-diagnostics.md) (`make use-case`)

## Honest status - what has and hasn't been verified

| area                                                                 | verification                                                                                                                                                                                          |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hardware parsing (nvidia-smi, rocm-smi, cgroups)                     | unit-tested against captured output; **live-detected only on a CPU box** - no GPU/Apple machine was available                                                                                         |
| Memory planner                                                       | tested for monotonicity and platform rules; numbers are **engineering estimates, not measured** on real GPUs                                                                                          |
| Model catalog                                                        | curated **snapshot** from the maintainers' knowledge; sizes approximate; it will go stale (extend via `~/.autoresearch/catalog.json`)                                                                 |
| Hub search/download                                                  | logic tested against **fake clients**; **never run against the live Hugging Face Hub** (no network access during development)                                                                         |
| TRL/PEFT/Transformers calls                                          | argument names checked against trl 1.17 / peft 0.21 / transformers 5.22 **source code**; **training and inference were never executed** (no torch/GPU). Run a 1-step smoke test on SmolLM2-135M first |
| Tabular backend, loop, registry, serving, monitoring, scrubbing, CLI | covered by 41 automated tests incl. a live HTTP service and end-to-end CLI runs on CPU                                                                                                                |
| Accuracy on real data                                                | none claimed. The built-in sample world is synthetic with crisp rules; near-perfect scores there only prove the mechanics                                                                             |

## Limits

- Offline `replay` scoring is a pessimistic screening estimate, not proof; a twin/verifier/labelled eval set is better.
- Scrubbing is baseline (hash/redact/round/k-anonymity), not differential privacy or legal compliance.
- No scheduler, no multi-user access control, no marketplace for weights; SQLite is single-writer; the stdlib server
  needs a TLS/auth proxy outside localhost.
- A good score does not make a model safe for consequential decisions; keep humans in the loop and validate with domain experts.

MIT licensed. Contributions: add plugins, catalog entries, hardware parsers and real-GPU test reports.

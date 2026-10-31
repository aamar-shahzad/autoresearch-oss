# Architecture

```
hardware.py ──► models/fit.py ◄── models/catalog.json            (what can run here?)
                    │              models/hub.py (search/pull)
config.json ─► connectors ─► quality ─► splits(by group) ─► mixtures + synth
                                                        │
                       techniques/recipes ─► loop (probe → train → eval on validation → name failure modes → mutate)
                                                        │              ▲ backends: tabular | llm(TRL+PEFT) | plugin
                                      holdout confirm ──┘              │ outcomes: twin | replay | verifier | plugin
                                              │
                               registry (SQLite: lineage of data → recipe → experiment → model)
                                              │
                    lifecycle (staging/production/rollback) ─► serve (gate + feedback log)
                                              ▲                         │
                              monitor (drift) ┴── ingest-live ◄────────┘
```
Principles: every recipe is written before it runs; the holdout is touched only by candidates that already pass
validation (and counted); a slice that cannot be measured blocks promotion; synthetic rows are always flagged and never
in the holdout; estimates (offline replay, memory plans) are labelled as estimates.

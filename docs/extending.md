# Extending

Everything swappable is looked up by name in `autoresearch_oss.plugins`:

| kind | you provide | used for |
|---|---|---|
| `connector` | `fn(spec, wf) -> DataFrame` | data sources (MES, CMMS, EHR, ticketing, warehouse) |
| `outcome` | `fn(spec) -> callable(df, actions) -> scores` | digital twin / rules engine / LLM judge / human-labelled eval |
| `verifier` | `fn(spec) -> callable(row, completion) -> 0..1` | free-text scoring and GRPO reward |
| `synth` | `fn(spec, outcome) -> callable(slice, n, seed) -> DataFrame` | synthetic episodes for rare slices |
| `backend` | `fn(spec, outcome, wf) -> backend` | a training engine (Unsloth, Axolotl, MLX, a cloud API...) |
| `hub` | object with `info/search/download` | model sources besides Hugging Face |

```python
from autoresearch_oss.plugins import register

@register("outcome", "my_twin")
def make(spec):
    def score(df, actions): ...        # return bool/probability per row
    score.verifiable = True            # marks it as a trustworthy reward (enables GRPO)
    return score
```
Expose it from your package with the `autoresearch_oss.plugins` entry point (see `examples/plugin_example`) and
reference it in config: `"outcome": {"type": "my_twin"}`. A plugin that fails to import is reported by `autoresearch
plugins` and never breaks the CLI.

A **backend** implements `name`, `bases`, and `train(recipe, df, workflow) -> TrainResult(model, compute_units, seconds)`
where `model.predict(df) -> DataFrame[action, confidence]` and `model.save(path)`. **Techniques** are entries in
`techniques.LIBRARY` (name, hyperparameter priors, whether they use failures). The loop's mutations (`loop.propose`)
are deliberately simple; replace them for smarter search.

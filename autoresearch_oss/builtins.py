"""Registers the built-in outcome functions, synthesizers and backends under the plugin registry.
Third-party packages register theirs the same way (see plugins.py)."""
from __future__ import annotations

from . import verifiers  # noqa: F401  (registers verifier + 'verifier' outcome)
from .backends import TabularBackend
from .outcomes import ReplayOutcome, load_callable
from .plugins import register
from .synth import LLMSynth, ResampleSynth


@register("outcome", "replay")
def _replay(spec):
    return ReplayOutcome(**{k: v for k, v in spec.items() if k in ("k", "radius_quantile")})


@register("outcome", "twin")
def _twin(spec):
    fn = load_callable(spec["callable"])
    try:
        fn.verifiable = True        # a twin gives a trustworthy reward, so GRPO becomes a candidate
    except AttributeError:
        pass
    return fn


@register("synth", "resample")
def _resample(spec, outcome):
    return ResampleSynth(spec.get("noise", 0.05))


@register("synth", "llm")
def _llm_synth(spec, outcome):
    return LLMSynth(load_callable(spec["complete"]), verify=outcome if spec.get("verify", True) else None)


@register("synth", "callable")
def _callable_synth(spec, outcome):
    return load_callable(spec["callable"])


@register("backend", "tabular")
def _tabular(spec, outcome, wf):
    if wf.task_type != "choice":
        raise ValueError("the tabular backend only supports task_type 'choice'")
    return TabularBackend(spec.get("bases", ["hgb", "rf"]), spec.get("seed", 0))


@register("backend", "llm")
def _llm(spec, outcome, wf):
    from .backends.llm import LLMBackend
    from .models.hub import DEFAULT_ROOT, local_models
    bases = list(spec.get("bases") or [])
    m = spec.get("model") or {}
    if m.get("path"):
        bases.append(m["path"])
    elif m.get("id"):
        local = next((x["path"] for x in local_models(spec.get("model_root", DEFAULT_ROOT)) if x["repo"] == m["id"]), None)
        if local is None and not spec.get("allow_hub_download", False):
            raise ValueError(f"{m['id']} is not downloaded. Run `autoresearch models pull {m['id']}` "
                             "(license, gating and disk are checked there) or set allow_hub_download.")
        bases.append(local or m["id"])
    if not bases:
        raise ValueError("llm backend needs a model: set backend.model.id or run `autoresearch models choose --config ...`")
    return LLMBackend(bases, outcome_fn=outcome, out_dir=spec.get("out_dir", "llm_runs"), mode=spec.get("mode", "auto"),
                      seq_len=spec.get("seq_len", 2048), batch_size=spec.get("batch_size", 1))


@register("hub", "huggingface")
def _hf(spec):
    from .models.hub import HubClient
    return HubClient(token=spec.get("token"))

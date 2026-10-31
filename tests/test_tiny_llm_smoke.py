"""Optional live GPU/CPU smoke for tiny LLM — run with RUN_TINY_LLM_SMOKE=1."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TINY_LLM_SMOKE") != "1",
    reason="set RUN_TINY_LLM_SMOKE=1 to download model and train 1 step",
)


def test_smol_lm_smoke():
    from autoresearch_oss.smoke_llm import run_smoke

    out = run_smoke("HuggingFaceTB/SmolLM2-135M-Instruct", max_steps=1)
    assert out.get("ok"), out

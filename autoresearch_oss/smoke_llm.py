"""One-step smoke test for tiny open-weight LLM backend (SmolLM2-135M)."""

from __future__ import annotations

import json
import os
import tempfile


def run_smoke(model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct", max_steps: int = 1) -> dict:
    import pandas as pd

    from ..backends.llm import to_sft
    from ..ontology import Workflow
    from ..plugins import get

    wf = Workflow.from_dict(
        {
            "name": "smoke",
            "description": "Pick one action.",
            "inputs": ["x"],
            "outcome_fields": ["outcome_success"],
            "action_space": ["a", "b"],
            "unsafe_actions": [],
            "slice_by": "x",
            "thresholds": {"fix_rate": [">=", 0.5]},
            "task_type": "choice",
        }
    )
    df = pd.DataFrame(
        [
            {"x": 1, "action": "a", "outcome_success": True},
            {"x": 2, "action": "b", "outcome_success": True},
        ]
    )
    rows = to_sft(df, wf)
    if len(rows) < 2:
        return {"ok": False, "error": "sft rows empty"}

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as e:
        return {"ok": False, "error": f"missing deps: {e}"}

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=True
    )
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
    ds = Dataset.from_list([{"text": r["prompt"] + r["completion"]} for r in rows])
    with tempfile.TemporaryDirectory() as td:
        args = TrainingArguments(
            output_dir=td,
            max_steps=max_steps,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_steps=max_steps,
            report_to=[],
        )
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
        trainer.train()
    return {"ok": True, "model": model_id, "steps": max_steps, "rows": len(rows)}


def main(model_id: str | None = None) -> int:
    mid = model_id or os.environ.get(
        "AUTORESEARCH_SMOKE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    out = run_smoke(mid)
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1

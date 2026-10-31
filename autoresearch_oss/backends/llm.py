"""LLM backend: SFT / DPO / GRPO with LoRA or QLoRA via Hugging Face TRL + PEFT.

Works for any workflow: `choice` tasks (closed action set, scored by log-likelihood, which also
gives a calibrated-ish confidence) and `generation` tasks (free text, scored by a verifier).

STATUS: the API usage was checked against trl 1.17 / peft 0.21 / transformers 5.22 sources
(trainer signatures, config fields, dataset formats). The training and inference paths have NOT
been executed on a GPU in this repo's CI -- run `autoresearch doctor` then a 1-step smoke run on a
tiny model (SmolLM2-135M) before a real job. The pure data-shaping helpers are unit-tested."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


class _Safe(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def _fmt(v) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, (float, np.floating)) else str(v)


def render_inputs(row, wf) -> str:
    if wf.prompt_template:
        return wf.prompt_template.format_map(_Safe({k: row[k] for k in wf.inputs}))
    return ", ".join(f"{k}={_fmt(row[k])}" for k in wf.inputs)


def format_prompt(row, wf) -> str:
    if wf.task_type == "generation":
        return f"{wf.description}\n\nInput:\n{render_inputs(row, wf)}\n\nOutput:"
    return (f"{wf.description} Reply with exactly one action from the list.\n"
            f"Actions: {', '.join(wf.action_space)}\nSituation: {render_inputs(row, wf)}\nAction:")


def _completion(row, wf) -> str:
    return " " + str(row["action"]).strip()


def to_sft(df: pd.DataFrame, wf) -> list:
    d = df[df["outcome_success"]]
    return [{"prompt": format_prompt(r, wf), "completion": _completion(r, wf)} for _, r in d.iterrows()]


def to_dpo_pairs(df: pd.DataFrame, wf, k: int = 1) -> list:
    """choice: chosen = action that WORKED in the most similar successful case with the same
    context group; rejected = the action that FAILED here.
    generation: pair a successful and a failed response to the SAME prompt."""
    pairs = []
    if wf.task_type == "generation":
        d = df.assign(_p=[format_prompt(r, wf) for _, r in df.iterrows()])
        for p, g in d.groupby("_p"):
            good, bad = g[g["outcome_success"]], g[~g["outcome_success"]]
            if len(good) and len(bad):
                pairs.append({"prompt": p, "chosen": _completion(good.iloc[0], wf),
                              "rejected": _completion(bad.iloc[0], wf)})
        return pairs
    from sklearn.neighbors import NearestNeighbors
    num = [c for c in wf.inputs if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in wf.inputs if c not in num]
    for key, g in (df.groupby(cat) if cat else [((), df)]):
        good, bad = g[g["outcome_success"]], g[~g["outcome_success"]]
        if len(good) < k or bad.empty or not num:
            continue
        mu, sd = good[num].mean(), good[num].std().replace(0, 1)
        nn = NearestNeighbors(n_neighbors=k).fit(((good[num] - mu) / sd).to_numpy())
        _, idx = nn.kneighbors(((bad[num] - mu) / sd).to_numpy())
        for (_, r), nb in zip(bad.iterrows(), idx):
            chosen = good.iloc[nb[0]]["action"]
            if chosen != r["action"]:
                pairs.append({"prompt": format_prompt(r, wf), "chosen": " " + chosen, "rejected": _completion(r, wf)})
    return pairs


def make_reward_fn(outcome_fn, wf):
    """Verifiable reward for GRPO. TRL passes extra dataset columns as kwargs, so the dataset rows
    carry the workflow inputs plus `reference` (the logged good action/response)."""
    def reward(prompts, completions, **cols):
        df = pd.DataFrame({k: cols[k] for k in wf.inputs})
        df["action"] = cols.get("reference", [""] * len(df))
        df["outcome_success"] = True
        acts = [c.strip().split()[0] if (wf.task_type == "choice" and c.strip()) else c.strip() for c in completions]
        return [float(x) for x in np.asarray(outcome_fn(df, acts), dtype=float)]
    return reward


def to_grpo_rows(df: pd.DataFrame, wf) -> list:
    d = df[df["outcome_success"]] if wf.task_type == "generation" else df
    return [{"prompt": format_prompt(r, wf), **{k: r[k] for k in wf.inputs}, "reference": r["action"]}
            for _, r in d.iterrows()]


def closed_set_probs(logliks: np.ndarray) -> np.ndarray:
    z = logliks - logliks.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def _from_pretrained(cls, base, dtype, **kw):
    """transformers >=4.56/5 renamed torch_dtype -> dtype; support both."""
    try:
        return cls.from_pretrained(base, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(base, torch_dtype=dtype, **kw)


@dataclass
class HFModel:
    model: object
    tokenizer: object
    wf: object
    max_new_tokens: int = 128

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        import torch
        self.model.eval()
        tok, dev = self.tokenizer, next(self.model.parameters()).device
        acts, conf = [], []
        with torch.no_grad():
            if self.wf.task_type == "generation":
                for _, r in df.iterrows():
                    enc = tok(format_prompt(r, self.wf), return_tensors="pt").to(dev)
                    out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                              output_scores=True, return_dict_in_generate=True,
                                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
                    gen = out.sequences[0, enc.input_ids.shape[1]:]
                    sc = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
                    acts.append(tok.decode(gen, skip_special_tokens=True).strip())
                    conf.append(float(sc.float().exp().mean()) if len(sc) else 0.0)
                return pd.DataFrame({"action": acts, "confidence": conf})
            A = list(self.wf.action_space)
            L = np.zeros((len(df), len(A)))
            for i, (_, r) in enumerate(df.iterrows()):
                p = tok(format_prompt(r, self.wf), return_tensors="pt").to(dev)
                for j, a in enumerate(A):
                    a_ids = tok(" " + a, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
                    ids = torch.cat([p.input_ids, a_ids], 1)
                    logits = self.model(ids).logits[0, p.input_ids.shape[1] - 1:-1]
                    L[i, j] = float(torch.log_softmax(logits.float(), -1).gather(1, a_ids[0][:, None]).sum())
        P = closed_set_probs(L)
        k = P.argmax(1)
        return pd.DataFrame({"action": np.array(A)[k], "confidence": P[np.arange(len(P)), k]})

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)            # LoRA adapter only
        self.tokenizer.save_pretrained(path)
        json.dump({"base": self.base_id, "workflow": self.wf.__dict__}, open(f"{path}/autoresearch.json", "w"))

    @classmethod
    def load(cls, path: str, wf, dtype=None, device_map=None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        meta = json.load(open(f"{path}/autoresearch.json"))
        tok = AutoTokenizer.from_pretrained(path)
        base = _from_pretrained(AutoModelForCausalLM, meta["base"], dtype or torch.bfloat16,
                                **({"device_map": device_map} if device_map else {}))
        m = cls(PeftModel.from_pretrained(base, path), tok, wf)
        m.base_id = meta["base"]
        return m


@dataclass
class TrainResult:
    model: HFModel
    compute_units: float
    seconds: float


class LLMBackend:
    """bases: Hub repo ids OR local directories from `autoresearch models pull`.
    mode: 'lora' | 'qlora' | 'auto' (auto picks from the hardware plan)."""
    name = "llm"

    def __init__(self, bases, outcome_fn=None, out_dir="llm_runs", mode="auto", hardware=None, seq_len=2048,
                 batch_size=1):
        self.bases, self.outcome_fn, self.out_dir = list(bases), outcome_fn, out_dir
        self.mode, self.hw, self.seq_len, self.batch_size = mode, hardware, seq_len, batch_size

    def _mode(self, base: str) -> str:
        if self.mode != "auto":
            return self.mode
        from ..hardware import detect
        from ..models import load_catalog
        from ..models.catalog import find
        from ..models.fit import plan
        e = find(load_catalog(), base.replace("__", "/")) or find(load_catalog(), os.path.basename(base))
        if e is None:
            return "lora"
        return plan(e, self.hw or detect(), self.seq_len, self.batch_size).best or "lora"

    def train(self, recipe, df: pd.DataFrame, workflow) -> TrainResult:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0, hp = time.time(), recipe.hp
        mode = self._mode(recipe.base)
        tok = AutoTokenizer.from_pretrained(recipe.base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        cuda = torch.cuda.is_available()
        bf16 = cuda and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
        kw = {}
        if mode == "qlora":
            from transformers import BitsAndBytesConfig
            kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
                  "device_map": {"": 0}}
        model = _from_pretrained(AutoModelForCausalLM, recipe.base, dtype, **kw)
        r = hp.get("lora_r", 16)
        lora = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
        out = os.path.join(self.out_dir, recipe.key())
        common = dict(output_dir=out, report_to=[], bf16=bf16, fp16=(cuda and not bf16), use_cpu=not cuda,
                      per_device_train_batch_size=self.batch_size, gradient_checkpointing=cuda,
                      save_strategy="no", logging_steps=10)
        epochs = {"num_train_epochs": hp.get("epochs", 2)}
        if hp.get("max_steps"):
            epochs = {"max_steps": hp["max_steps"]}
        tech = recipe.technique
        if tech == "sft":
            from trl import SFTConfig, SFTTrainer
            tr = SFTTrainer(model=model, processing_class=tok, peft_config=lora,
                            train_dataset=Dataset.from_list(to_sft(df, workflow)),
                            args=SFTConfig(learning_rate=hp.get("lr", 2e-4), max_length=self.seq_len, **epochs, **common))
        elif tech == "dpo":
            from trl import DPOConfig, DPOTrainer
            pairs = to_dpo_pairs(df, workflow)
            if not pairs:
                raise ValueError("DPO needs preference pairs: keep failures in the mixture (use_failures=True)")
            tr = DPOTrainer(model=model, processing_class=tok, peft_config=lora,
                            train_dataset=Dataset.from_list(pairs),
                            args=DPOConfig(learning_rate=hp.get("lr", 5e-6), beta=hp.get("beta", 0.1),
                                           max_length=self.seq_len, **epochs, **common))
        elif tech == "grpo":
            from trl import GRPOConfig, GRPOTrainer
            if self.outcome_fn is None:
                raise ValueError("GRPO needs an outcome function (verifiable reward)")
            tr = GRPOTrainer(model=model, processing_class=tok, peft_config=lora,
                             reward_funcs=make_reward_fn(self.outcome_fn, workflow),
                             train_dataset=Dataset.from_list(to_grpo_rows(df, workflow)),
                             args=GRPOConfig(learning_rate=hp.get("lr", 1e-5), num_generations=hp.get("group_size", 8),
                                             max_steps=hp.get("steps", 200), **{k: v for k, v in common.items() if k != "max_steps"}))
        else:
            raise ValueError(f"llm backend has no technique {tech!r}")
        tr.train()
        m = HFModel(tr.model, tok, workflow)
        m.base_id = recipe.base
        secs = time.time() - t0
        return TrainResult(m, secs / 60.0, secs)   # compute unit = wall-clock minutes (multiply by $/min)

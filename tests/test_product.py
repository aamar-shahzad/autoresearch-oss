import json
import os
import subprocess
import sys
import threading
import urllib.request

import pandas as pd
import pytest

from autoresearch_oss import builtins  # noqa: F401
from autoresearch_oss.hardware import (GPU, HardwareProfile, detect, parse_cgroup_limit, parse_cuda_version,
                                       parse_nvidia_smi, parse_rocm_smi)
from autoresearch_oss.models import load_catalog, plan, recommend
from autoresearch_oss.models.catalog import filter_entries, find
from autoresearch_oss.models.chooser import choose, render
from autoresearch_oss.models.hub import HubClient, HubError, local_models, verify_local
from autoresearch_oss.ontology import Registry, Workflow
from autoresearch_oss.plugins import available, get, register
from autoresearch_oss.project import init_project, list_templates, validate_data
from autoresearch_oss.sample.cnc import WORKFLOW, gen_episodes
from autoresearch_oss.serve import Service, load_live_log
from autoresearch_oss.verifiers import VerifierOutcome

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hw(vram=None, vendor="nvidia", cc="8.9", acc="cuda", n=1, ram=64, libs=None, disk=500):
    g = [GPU(i, "TestGPU", vendor, vram, vram - 0.5, cc) for i in range(n)] if vram else []
    return HardwareProfile("Linux", "x86_64", "cpu", 8, 16, ram, ram - 4, disk, "/", g, acc if vram else "cpu",
                           libs=libs or {})


# ------------------------------------------------------------- hardware ----------------
def test_nvidia_parser_multi_gpu_and_cuda_version():
    g = parse_nvidia_smi("0, NVIDIA GeForce RTX 4090, 24564, 23800, 8.9, 550.54\n1, Tesla T4, 15360, 15000, 7.5, 550.54")
    assert [x.name for x in g] == ["NVIDIA GeForce RTX 4090", "Tesla T4"]
    assert abs(g[0].vram_gb - 23.99) < 0.05 and g[0].bf16 and not g[1].bf16      # T4 has no bf16
    assert parse_cuda_version("| NVIDIA-SMI 550  Driver Version: 550  CUDA Version: 12.4 |") == "12.4"
    assert parse_nvidia_smi("garbage\n") == []


def test_rocm_and_cgroup_parsers():
    j = json.dumps({"card0": {"Card Series": "Instinct MI210", "VRAM Total Memory (B)": str(64 * 1024 ** 3),
                              "VRAM Total Used Memory (B)": str(1024 ** 3)}, "system": {}})
    g = parse_rocm_smi(j)
    assert g[0].vendor == "amd" and round(g[0].vram_gb) == 64 and round(g[0].free_gb) == 63
    assert parse_rocm_smi("not json") == []
    assert parse_cgroup_limit("max\n") is None and abs(parse_cgroup_limit(str(8 * 1024 ** 3)) - 8) < 1e-6


def test_detect_real_machine_is_sane():
    h = detect()
    assert h.ram_gb > 0 and h.cores_logical >= 1 and h.disk_free_gb > 0
    assert h.accelerator in ("cpu", "cuda", "rocm", "mps") and "torch" in h.libs
    assert "OS" in h.summary()


# -------------------------------------------------------------- catalog / fit -----------
def test_catalog_integrity():
    cat = load_catalog()
    ids = [e.id for e in cat]
    assert len(ids) == len(set(ids)) and len(ids) >= 30
    assert all(e.license_class in ("permissive", "custom", "research") and e.params_b > 0 for e in cat)
    assert all(e.gated for e in cat if e.license_class == "custom" and e.family.startswith(("Llama", "Gemma")))
    assert find(cat, "Qwen3-8B").id == "Qwen/Qwen3-8B"
    assert all(e.license_class == "permissive" for e in filter_entries(cat, license_class="permissive"))


def test_user_catalog_extension(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"models": [{"id": "acme/Tiny-1B", "family": "Acme", "params_b": 1.0, "license_id": "apache-2.0",
                                          "license_class": "permissive"}]}))
    assert find(load_catalog([str(p)]), "acme/Tiny-1B")


def test_planner_scales_with_hardware():
    cat = load_catalog()
    e = find(cat, "Qwen3-8B")
    assert plan(e, hw(24), 2048).best in ("lora", "qlora")
    assert plan(e, hw(24), 2048).mode("full").verdict == "no"          # 16 B/param can't fit
    small = plan(e, hw(12))                                          # 8B @ 2048 tokens: QLoRA only
    assert small.best == "qlora" and small.mode("lora").verdict == "no"
    assert plan(e, hw(8)).best is None                               # honest: not at 2048 tokens on 8 GB
    assert plan(find(cat, "Qwen3-32B"), hw(12)).best is None
    assert plan(e, hw(None)).best is None                              # CPU + 8B: impractical
    assert plan(find(cat, "SmolLM2-135M-Instruct"), hw(None, ram=16), seq=512).best == "lora"   # CPU smoke test ok


def test_planner_platform_rules():
    e = find(load_catalog(), "Qwen3-8B")
    mac = plan(e, hw(24, vendor="apple", acc="mps"))
    assert mac.mode("qlora").verdict == "no" and any("bitsandbytes" in c for c in mac.mode("qlora").caveats)
    t4 = plan(e, hw(16, cc="7.5"))
    assert any("bf16" in c for m in t4.modes for c in m.caveats if m.mode == "lora")
    multi = plan(find(load_catalog(), "Qwen3-14B"), hw(12, n=2))
    assert multi.best is not None and any("sharding" in c for m in multi.modes for c in m.caveats)


def test_planner_disk_and_missing_libs():
    e = find(load_catalog(), "Qwen3-8B")
    p = plan(e, hw(24, disk=5))
    assert not p.disk_ok and not p.trainable
    assert set(p.missing_libs) >= {"torch", "peft", "trl"}
    ok = plan(e, hw(24, libs={k: "1" for k in ("torch", "transformers", "peft", "trl", "bitsandbytes")}))
    assert ok.missing_libs == []


def test_recommend_orders_by_fit_then_license_then_size():
    ps = recommend(hw(24), load_catalog(), only_trainable=True)
    assert ps and all(p.best for p in ps)
    first = ps[0]
    assert first.entry.license_class == "permissive" and first.mode(first.best).verdict == "fits"
    assert recommend(hw(24), load_catalog(), license_class="permissive")[0].entry.license_class == "permissive"


def test_chooser_is_scriptable():
    plans = recommend(hw(24), load_catalog(), only_trainable=True)
    assert "model" in render(plans[:3])
    assert choose(plans, input_fn=lambda _: "2", print_fn=lambda *_: None) is plans[1]
    assert choose(plans, input_fn=lambda _: plans[0].entry.short, print_fn=lambda *_: None) is plans[0]
    seq = iter(["zzz", "q"])
    assert choose(plans, input_fn=lambda _: next(seq), print_fn=lambda *_: None) is None


# ------------------------------------------------------------------ hub ---------------
class _Sib:
    def __init__(self, f, size): self.rfilename, self.size = f, size


class _Model:
    def __init__(self, sha="abc123def", gated=False, private=False, lic="apache-2.0", files=None):
        self.sha, self.gated, self.private, self.tags, self.pipeline_tag = sha, gated, private, [], "text-generation"
        self.card_data = {"license": lic}
        self.siblings = files if files is not None else [_Sib("model.safetensors", 4 * 1024 ** 3), _Sib("config.json", 100)]


class _Api:
    def __init__(self, **kw): self.kw = kw
    def model_info(self, repo, revision=None, files_metadata=False): return _Model(**self.kw)
    def list_models(self, **kw): return [type("M", (), {"id": "a/b", "downloads": 5, "likes": 1, "gated": False})()]


def _dl(repo_id, revision, local_dir, allow_patterns, token):
    os.makedirs(local_dir, exist_ok=True)
    open(os.path.join(local_dir, "config.json"), "w").write("{}")
    open(os.path.join(local_dir, "model.safetensors"), "wb").write(b"0" * 1000)
    return local_dir


def test_hub_download_success_manifest_and_registry(tmp_path):
    reg = Registry()
    c = HubClient(api=_Api(), downloader=_dl, offline=False)
    m = c.download("org/Model", str(tmp_path), registry=reg)
    assert m["revision"] == "abc123def" and m["license_accepted"] and m["bytes"] == 1002
    assert reg.get(m["registry_id"])["payload"]["repo"] == "org/Model"
    lm = local_models(str(tmp_path))
    assert lm[0]["repo"] == "org/Model" and verify_local(lm[0]) == []
    open(os.path.join(m["path"], "model.safetensors"), "ab").write(b"x")
    assert any("size changed" in x for x in verify_local(local_models(str(tmp_path))[0]))


@pytest.mark.parametrize("kw,token,msg", [
    ({"gated": True}, None, "gated"),
    ({"private": True}, "t", "private"),
    ({"lic": "llama3.1"}, "t", "conditions"),
    ({"files": [_Sib("pytorch_model.bin", 10)]}, "t", "safetensors"),
])
def test_hub_refusals(tmp_path, kw, token, msg):
    c = HubClient(token=token, api=_Api(**kw), downloader=_dl, offline=False)
    with pytest.raises(HubError, match=msg):
        c.download("org/Model", str(tmp_path))


def test_hub_license_acceptance_and_disk_and_offline(tmp_path):
    c = HubClient(token="t", api=_Api(lic="gemma"), downloader=_dl, offline=False)
    assert c.download("org/M", str(tmp_path), accept_license=True)["license_accepted"]
    assert c.download("org/M2", str(tmp_path / "b"), confirm=lambda i: True)
    big = HubClient(api=_Api(files=[_Sib("m.safetensors", 10 ** 15)]), downloader=_dl, offline=False)
    with pytest.raises(HubError, match="disk"):
        big.download("org/Huge", str(tmp_path / "c"))
    with pytest.raises(HubError, match="offline"):
        HubClient(offline=True).search("x")
    assert HubClient(api=_Api(), offline=False).search("x")[0]["repo"] == "a/b"


# -------------------------------------------------------------- plugins ---------------
def test_plugin_registration_and_lookup():
    @register("outcome", "always_true_test")
    def f(spec):
        return lambda df, acts: [1.0] * len(df)
    assert "always_true_test" in available("outcome")["outcome"]
    assert get("outcome", "always_true_test")({})(pd.DataFrame({"a": [1, 2]}), ["x", "y"]) == [1.0, 1.0]
    with pytest.raises(KeyError, match="available"):
        get("outcome", "nope")
    with pytest.raises(ValueError):
        register("bogus", "x")


# ------------------------------------------------------------ verifiers ---------------
def test_verifiers():
    df = pd.DataFrame({"action": ['{"invoice_number": "A-1", "total": "10.00", "currency": "EUR"}']})
    v = lambda spec: VerifierOutcome(get("verifier", spec["type"])(spec))
    good = '```json\n{"invoice_number": "a-1", "total": "10.00", "currency": "EUR"}\n```'
    assert v({"type": "json_fields", "required": ["total"], "match": ["invoice_number", "total", "currency"]})(df, [good])[0] == 1.0
    partial = '{"invoice_number": "A-1", "total": "99", "currency": "EUR"}'
    assert abs(v({"type": "json_fields", "required": ["total"], "match": ["invoice_number", "total", "currency"]})(df, [partial])[0] - 2 / 3) < 1e-9
    assert v({"type": "json_fields", "required": ["total"]})(df, ["not json"])[0] == 0.0
    assert v({"type": "exact"})(pd.DataFrame({"action": ["Hello  World"]}), ["hello world"])[0] == 1.0
    assert v({"type": "regex", "pattern": r"\bDROP\s+TABLE"})(df, ["please drop   table x"])[0] == 1.0


def test_generation_workflow_validation_and_prompts():
    from autoresearch_oss.backends import llm
    wf = Workflow.from_dict({"name": "x", "description": "Extract.", "inputs": ["document"], "outcome_fields": [],
                             "action_space": [], "unsafe_actions": [], "slice_by": "t", "thresholds": {},
                             "task_type": "generation", "prompt_template": "DOC: {document}"})
    row = {"document": "hello", "action": '{"a": 1}'}
    assert llm.format_prompt(row, wf).endswith("Output:") and "DOC: hello" in llm.format_prompt(row, wf)
    df = pd.DataFrame([{"document": "d1", "action": "good", "outcome_success": True},
                       {"document": "d1", "action": "bad", "outcome_success": False}])
    pairs = llm.to_dpo_pairs(df, wf)
    assert pairs[0]["chosen"] == " good" and pairs[0]["rejected"] == " bad"
    assert llm.to_grpo_rows(df, wf)[0]["reference"] == "good" and len(llm.to_grpo_rows(df, wf)) == 1
    with pytest.raises(ValueError):
        Workflow.from_dict({"name": "x", "description": "", "inputs": [], "outcome_fields": [], "action_space": [],
                            "unsafe_actions": [], "slice_by": "t", "thresholds": {}})  # choice with no actions
    r = llm.make_reward_fn(VerifierOutcome(get("verifier", "exact")({})), wf)
    assert r(["p"], ["good"], document=["d1"], reference=["good"]) == [1.0]


# ---------------------------------------------------- templates / validate -------------
def test_templates_scaffold_and_validate(tmp_path):
    assert set(list_templates()) >= {"maintenance-diagnostics", "support-triage", "structured-extraction"}
    for t in list_templates():
        d = tmp_path / t
        files = init_project(str(d), t)
        cfg = json.load(open(files[0]))
        Workflow.from_dict(cfg["workflow"])                       # every template is a valid workflow
        assert os.path.exists(d / "data" / "episodes.example.csv")


def test_validate_data_flags_problems(tmp_path):
    d = tmp_path / "p"
    init_project(str(d), "maintenance-diagnostics")
    gen_episodes(2000, seed=1).drop(columns=["flow_after", "temp_after"]).to_csv(d / "data" / "episodes.csv", index=False)
    cfg = json.load(open(d / "autoresearch.json"))
    cfg["data"]["path"] = str(d / "data" / "episodes.csv")
    r = validate_data(cfg)
    assert r["problems"] == [] and r["cleaning"]["output_rows"] == 2000
    assert any("thin slices" in w for w in r["warnings"])          # the rare pump slice is flagged up front


# ---------------------------------------------------------- serving + feedback ----------
def test_http_service_gate_feedback_and_ingest(tmp_path):
    import joblib
    from autoresearch_oss.backends import TabularBackend
    from autoresearch_oss.mixtures import MixtureSpec
    from autoresearch_oss.techniques import Recipe
    hist = gen_episodes(3000, seed=0)
    res = TabularBackend().train(Recipe("hgb", "imitate", MixtureSpec("a"), {"max_iter": 30}), hist, WORKFLOW)
    log = str(tmp_path / "live.jsonl")
    svc = Service(res.model, "model/test@v1", WORKFLOW, log_path=log, api_key="s3cret")
    srv = svc.make_server("127.0.0.1", 0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def call(path, body=None, key="s3cret"):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert call("/health")[0] == 200
        assert call("/v1/model", key="wrong")[0] == 401
        row = hist.iloc[0]
        inputs = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in WORKFLOW.inputs}
        inputs["asset"] = "CNC-999"
        st, out = call("/v1/predict", {"inputs": inputs})
        assert st == 200 and out["action"] in WORKFLOW.action_space and "needs_approval" in out
        assert call("/v1/predict", {"inputs": {"flow": 1}})[0] == 400              # missing inputs rejected
        st, fb = call("/v1/feedback", {"request_id": out["request_id"], "final_action": "clean_fan",
                                       "outcome_success": True, "slice": "fan"})
        assert st == 200 and fb["status"] in ("accepted", "overridden")
        assert call("/v1/feedback", {"request_id": out["request_id"], "final_action": "x", "outcome_success": True})[0] == 400
    finally:
        srv.shutdown()
    live = load_live_log(log, WORKFLOW)
    assert len(live) == 1 and live.iloc[0]["source"] == "live" and live.iloc[0]["fault_family"] == "fan"
    assert {json.loads(l)["type"] for l in open(log)} == {"prediction", "episode"}


# -------------------------------------------------------------------- CLI -------------
def _cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "autoresearch_oss", *args], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=ROOT), cwd=cwd)


def test_cli_doctor_models_init():
    r = _cli("doctor", "--json")
    assert r.returncode == 0 and "ram_gb" in json.loads(r.stdout)
    r = _cli("models", "list", "--all", "--limit", "3", "--seq", "512")
    assert r.returncode == 0 and "SmolLM2" in r.stdout and "Hardware:" in r.stdout
    r = _cli("models", "info", "Qwen3-8B")
    assert r.returncode == 0 and "lora" in r.stdout and "qlora" in r.stdout
    assert _cli("models", "choose", "--select", "1", "--all", "--seq", "512").returncode == 0
    assert "maintenance-diagnostics" in _cli("init", "list").stdout


def test_cli_train_preflight_blocks_unfit_llm(tmp_path):
    p = tmp_path / "proj"
    _cli("init", "support-triage", str(p))
    pd.DataFrame(json.load(open(f"{ROOT}/autoresearch_oss/templates/support-triage.json"))["example_rows"]).to_csv(p / "data" / "episodes.csv", index=False)
    cfg = json.load(open(p / "autoresearch.json"))
    cfg["backend"]["model"] = {"id": "Qwen/Qwen3-32B"}
    cfg["backend"]["allow_hub_download"] = True
    cfg["data"]["path"] = str(p / "data" / "episodes.csv")
    json.dump(cfg, open(p / "autoresearch.json", "w"))
    r = _cli("train", "--config", str(p / "autoresearch.json"), "--db", str(tmp_path / "r.db"), "--out", str(tmp_path / "o"))
    assert r.returncode == 1 and "preflight FAILED" in r.stdout


def test_example_plugin_registers():
    import importlib
    import sys as _s
    _s.path.insert(0, f"{ROOT}/examples/plugin_example")
    importlib.import_module("mycorp_autoresearch")
    a = available()
    assert "mes" in a["connector"] and "sql_check" in a["outcome"]
    assert get("outcome", "sql_check")({}).verifiable

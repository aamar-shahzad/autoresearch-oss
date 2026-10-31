"""Will this model train on THIS machine? A memory planner, not an oracle.

All numbers are conservative engineering estimates (roughly +/-25%): weights, LoRA/optimizer
state, a logits+activation term (gradient checkpointing assumed), and CUDA context. They are
meant to rule models in or out and pick a training mode, not to replace a dry run."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..hardware import HardwareProfile
from .catalog import Entry

Q4 = 0.55           # bytes per param for 4-bit incl. quantisation constants
RESERVE = 0.92      # fraction of device memory we are willing to plan against


def activation_gb(e: Entry, seq: int, batch: int) -> float:
    tokens = seq * batch
    body = 0.35 * (e.params_b ** 0.55) * tokens / 2048          # per-layer activations w/ checkpointing
    logits = e.vocab * tokens * 8 / 1e9                          # fp32 logits + grad
    return body + logits


def required_gb(e: Entry, mode: str, seq: int = 2048, batch: int = 1) -> float:
    P = e.params_b
    act = activation_gb(e, seq, batch)
    return {
        "full": 16 * P + act + 2.0,                  # bf16 weights+grads, fp32 master + Adam m,v
        "lora": 2.0 * P + 0.03 * P + act + 1.5,
        "qlora": Q4 * P + 0.03 * P + act + 1.8,
        "infer-bf16": 2.0 * P + 0.05 * P * seq / 2048 + 1.0,
        "infer-int4": Q4 * P + 0.05 * P * seq / 2048 + 0.8,
    }[mode]


@dataclass
class ModePlan:
    mode: str
    required_gb: float
    available_gb: float
    verdict: str                       # fits | tight | no
    caveats: list = field(default_factory=list)


@dataclass
class Plan:
    entry: Entry
    modes: list
    best: str | None
    disk_needed_gb: float
    disk_ok: bool
    missing_libs: list
    where: str                         # which device the plan targets
    inference: ModePlan | None = None

    @property
    def trainable(self) -> bool:
        return self.best is not None and self.disk_ok

    def mode(self, name: str) -> ModePlan | None:
        return next((m for m in self.modes if m.mode == name), None)


def _verdict(req: float, avail: float) -> str:
    if avail <= 0 or req > avail:
        return "no"
    return "fits" if req <= 0.85 * avail else "tight"


def plan(e: Entry, hw: HardwareProfile, seq: int = 2048, batch: int = 1, modes=("lora", "qlora", "full")) -> Plan:
    has_gpu = bool(hw.gpus)
    n_gpu = len(hw.gpus)
    if has_gpu:
        g = max(hw.gpus, key=lambda x: x.vram_gb)
        avail = min(g.vram_gb * RESERVE, max(g.free_gb, 0.0) * 0.98 + 0.0)
        where = f"{g.name} ({g.vram_gb:.0f} GB)" + ("" if n_gpu == 1 else f" x{n_gpu}")
    else:
        avail = hw.ram_available_gb * 0.70
        where = f"CPU / {hw.ram_gb:.0f} GB RAM"

    out = []
    for m in modes:
        req = required_gb(e, m, seq, batch)
        if not has_gpu:
            req = max(0.5, req - 1.0)     # no CUDA context on CPU
        caveats = []
        a = avail
        v = _verdict(req, a)
        if v == "no" and n_gpu > 1 and m in ("lora", "qlora"):
            pooled = hw.pooled_memory_gb * 0.85
            if req <= pooled:
                v, a = "tight", pooled
                caveats.append(f"needs model sharding across {n_gpu} GPUs (device_map='auto'): works, but slow")
        if not has_gpu:
            if m != "lora":
                v = "no"
                caveats.append("needs a GPU")
            elif e.params_b > 3.0:
                v = "no"
                caveats.append("CPU training above ~3B parameters is impractical")
            else:
                caveats.append("CPU training is very slow; use only for small models / smoke tests")
                if v == "fits":
                    v = "tight"
        elif hw.accelerator == "mps" and m == "qlora":
            v = "no"
            caveats.append("bitsandbytes is unavailable on Apple Silicon; use bf16 LoRA (or MLX-LM 4-bit LoRA)")
        elif hw.accelerator == "rocm" and m == "qlora":
            caveats.append("bitsandbytes on ROCm is experimental")
        if has_gpu and not hw.gpus[0].bf16 and hw.accelerator == "cuda" and m in ("lora", "full"):
            caveats.append("GPU lacks bf16 (compute capability < 8.0): training will use fp16")
        if e.active_params_b and m in ("lora", "qlora", "full"):
            caveats.append("MoE: memory counts total parameters; adapter targeting differs, verify support")
        if m == "full" and v != "no":
            caveats.append("full fine-tuning rarely beats LoRA on small datasets and is far more fragile")
        out.append(ModePlan(m, round(req, 1), round(a, 1), v, caveats))

    best = next((m.mode for m in out if m.mode in ("lora", "qlora") and m.verdict == "fits"), None) \
        or next((m.mode for m in out if m.mode in ("lora", "qlora") and m.verdict == "tight"), None)

    need = ["torch", "transformers", "peft", "trl"]
    if best == "qlora" or (best is None and out and any(m.mode == "qlora" and m.verdict != "no" for m in out)):
        need.append("bitsandbytes")
    missing = [n for n in need if not hw.lib(n)]
    disk_needed = 2.0 * e.params_b * 1.1 + 1.0
    inf = None
    for im in ("infer-bf16", "infer-int4"):
        req = required_gb(e, im, seq, 1)
        v = _verdict(req, avail if has_gpu else hw.ram_available_gb * 0.7)
        inf = ModePlan(im, round(req, 1), round(avail if has_gpu else hw.ram_available_gb * 0.7, 1), v)
        if v != "no":
            break
    return Plan(e, out, best, round(disk_needed, 1), hw.disk_free_gb >= disk_needed, missing, where, inf)


_RANK = {"fits": 0, "tight": 1, "no": 2}


def recommend(hw: HardwareProfile, entries, seq: int = 2048, batch: int = 1, license_class=None,
              only_trainable: bool = False) -> list:
    plans = [plan(e, hw, seq, batch) for e in entries
             if not license_class or e.license_class in (license_class if isinstance(license_class, (list, tuple, set)) else [license_class])]
    if only_trainable:
        plans = [p for p in plans if p.trainable]

    def key(p: Plan):
        best = p.mode(p.best) if p.best else None
        if not best:        # not trainable here: show the closest-to-feasible (smallest) first
            return (3, 0 if p.entry.license_class == "permissive" else 1, p.entry.params_b)
        return (_RANK[best.verdict], 0 if p.entry.license_class == "permissive" else 1, -p.entry.params_b)
    return sorted(plans, key=key)

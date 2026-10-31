"""Detect what this machine can actually run -- before the user picks a model.

Detection is read-only and defensive: every probe is optional and failure degrades to
'unknown' rather than crashing. Parsers are pure functions so they can be tested against
captured tool output from machines we don't own."""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

GB = 1024 ** 3
LIBS = ["torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes",
        "huggingface_hub", "vllm", "flash_attn", "llama_cpp_python", "mlx", "mlx_lm", "scikit-learn"]


@dataclass
class GPU:
    index: int
    name: str
    vendor: str                 # nvidia | amd | apple
    vram_gb: float              # total (apple: usable share of unified memory)
    free_gb: float
    compute_capability: str = ""
    driver: str = ""

    @property
    def bf16(self) -> bool:
        if self.vendor == "nvidia" and self.compute_capability:
            try:
                return float(self.compute_capability) >= 8.0
            except ValueError:
                return False
        return self.vendor in ("amd", "apple")


@dataclass
class HardwareProfile:
    os: str
    arch: str
    cpu_model: str
    cores_physical: int
    cores_logical: int
    ram_gb: float
    ram_available_gb: float
    disk_free_gb: float
    disk_path: str
    gpus: list = field(default_factory=list)
    accelerator: str = "cpu"            # cuda | rocm | mps | cpu
    cuda_version: str = ""
    unified_memory: bool = False
    in_container: bool = False
    libs: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # ---- memory the planner may use ------------------------------------------
    @property
    def device_memory_gb(self) -> float:
        """Memory on the single best accelerator (data-parallel does not pool VRAM)."""
        if self.gpus:
            return max(g.vram_gb for g in self.gpus)
        return 0.0

    @property
    def pooled_memory_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    def lib(self, name: str) -> str | None:
        return self.libs.get(name)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        L = [f"OS            {self.os} ({self.arch})" + ("  [container]" if self.in_container else ""),
             f"CPU           {self.cpu_model}  ({self.cores_physical} physical / {self.cores_logical} logical cores)",
             f"RAM           {self.ram_gb:.1f} GB total, {self.ram_available_gb:.1f} GB available",
             f"Disk          {self.disk_free_gb:.0f} GB free at {self.disk_path}",
             f"Accelerator   {self.accelerator}" + (f"  (CUDA {self.cuda_version})" if self.cuda_version else "")
             + ("  [unified memory]" if self.unified_memory else "")]
        if self.gpus:
            for g in self.gpus:
                L.append(f"  GPU {g.index}      {g.name}  {g.vram_gb:.1f} GB ({g.free_gb:.1f} free)"
                         + (f"  cc {g.compute_capability}" if g.compute_capability else "")
                         + ("  bf16" if g.bf16 else ""))
        else:
            L.append("  GPU         none detected")
        have = [f"{k} {v}" for k, v in self.libs.items() if v]
        miss = [k for k, v in self.libs.items() if not v and k in ("torch", "transformers", "peft", "trl")]
        L.append("Libraries     " + (", ".join(have) if have else "none of the ML stack installed"))
        if miss:
            L.append("Missing       " + ", ".join(miss) + '   ->  pip install "autoresearch-oss[llm]"')
        for n in self.notes:
            L.append(f"note: {n}")
        return "\n".join(L)


# ---------------------------------------------------------------- pure parsers ----
def parse_nvidia_smi(csv_text: str) -> list:
    """Input from: nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version
                   --format=csv,noheader,nounits   (memory in MiB)"""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        try:
            total, free = float(parts[2]) / 1024, float(parts[3]) / 1024
        except ValueError:
            continue
        gpus.append(GPU(int(parts[0]), parts[1], "nvidia", round(total, 2), round(free, 2),
                        parts[4] if len(parts) > 4 else "", parts[5] if len(parts) > 5 else ""))
    return gpus


def parse_cuda_version(nvidia_smi_header: str) -> str:
    m = re.search(r"CUDA Version:\s*([\d.]+)", nvidia_smi_header)
    return m.group(1) if m else ""


def parse_rocm_smi(json_text: str) -> list:
    """Best effort over `rocm-smi --showproductname --showmeminfo vram --json` (format varies by version)."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return []
    gpus = []
    for key, v in data.items():
        m = re.match(r"card(\d+)", key)
        if not m or not isinstance(v, dict):
            continue
        total = next((float(x) for k, x in v.items() if "Total Memory" in k and "VRAM" in k), 0.0)
        used = next((float(x) for k, x in v.items() if "Total Used Memory" in k and "VRAM" in k), 0.0)
        name = v.get("Card Series") or v.get("Card series") or v.get("Card model") or "AMD GPU"
        gpus.append(GPU(int(m.group(1)), str(name), "amd", round(total / GB, 2), round((total - used) / GB, 2)))
    return gpus


def parse_cgroup_limit(text: str) -> float | None:
    t = text.strip()
    if not t or t == "max":
        return None
    try:
        v = int(t)
    except ValueError:
        return None
    return v / GB if v < 1 << 60 else None


# --------------------------------------------------------------------- probes ----
def _run(cmd: list, timeout: int = 8) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if out:
            return out
    return platform.processor() or platform.machine()


def _memory() -> tuple:
    try:
        import psutil
        vm = psutil.virtual_memory()
        total, avail = vm.total / GB, vm.available / GB
        phys, logi = psutil.cpu_count(logical=False) or 0, psutil.cpu_count() or 0
    except ImportError:
        total = avail = 0.0
        phys = logi = os.cpu_count() or 1
        try:
            for line in open("/proc/meminfo"):
                k, v = line.split(":")
                if k == "MemTotal":
                    total = int(v.split()[0]) * 1024 / GB
                if k == "MemAvailable":
                    avail = int(v.split()[0]) * 1024 / GB
        except OSError:
            pass
    container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            lim = parse_cgroup_limit(open(p).read())
        except OSError:
            lim = None
        if lim:
            total, avail = min(total, lim), min(avail, lim)
            container = True
            break
    return total, avail, phys or logi, logi, container


def _libs() -> dict:
    out = {}
    for name in LIBS:
        try:
            out[name] = md.version(name.replace("_", "-") if name == "huggingface_hub" else name)
        except md.PackageNotFoundError:
            try:
                out[name] = md.version(name.replace("_", "-"))
            except md.PackageNotFoundError:
                out[name] = None
    return out


def detect(disk_path: str | None = None) -> HardwareProfile:
    disk_path = disk_path or os.path.expanduser("~")
    total, avail, phys, logi, container = _memory()
    try:
        free = shutil.disk_usage(disk_path).free / GB
    except OSError:
        free = 0.0
    hp = HardwareProfile(platform.system(), platform.machine(), _cpu_model(), phys, logi,
                         round(total, 2), round(avail, 2), round(free, 1), disk_path,
                         in_container=container, libs=_libs())
    # NVIDIA
    if shutil.which("nvidia-smi"):
        gpus = parse_nvidia_smi(_run(["nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
            "--format=csv,noheader,nounits"]))
        if gpus:
            hp.gpus, hp.accelerator = gpus, "cuda"
            hp.cuda_version = parse_cuda_version(_run(["nvidia-smi"]))
    # AMD
    if not hp.gpus and shutil.which("rocm-smi"):
        gpus = parse_rocm_smi(_run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"]))
        if gpus:
            hp.gpus, hp.accelerator = gpus, "rocm"
            hp.notes.append("ROCm detected: bitsandbytes/QLoRA support on AMD is experimental.")
    # Apple Silicon (unified memory; macOS lets the GPU wire roughly 2/3-3/4 of RAM by default)
    if not hp.gpus and platform.system() == "Darwin" and platform.machine() == "arm64":
        share = 0.75 if total >= 36 else 0.66
        hp.gpus = [GPU(0, _cpu_model() or "Apple Silicon", "apple", round(total * share, 2),
                       round(avail * share, 2))]
        hp.accelerator, hp.unified_memory = "mps", True
        hp.notes.append("Apple Silicon: PyTorch MPS has no bitsandbytes; use LoRA in bf16/fp16, or MLX-LM for 4-bit LoRA.")
    if hp.in_container and hp.gpus:
        hp.notes.append("Running in a container: confirm the GPU is exposed (--gpus all) and memory limits are intended.")
    if not hp.gpus:
        hp.notes.append("No GPU: only inference and very small-model LoRA are practical (slow).")
    if hp.libs.get("torch") and hp.accelerator in ("cuda", "rocm"):
        hp.notes.append("Torch is installed; run `autoresearch doctor --probe` to confirm it can see the GPU.")
    return hp


def probe_torch() -> dict:
    """Run in a subprocess so a broken CUDA install can't crash the CLI."""
    code = ("import json,torch;d={'torch':torch.__version__,'cuda':torch.cuda.is_available(),"
            "'hip':bool(getattr(torch.version,'hip',None)),'mps':bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()),"
            "'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]};print(json.dumps(d))")
    out = _run([os.sys.executable, "-c", code], timeout=60)
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "torch not importable"}

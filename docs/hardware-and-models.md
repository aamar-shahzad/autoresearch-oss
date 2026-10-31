# Hardware detection and choosing a model

```bash
autoresearch doctor            # what this machine has: CPU, RAM (container limits honoured), disk, NVIDIA/AMD/Apple GPUs, ML libs
autoresearch doctor --probe    # also imports torch in a subprocess to confirm it can really use the GPU
autoresearch models list       # open models that can be fine-tuned HERE, best fit first
autoresearch models info Qwen3-8B
autoresearch models choose --config autoresearch.json --pull   # interactive pick -> writes the config -> downloads
```

## What is detected
| | how |
|---|---|
| NVIDIA | `nvidia-smi --query-gpu` (name, total/free VRAM, compute capability, driver) + CUDA version |
| AMD | `rocm-smi --json` (best effort; output differs across ROCm versions) |
| Apple Silicon | unified memory; the GPU may wire ~66-75% of RAM by default |
| RAM | psutil, clamped by cgroup limits (Docker/Kubernetes) |
| Libraries | torch, transformers, peft, trl, bitsandbytes, vllm, mlx... via package metadata |

Not detected: Intel GPUs/NPUs, TPUs, remote/multi-node clusters, GPU sharing (MIG/vGPU) specifics.
If detection is wrong for your setup, report the `autoresearch doctor --json` output.

## How "fits" is decided
`models/fit.py` estimates memory for each training mode:

| mode | memory model |
|---|---|
| `full` | 16 B/param (bf16 weights+grads, fp32 master, Adam) + activations |
| `lora` | 2 B/param weights + adapters/optimizer + activations |
| `qlora` | ~0.55 B/param (4-bit) + adapters + activations (needs bitsandbytes: CUDA, experimental on ROCm, unavailable on Apple) |

Activations = per-layer term (gradient checkpointing assumed) + fp32 logits (`vocab x tokens x 8 B`), which is why
big-vocabulary models need more memory at long sequence lengths. Verdicts: `fits` (<=85% of usable memory),
`tight` (<=100%), `no`. Usable memory = 92% of VRAM. Multi-GPU: if one card is too small, LoRA/QLoRA can
shard the model across cards (slow); VRAM is not pooled for data-parallel training.

**These are conservative estimates, roughly +/-25%.** Shorter `--seq`, `--batch 1`, a smaller `lora_r`, or
frameworks with chunked-loss kernels can fit models the planner marks `no`. Always do a one-step dry run.

## The catalog
`models/catalog.json` is a curated **snapshot** (see its `snapshot` field): ids, approximate parameter counts, vocab,
license, gating. It will go stale. Ways to stay current without code changes:
- add entries in `~/.autoresearch/catalog.json` (same schema) - merged over the built-in list;
- `autoresearch models search <text>` queries the live Hub; `models info` shows live license/gating/size/revision.

License classes: `permissive` (Apache-2.0/MIT), `custom` (Llama/Gemma/other terms with conditions: attribution,
acceptable-use, scale thresholds - read them), `research` (non-commercial). `models list` defaults to permissive+custom.

## Downloading safely
`models pull` refuses: private repos, repos without `.safetensors` (pickle `.bin` can execute code), gated repos with no
`HF_TOKEN`, non-permissive licenses unless you confirm (`--accept-license` or an interactive prompt), and downloads
that would not fit on disk. It pins the commit SHA, stores a manifest (`.autoresearch.json`) and records a `basemodel`
artifact in the registry. `models verify <repo>` re-checks file sizes. Gated/licensed models are never fetched
silently.

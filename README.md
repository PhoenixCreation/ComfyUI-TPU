# ComfyUI on TPU

> **TPU-enabled fork of [ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — run diffusion (and soon video/audio) models on Google Cloud TPUs through the standard ComfyUI node graph.
>
> For anything about core ComfyUI (nodes, workflows, custom nodes, desktop app, API, installation on CUDA/ROCm/MPS), see the **upstream README**: [`docs/reference/comfyui-upstream-README.md`](docs/reference/comfyui-upstream-README.md) or online at [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI/blob/master/README.md).

[![ComfyUI TPU](https://img.shields.io/badge/TPU-v5e--8%20%E2%80%A2%20Krea2%20Turbo-4285F4?style=flat)](#progress)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?style=flat)](#how-to-setup)
[![torch-xla 2.8](https://img.shields.io/badge/torch--xla-2.8.0-orange?style=flat)](#how-to-setup)

---

## Overview

This project adds **Google Cloud TPU** as a first-class execution backend for ComfyUI.

**Long-term goal:** support many models and modalities on TPU — image generation, editing, video, upscaling, and audio — through the existing ComfyUI graph and API, with one launch path, one validator, and one operability contract per profile.

**Design principles:**

- **Gated, not forked** — every TPU code path is behind `--tpu`; without it the process is upstream ComfyUI verbatim. The only module that imports `torch_xla` is `comfy/xla_backend.py`.
- **Profile-per-phase** — each model lands as a single frozen profile (fixed shapes/dtype/sampler) before being generalized. One profile compilable once is more reliable than many shapes compiled on demand.
- **Fail closed** — artifact digest mismatch, unsupported node, or startup compile failure keeps the server non-ready; the queue never serves a partial profile.

See [docs/architecture/overview.md](docs/architecture/overview.md) for the system layout and [docs/architecture/decisions/001-krea2-tpu-support.md](docs/architecture/decisions/001-krea2-tpu-support.md) for the Phase 1 decision record.

## Progress

> Detailed tracker (T5e-8 unless noted): **[docs/progress.md](docs/progress.md)**

| Stage | What |
|---|---|
| **Already done** | **Krea2 Turbo**, fixed `1920×1080`, batch 1, 8 steps, `er_sde`/`simple`, CFG 1.0 — Krea2 diffusion + Qwen3-VL-4B text encoder + Qwen Image VAE — on a single-controller **T5e-8** slice. Verified end-to-end 2026-08-16 (`output/krea2_automatic_00001_.png`, RGB 1920×1080). Warm-up 105.4s, steady-state 32.7s. 101 headless unit tests. |
| **In progress** | Krea2 — **multi-dimension** support (lift the fixed 1920×1080 constraint). |
| **Roadmap** | **Ideogram 4** · **Nvidia PiD upscaling** · **MiniMax H3** · **Other clusters** (beyond v5e-8). |

> **Constraint:** Phase 1 only supports the `krea2-1920x1080` profile, T5e-8, BF16, no LoRA/ControlNet/patches. Prompt text and seed are free; every shape-affecting value is validated.

## How to Setup

> Full operator guide: **[docs/deployment.md](docs/deployment.md)** (picks up all of the former `deployment/README.md`).

Run from the repo root (`ComfyUI-TPU/`):

```bash
# 1. Install the exact pinned env — torch and torch-xla MUST stay on the same line.
python -m pip install -r deployment/requirements-tpu.txt

# 2. Stage the three 33 GiB BF16 artifacts as symlinks.
export KREA2_MODEL_ROOT=/kaggle/input/models/helltester2/krea2-bf16/transformers/default/1/models
deployment/stage_models.sh

# 3. (Only after changing artifacts) re-pin SHA-256 into the manifest:
# python deployment/hash_artifacts.py

# 4. Fix PJRT/topology BEFORE the first torch_xla import, pick a writable cache dir, launch.
source deployment/tpu_env.sh
export TPU_CACHE_DIR=/kaggle/working/tpu/.tpu-cache-krea2
deployment/launch.sh
```

What that does: verifies artifact digests against `deployment/model_manifest.json`, creates an 8-device `model` mesh via `torch_xla` SPMD, compiles the canonical workflow (`workflows/Krea2-turbo-tpu.json`) once, and opens the API only when `ready`. Check readiness:

```bash
deployment/healthcheck.sh                  # exits 0 only on ready
curl http://127.0.0.1:8188/tpu/status | jq # state, mesh, artifact_hashes, fields
```

Submit the profile (from another shell) and poll `/history/<prompt_id>` for `status_str: success`:

```bash
python - <<'PY'
import json, urllib.request
with open("workflows/Krea2-turbo-tpu.json") as f:
    prompt = json.load(f)
req = urllib.request.Request("http://127.0.0.1:8188/prompt",
    data=json.dumps({"prompt": prompt}).encode(),
    headers={"Content-Type":"application/json"})
print(urllib.request.urlopen(req).read().decode())
PY
```

Flags: `--tpu` requires `--tpu-cache-dir`. CUDA/DirectML/VRAM/precision/attention/fast flags are rejected at parse time. `--bf16-*` is accepted (redundant). `--no-tpu-warmup` skips startup compile (first request pays it). See `tests/tpu/test_cli_tpu.py` for the full matrix.

## What You Should Know

- **Non-TPU backends untouched.** CUDA / ROCm / MPS / XPU / NPU / MLU / DirectML are gated — run without `--tpu` and this repo behaves like upstream.
- **BF16-only, T5e-8-only (for now).** The profile pins all compute to BF16; the mesh is exactly 8 devices. Other resolutions, clusters, and dtypes are on the roadmap, not in Phase 1.
- **Prompt freedom:** any text + any seed; dimensions, steps, sampler, scheduler, CFG, and save prefix are fixed and validated pre-queue (`tpu_profile_*` errors).
- **No runtime patching.** LoRA, ControlNet, model-patcher hooks/wrappers, and explicit `device` widgets are rejected — TPU placement is process-wide.
- **Cache is write-only on `torch-xla 2.8.0`.** `TPU_CACHE_DIR` holds diagnostic/sharding reports and compilation entries, but deserialization is unsupported upstream — treat it as restart-diagnostic, not a guaranteed cold-start speedup. Use a fresh empty dir to rule out stale state.
- **Host memory vs device memory.** Under TPU, VRAM queries report host `available` memory (no HBM query pool). Expect RSS ~24 GiB after a generation; `buff/cache` in `free -h` is reclaimable file cache from the 33 GiB model read, not a leaked process.
- **Mass cache.** Linux `buff/cache` after staging is file cache from mmap'd safetensors. After a request, `ps -o rss` is the source of truth for process footprint (~348 GiB host-available on the verified run).
- **Upstream parity.** Frontend, templates, and custom nodes work as usual; `comfy-aimdo` is optional (absent on the TPU-minimal image, guarded at runtime).

## Docs

| Doc | What |
|---|---|
| [docs/deployment.md](docs/deployment.md) | Full deployment, operations, tests, verification, memory notes |
| [docs/progress.md](docs/progress.md) | Done / in-progress / roadmap (T5e-8 unless noted) |
| [docs/architecture/overview.md](docs/architecture/overview.md) | System layout, lifecycle, where to look |
| [docs/architecture/decisions/001-krea2-tpu-support.md](docs/architecture/decisions/001-krea2-tpu-support.md) | Analysis of the Krea2 changes (all 48 files) |
| [docs/reference/comfyui-upstream-README.md](docs/reference/comfyui-upstream-README.md) | Upstream ComfyUI README (verbatim) |

## Tests

```bash
python -m pytest tests/tpu -q   # 101 passed, no TPU or torch_xla needed
```

`tests/tpu/conftest.py` installs a stub accelerator + `args.tpu` before imports so the suite exercises the real TPU branches against the real bundled Qwen tokenizer and real param shapes.

---

*ComfyUI is an open project by [@comfyanonymous](https://github.com/comfyanonymous/ComfyUI) and contributors. TPU support in this fork is maintained separately — please file TPU issues against this repo and core ComfyUI issues upstream.*

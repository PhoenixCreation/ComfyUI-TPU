# Deployment — Krea2 Turbo on TPU v5e-8

> Phase 1 of the TPU port: one fixed profile (`krea2-1920x1080`) on a **single-controller TPU v5e-8** slice.
> Full technical decision record: [`docs/architecture/decisions/001-krea2-tpu-support.md`](architecture/decisions/001-krea2-tpu-support.md).

## Prerequisites

- A TPU v5e-8 VM (8 chips, `TPU_ACCELERATOR_TYPE=v5litepod-8`, HBM 15.75 GiB / chip) with exclusive PJRT access — no other process may hold `/dev/vfio/*`.
- Python 3.12, writable cache directory for the XLA persistent compilation cache.
- The three BF16 artifacts available on a mounted dataset or local disk (33 GiB total).

## Quick Start

Run from the repo root (`ComfyUI-TPU/`):

```bash
# 1. Install the exact verified dependency set. torch and torch-xla must stay
#    on the same release line; the +cpu torch wheel is correct for TPU.
python -m pip install -r deployment/requirements-tpu.txt

# 2. Stage the three large artifacts as symlinks. Override KREA2_MODEL_ROOT
#    when the model dataset is mounted elsewhere.
export KREA2_MODEL_ROOT=/kaggle/input/models/helltester2/krea2-bf16/transformers/default/1/models
deployment/stage_models.sh

# 3. The checked-in manifest already contains the verified SHA-256 values.
#    Do not rehash on every launch. Run this only after changing artifacts:
# python deployment/hash_artifacts.py

# 4. Set PJRT/topology before the first torch_xla import and choose a writable
#    cache directory. The launch performs a full generation warm-up.
source deployment/tpu_env.sh
export TPU_CACHE_DIR=/kaggle/working/tpu/.tpu-cache-krea2
deployment/launch.sh
```

The warm-up refuses to become `ready` while any digest is unpinned, missing, or mismatched. It generates the fixed profile once before opening the queue. On the verified v5e-8 runtime, Qwen3-VL stays on CPU to leave HBM for the 1920×1080 denoiser. Krea2 executes with XLA boundaries between transformer blocks — a monolithic denoiser graph exceeds the per-chip HBM limit.

PyTorch/XLA 2.8.0 writes compilation-cache entries on this image but reports that executable deserialization is unsupported. Treat the cache directory as diagnostic / write-only state, not as a guaranteed cold-start speedup. To rule out stale state, stop ComfyUI and relaunch with a new empty `TPU_CACHE_DIR`.

## Deployment Artifacts

| Path | Purpose |
|---|---|
| `deployment/requirements-tpu.txt` | Pinned env: `torch 2.8.0+cpu`, `torch-xla 2.8.0`, `libtpu 0.0.17`, `transformers 5.12.1`, … Install order matters: `libtpu` before `torch-xla`. |
| `deployment/tpu_env.sh` | `PJRT_DEVICE=TPU`, `TPU_SKIP_MDS_QUERY=1` (avoids blocking on GCE metadata), static topology `TPU_CHIPS_PER_HOST_BOUNDS=2,4,1` / `TPU_HOST_BOUNDS=1,1,1`, clears `TPU_PROCESS_ADDRESSES` / `XRT_TPU_CONFIG`. **Source before any `torch_xla` import.** |
| `deployment/stage_models.sh` | Symlinks `krea2_turbo_bf16.safetensors` → `models/diffusion_models/`, `qwen3vl_4b_bf16.safetensors` → `models/text_encoders/`, `qwen_image_vae.safetensors` → `models/vae/`. Honors `KREA2_MODEL_ROOT`. |
| `deployment/model_manifest.json` | Approved artifact allowlist + pinned SHA-256 + dtype (`bf16`). Warm-up fingerprint includes every digest. |
| `deployment/hash_artifacts.py` | Run once after placing artifacts to pin SHA-256 into the manifest. Fails if any artifact is missing / unreadable. |
| `deployment/launch.sh` | `python main.py --tpu --tpu-cache-dir "$TPU_CACHE_DIR" --tpu-profile krea2-1920x1080 --tpu-warmup --listen 0.0.0.0 --port 8188`. |
| `deployment/healthcheck.sh` | Polls `GET /tpu/status`; exits `0` only on `state == "ready"` (otherwise `503` body carries `last_error`). |
| `workflows/Krea2-turbo-tpu.json` | Canonical workflow the warm-up compiles and the validator accepts — must not drift. |

## Operations

| What | How |
|---|---|
| Warm-up / readiness state | `GET /tpu/status` → `state`, `mesh`, `artifact_hashes`, `fields`, `warmup_timestamps` |
| Healthcheck | `deployment/healthcheck.sh` (uses `TPU_HEALTHCHECK_URL` env, default `http://127.0.0.1:8188`) |
| Cold-start expectations | Startup warm-up compiles the fixed profile inline. In-process requests reuse compiled block programs. Cross-process deserialization is unsupported by `torch-xla 2.8.0` on the verified image. |
| Skip startup compile | `--no-tpu-warmup` (first request pays the compile). Not recommended for production. |
| Steady-state markers | One `tpu_request` structured log line per request: stage durations, `execution_interval_ms`, compile-counter deltas. |
| Sharding / cache reports | Written by the adapter into `readiness.fields` (`sharding_report`, `cache_profile`) and `TPU_CACHE_DIR`. |

## CLI Flags

`--tpu` requires `--tpu-cache-dir`. CUDA / DirectML device flags, VRAM management (`--lowvram`, `--highvram`, …), non-BF16 precision flags, attention flags (`--use-*-attention`, `--disable-xformers`, …), `--fast` and Triton are **rejected at parse time**. `--bf16-*` flags are accepted as redundant (the profile fixes all compute to BF16). See `tests/tpu/test_cli_tpu.py` for the full matrix.

## Tests

```bash
cd ComfyUI-TPU
python -m pytest tests/tpu -q   # 101 passed, headless
```

`tests/tpu/conftest.py` installs `args.tpu` plus a stub accelerator before any test import, so the suite exercises the TPU-mode code paths without `torch_xla` (real bundled Qwen tokenizer is used for the fixed-length contract; sharding policy is validated against real parameter names/shapes).

## End-to-End Verification

Wait until readiness and the UI are both healthy:

```bash
deployment/healthcheck.sh
curl -fsS -o /dev/null http://127.0.0.1:8188/
```

Submit the canonical workflow from another shell:

```bash
python - <<'PY'
import json, urllib.request
with open("workflows/Krea2-turbo-tpu.json") as f:
    prompt = json.load(f)
req = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=json.dumps({"prompt": prompt}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    print(r.read().decode())
PY
```

Check `/history/<prompt_id>` for `status_str: success`, then verify the saved file under `output/` with Pillow — the acceptance result must be RGB and exactly `(1920, 1080)`.

The verified run on 2026-08-16 saved `output/krea2_automatic_00001_.png` at 1920×1080. Warm-up took 105.4 seconds; the immediately following API generation took 32.7 seconds end-to-end.

## Memory Notes

Use process RSS and `available` memory when diagnosing host use:

```bash
ps -p "$(pgrep -f '^python main.py.*--tpu' | head -1)" -o pid,etime,rss,stat,args
free -h
```

Linux `buff/cache` is reclaimable file cache from reading the 33 GiB model set — it does not mean an old model process is still resident. After the verified request, ComfyUI RSS was about 24 GiB and host available memory was about 348 GiB.

## Scope Notes

- Phase 1: Krea2 Turbo only. LoRA / ControlNet / runtime model patching are rejected by the pre-queue validator.
- Prompt text and seed are free; every shape-affecting value is fixed by the profile.
- Non-TPU backends are untouched: every TPU path is gated on `--tpu`.

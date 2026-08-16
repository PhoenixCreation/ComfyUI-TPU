# ComfyUI TPU deployment — Krea2 Turbo on a TPU v5e-8

Phase 1 of the TPU port: one fixed production profile
(`krea2-1920x1080` — Krea2 Turbo + Qwen3-VL-4B text encoder + Qwen Image
VAE, 1920x1080 batch 1, 8 steps, `er_sde`/`simple`, CFG 1) on a single
controller TPU v5e-8 slice. Full contract: `docs/spec/krea2-tpu-implementation-spec.md`.

## Layout

| Path | Purpose |
|---|---|
| `comfy/xla_backend.py` | The only module that touches `torch_xla`. PJRT env is fixed before the first import. |
| `comfy/accelerator.py` | Process-wide adapter boundary (default no-op / XLA), `mark_step`, stage tracker, metrics. |
| `comfy/tpu_profile.py` | Profile constants, artifact manifest verification, pre-queue prompt validator, readiness state machine. |
| `comfy/tpu_sharding.py` | Sharding policy + report for the three artifacts. |
| `comfy/text_encoders/krea2.py` | Krea2 TE: fixed-length tokenization (512 to 478 conditioning), tap stack flattening. |
| `workflows/Krea2-turbo-tpu.json` | Canonical workflow the warm-up compiles and the validator accepts (must not drift). |
| `deployment/` | Pinned requirements, env, launch/healthcheck scripts, artifact manifest + hash tool. |
| `tests/tpu/` | 101 headless unit tests (no TPU, no torch_xla needed). |

## Deployment

Run these commands from `/kaggle/working/tpu/ComfyUI-TPU`:

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

The warm-up refuses to enter `ready` while any digest is unpinned, missing,
or mismatched. It generates the fixed profile once before opening the queue.
On the verified TPU v5e-8 runtime, Qwen3-VL stays on CPU to leave HBM for the
1920x1080 denoiser. Krea2 executes with XLA boundaries between transformer
blocks; a monolithic denoiser graph exceeds the 15.75 GiB per-chip HBM limit.

PyTorch/XLA 2.8.0 writes compilation-cache entries on this image but reports
that executable deserialization is unsupported. Treat the directory as
diagnostic/write-only state, not as a guaranteed cold-start speedup. To rule
out stale state, stop ComfyUI and launch with a new empty `TPU_CACHE_DIR`.

## Operations

| What | How |
|---|---|
| Warm-up / readiness state | `GET /tpu/status` → `state`, `mesh`, `artifact_hashes`, `fields`, `warmup_timestamps` |
| Healthcheck | `deployment/healthcheck.sh` (calls `/tpu/status`; exits 0 only when `ready`) |
| Cold-start expectations | Startup warm-up compiles the fixed profile inline. In-process requests reuse compiled block programs. Cross-process deserialization is unsupported by torch-xla 2.8.0 on the verified image. |
| Skip startup compile | `--no-tpu-warmup` (first request pays the compile). Not recommended for production. |
| Steady-state markers | One `tpu_request` structured log line per request (spec section 15): stage durations, execution interval, compile-counter deltas. |
| Sharding / cache reports | Written by the adapter into `readiness.fields` (`sharding_report`, `cache_profile`) and the cache dir. |

## Flags

`--tpu` requires `--tpu-cache-dir`. CUDA/directML device flags, VRAM
management, non-BF16 precision flags, attention flags, `--fast` and
triton are rejected at parse time; `--bf16-*` flags are accepted as
redundant (the profile fixes all compute to BF16). See
`tests/tpu/test_cli_tpu.py` for the full matrix.

## Tests

```bash
cd /kaggle/working/tpu/ComfyUI-TPU
python -m pytest tests/tpu -q          # 101 passed, headless
```

`tests/tpu/conftest.py` installs `args.tpu` plus a stub accelerator before
any test import, so the suite exercises the TPU-mode code paths without
`torch_xla` (spec section 17.1). The real bundled Qwen tokenizer is used
for the fixed-length contract; sharding policy is validated against real
parameter names/shapes of all three artifacts.

## End-to-end verification

Wait until readiness and the UI are both healthy:

```bash
deployment/healthcheck.sh
curl -fsS -o /dev/null http://127.0.0.1:8188/
```

Submit the canonical workflow from another shell:

```bash
python - <<'PY'
import json
import urllib.request

with open("workflows/Krea2-turbo-tpu.json") as f:
    prompt = json.load(f)
request = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=json.dumps({"prompt": prompt}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    print(response.read().decode())
PY
```

Check `/history/<prompt_id>` for `status_str: success`, then verify the saved
file under `output/` with Pillow. The acceptance result must be RGB and
exactly `(1920, 1080)`.

The verified run on 2026-08-16 saved
`output/krea2_automatic_00001_.png` at 1920x1080. Warm-up took 105.4 seconds;
the immediately following API generation took 32.7 seconds end to end.

## Memory notes

Use process RSS and `available` memory when diagnosing host use:

```bash
ps -p "$(pgrep -f '^python main.py.*--tpu' | head -1)" -o pid,etime,rss,stat,args
free -h
```

Linux `buff/cache` is reclaimable file cache from reading the 33 GiB model
set; it does not mean an old model process is still resident. After the
verified request, ComfyUI RSS was about 24 GiB and host available memory was
about 348 GiB.

## Scope notes

- Phase 1: Krea2 Turbo only. LoRA/ControlNet/runtime model patching are
  rejected by the pre-queue validator (spec section 13).
- Prompt text and seed are free; every shape-affecting value is fixed by
  the profile.
- Non-TPU backends are untouched: every TPU path is gated on `--tpu`.

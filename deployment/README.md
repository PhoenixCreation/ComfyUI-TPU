# Deployment — operational scripts (TPU)

> **Full guide has moved → [`docs/deployment.md`](../docs/deployment.md)**
>
> This folder now holds **only** the runnable deployment artifacts. All narrative / operator docs live under `docs/`.

## What stays here (and why)

| File | Why it stays in `deployment/` |
|---|---|
| `requirements-tpu.txt` | Pinned install — path hard-coded by `deployment/launch.sh` & CI. |
| `model_manifest.json` | Allowlist + SHA-256 — read at runtime by `comfy/tpu_profile.py`. |
| `tpu_env.sh` | PJRT env — must be `source`'d **before** first `torch_xla` import. |
| `stage_models.sh` | Symlinks artifacts into `models/`; honors `KREA2_MODEL_ROOT`. |
| `hash_artifacts.py` | Pins digests into `model_manifest.json` after staging. |
| `launch.sh` | Production launch: `python main.py --tpu --tpu-cache-dir …`. |
| `healthcheck.sh` | Readiness probe (`GET /tpu/status` → `ready`). |

## Quick pointer

```bash
python -m pip install -r deployment/requirements-tpu.txt
export KREA2_MODEL_ROOT=/path/to/krea2-bf16/models
deployment/stage_models.sh
source deployment/tpu_env.sh
export TPU_CACHE_DIR=/tmp/tpu-cache
deployment/launch.sh        # — see docs/deployment.md for full steps, operations, flags, verification
deployment/healthcheck.sh   # exits 0 only when GET /tpu/status == ready
```

For profile details, sharding, validator, readiness FSM, and the decision record, see:
- `docs/deployment.md`
- `docs/progress.md`
- `docs/architecture/decisions/001-krea2-tpu-support.md`
- `docs/architecture/overview.md`

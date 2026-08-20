# Architecture Overview — ComfyUI on TPU

Long-term goal: run **multiple ComfyUI models on Google Cloud TPU** (image / video / audio) through the standard node graph, with device placement, compilation, and operability owned by the framework — not by each model.

## Principles

- **Gate, don't fork.** All TPU behavior is behind `--tpu`. The default code path is identical to upstream ComfyUI. The only module that imports `torch_xla` is `comfy/xla_backend.py`.
- **Phase per profile.** Each model lands as a single frozen profile (fixed shapes, dtype, sampler, dimensions) before being generalized. `krea2-1920x1080` is Phase 1.
- **Shape stability.** Fixed profiles let XLA compile once and reuse programs; dynamic shapes are introduced only when the team has proven the static case (warm-up, sharding, HBM fit, validator).
- **Fail closed.** Artifact digest mismatch, unsupported node, or startup compile failure keeps the server in `failed` / `compiling` — the queue never serves a partial profile.

## System Layout

```
ComfyUI-TPU/
  comfy/
    xla_backend.py      ← only import of torch_xla (PJRT, mesh, sharding, metrics)
    accelerator.py      ← process-wide adapter + StageTracker
    tpu_profile.py      ← profile constants, manifest, validator, readiness FSM
    tpu_sharding.py     ← named policy krea2-tpu-v1
    model_management.py ← xla_enabled() gate for all dtype/device/memory queries
    model_patcher.py    ← whole-model sharded transfer, mutation rejection
  deployment/           ← pinned env + staging + manifest + launch/healthcheck
  workflows/            ← canonical Krea2-turbo-tpu.json (compiled at warm-up)
  tests/tpu/            ← 101 headless tests (stub accelerator, real tokenizer)
  docs/
    deployment.md       ← operator guide
    progress.md         ← done / in-progress / roadmap
    architecture/
      overview.md       ← this file
      decisions/        ← ADRs per phase
```

## Request Lifecycle (TPU)

```
 source tpu_env.sh
        │
        ▼
 initialize_accelerator() ── set PJRT env, fingerprint cache path, import torch_xla,
        │                     validate 8 devices, enable SPMD, create mesh
        ▼
 run_tpu_warmup() ────────── verify_artifacts() → compile canonical workflow →
        │                     readiness {loading→compiling→ready|failed}
        ▼
 server GET /tpu/status ◄──── readiness snapshot (state, mesh, hashes, fields)
        │
 prompt POST /prompt ──► validate_prompt() (upstream nodes only) ──► queue
        │                                                      │
        └─► StageTracker (execution_interval_ms, per-stage ms, tpu_request log)
```

## Where to Look

| Concern | File |
|---|---|
| PJRT fix, mesh, cache fingerprint, sharding annotation | `comfy/xla_backend.py` |
| Profile / tokenizer budget / validator / readiness | `comfy/tpu_profile.py` |
| Per-artifact sharding rules & reports | `comfy/tpu_sharding.py` |
| Dtype/device/memory policy | `comfy/model_management.py: xla_enabled()` |
| Full-resident transfer, mutation guards | `comfy/model_patcher.py: _load_tpu` |
| CLI conflicts & `enables_dynamic_vram` | `comfy/cli_args.py` |
| Warm-up sequencing | `main.py: run_tpu_warmup` |
| Readiness gate & status endpoint | `server.py` |
| Per-request instrumentation | `execution.py` + `comfy/accelerator.py: StageTracker` |

## Adding a New Profile

1. Define constants in `tpu_profile.py` (dimensions, latent, tokenizer constants, manifest entry), add loader allowlist.
2. Add sharding rules in `tpu_sharding.py`, bump `POLICY_VERSION`, validate against real param shapes.
3. Add CLI choice in `cli_args.py`, wire `xla_backend.MESH_SHAPE` / topology if the slice differs, extend fingerprint.
4. Add canonical workflow in `workflows/` and extend validator fields.
5. Extend `transfer_sharded` / text-encoder handling if the new model family needs it.
6. Add tests in `tests/tpu/` and a new ADR in `docs/architecture/decisions/`.

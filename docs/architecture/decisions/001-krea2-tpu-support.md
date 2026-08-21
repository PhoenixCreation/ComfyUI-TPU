# ADR 001 — Krea2 Turbo TPU Support

*Status: Accepted — shipped as one decision: fixed `1920×1080` (`770ac59c` + `73576271`) + dynamic multi-dimension lift (`bb7671ba`).*
*Scope: TPU v5e-8 (`v5litepod-8`, single-host 8 chips) · profiles `krea2` (dynamic) and `krea2-1920x1080` (compat alias, now also dynamic).*
*Spec reference: the checked-in implementation spec at `docs/spec/krea2-tpu-implementation-spec.md` (not yet published; referenced by `deployment/README.md` in the original commits).*

---

## 1. Context

ComfyUI has no TPU execution path. The long-term goal is to run multiple image / video / audio models on TPU through the existing node graph. Krea2 landed as a single profile on a single slice so the team could prove end-to-end correctness, HBM fit, and operability before generalizing, then lifted the shape constraint without a new decision.

**Why Krea2 Turbo first:** standalone diffusion model (no separate transformer split), well-understood BF16 checkpoint, permissive sampler (`er_sde`), and available pinned BF16 artifacts that fit the v5e-8 HBM budget with SPMD sharding.

**Profile constraint (spec §4, as shipped):** prompt text + seed are free; sampler/scheduler/CFG/batch/save-prefix remain frozen and validated. Dimensions were initially frozen to `width=1920 height=1080` (latent `1×16×135×240`, conditioning `B×478×30720`, tokens `512→478`) so XLA could compile once and reuse block programs. Commit `bb7671ba` lifted the dimension freeze to any `W×H` with `W%8==0 && H%8==0`, `512≤W,H≤2048`, area `262k–2.1M` (`comfy/tpu_profile.py:41` `is_valid_krea2_size()` / `latent_shape_for()`), handled by `pad_to_patch_size` per input — same dtype/sampler, new shapes compile on demand and stay in-memory (`CachedCompile`). This keeps validation decidable while allowing multiple 1M–2M resolutions without a new profile per size.

---

## 2. Decision

Add a minimal, gated TPU path that leaves all non-TPU backends (CUDA / ROCm / MPS / XPU / NPU / MLU / DirectML) untouched. Every TPU branch is `if args.tpu` and fails closed.

### 2.1 New modules

| Module | Responsibility | Imports `torch_xla`? |
|---|---|---|
| `comfy/xla_backend.py` | **Only** module that imports `torch_xla`. Fixes PJRT env before first import, creates 8-device `model` mesh via `runtime.use_spmd()` before the first XLA device, validates device count, sharding helpers, metrics, fingerprint-separated cache path, sharding report emission. | yes (lazy, inside `initialize`) |
| `comfy/accelerator.py` | Process-wide adapter boundary. Default adapter is no-op. In TPU mode holds `XlaAccelerator`. Also owns `StageTracker` / `stage_timer` instrumentation (spec §15) so the shape is identical in non-TPU runs. | no |
| `comfy/tpu_profile.py` | Profile constants, tokenizer constants (512→478), manifest I/O + SHA-256 verification, prompt validator, `ReadinessTracker` state machine. | no |
| `comfy/tpu_sharding.py` | Named sharding policy `krea2-tpu-v1` covering all three artifacts, validated against real param names/shapes. | no |

### 2.2 CLI & startup ordering (`comfy/cli_args.py`, `main.py`)

- New flags: `--tpu` (requires `--tpu-cache-dir`), `--tpu-profile {krea2,krea2-1920x1080}` (`krea2` dynamic, `krea2-1920x1080` compat alias now also dynamic; `comfy/cli_args.py:119`, `comfy/tpu_profile.py:16`), `--tpu-warmup` / `--no-tpu-warmup` (default on).
- `--tpu` conflicts with every GPU/precision/attention/VRAM/fast flag — rejected at parse time with `parser.error`. `--bf16-*` flags are accepted as redundant. `--fp16-intermediates` is rejected (profile pins BF16).
- `enables_dynamic_vram()` returns `False` under TPU.
- `main.py` imports `comfy.accelerator` and calls `initialize_accelerator()` **before** `comfy_aimdo` or any `torch.cuda` probing (spec §7 ordering). `cuda_malloc` / `OCL_SET_SVM_SIZE` are skipped under TPU. `comfy_aimdo.control` import is guarded.
- Warm-up driver (`run_tpu_warmup`) runs before `prompt_server.add_routes()`:
  `initializing → loading` (verify manifest) → `compiling` (execute canonical workflow `workflows/Krea2-turbo-tpu.json` with correct `execute_outputs`, temp output dir, compile-counter deltas) → `ready` / `failed`. Queue is gated on readiness via `server.py`. Under `krea2` dynamic, warm-up compiles `1920×1080`; additional sizes compile on demand at first use (`UncachedCompile` +14–20) and remain `CachedCompile` thereafter.

### 2.3 Execution & serving (`execution.py`, `server.py`)

- `execution.py` creates a per-request `StageTracker(prompt_id, profile)` and records host-side `execution_interval_ms`. Structured log `tpu_request` emitted once per request (spec §15).
- `server.py` exposes `GET /tpu/status` → `{state, profile, mesh, cache_dir, artifact_hashes, last_error, fields{sharding_report, cache_profile}, warmup_timestamps}`. Pre-queue hook runs `tpu_profile.validate_prompt` against upstream nodes only; failures return `tpu_profile_*` error codes.

### 2.4 Model loading & dtype / device (`comfy/model_management.py`, `comfy/model_patcher.py`, others)

**Model management** (`model_management.py`):
- `xla_enabled() = args.tpu`. All device/dtype/memory queries branch on it.
- `get_torch_device()` returns the XLA logical device (fails loudly if accelerator not yet initialized).
- `get_total_memory` / `get_free_memory`: report host `psutil` memory under TPU (no host-queryable HBM pool), models stay resident.
- `unet_dtype` / `text_encoder_dtype` / `vae_dtype` → `bf16`. `should_use_bf16 → True`, `should_use_fp16 → False`. `xformers_enabled → False`, `ENABLE_PYTORCH_ATTENTION = True`.
- `LoadedModel.model_load` on TPU: full-resident path (`device_to`, no partial-load / per-layer offload). `model_unload` waits on device ops. Pin budget is no-op.

**Model patcher** (`model_patcher.py`):
- Rejects LoRA / weight patches, object patches (except `manual_cast_dtype` / `model_sampling`), weight wrappers, hooks, wrappers/injections under TPU with actionable errors.
- `_load_tpu`: whole-model transfer via `module.to(xla_device)` + per-param `mark_sharding` with final spec. Mutation attempts raise before any transfer. Covers diffusion, text encoder, VAE loaders uniformly.
- `tpu_profile._ARTIFACT_DIR_BY_NAME` drives loader artifact allowlists.

**Memory / AIMDO** (`memory_management.py`, `model_patcher.py`, `model_management.py`, `pinned_memory.py`, `ops.py`):
- `comfy_aimdo` imports made optional (`try/except ImportError → comfy_aimdo=None`); guard via `aimdo_enabled`. TPU-minimal image omits AIMDO entirely.
- `pin_memory`, `ensure_pin_budget`, `aimdo` paths are no-ops under TPU.

### 2.5 Sampler & model numerics (`comfy/samplers.py`, `comfy/k_diffusion/sampling.py`, `comfy/ldm/krea2/model.py`, `comfy/ldm/flux/math.py`, etc.)

- `samplers.py`: sigmas created/broadcast on host, moved to XLA only if needed; avoids XLA-host sync on every step.
- `k_diffusion/sampling.py`: `torchsde` import soft-fails (Krea2 `er_sde` does not need it, keeps TPU image small); `default_noise_sampler` generates on CPU then `.to(xla)` because `torch.Generator` cannot target the SPMD virtual device; `sample_er_sde` calls `accelerator.mark_step()` per denoising step so eight passes stay as eight lazy graphs rather than one retained execution (HBM fix on `73576271`).
- `ldm/krea2/model.py`: removed `einops.rearrange` in hot paths (`reshape`+`transpose`+`permute`); block loop calls `accelerator.mark_step()` after each DiT block for the same HBM reason (monolithic denoiser graph >15.75 GiB).
- `ldm/flux/math.py`, `ldm/joyimage/model.py`, `ldm/lightricks/vae/na_diffusion_decoder.py`, `ldm/modules/attention.py`: minor dtype/device fixes for XLA BF16 path (avoid CPU round-trips, keep on device).

### 2.6 Text encoder (`comfy/text_encoders/krea2.py`, `comfy/text_encoders/llama.py`)

Fixed-length tokenization derived from pinned `qwen25_tokenizer` / `transformers==5.12.1`:
- Post-template input is exactly **512** tokens; **34** prefix tokens are stripped, leaving **478** conditioning tokens fed as `(B,478,30720)` (12 tapped layers × 2560).
- Content budget is 473; over-long prompts are truncated before the 5 closing tokens; empty prompt → 0 content tokens. Pad token 151643.
- Tap stack flattened without mutating caller conditioning dict. Attention mask stripped correctly; `llama.py` updated for same stripping path.
- Under TPU the Qwen3-VL text encoder stays on **CPU** (verified run: leaves HBM for 1920×1080 denoiser).

### 2.7 Sharding (`comfy/tpu_sharding.py`, `comfy/xla_backend.py`)

- Policy `krea2-tpu-v1` is part of cache fingerprint.
- **Krea2 DiT**: `wk`/`wv` (1536 = 12 KV-heads ×128) replicated — 12 does not divide by 8; `txtfusion.*` attention (2560 =20×128, 20∤8) replicated; `last.modulation.lin`/`txtfusion.projector` replicated (tiny F32). All other Krea2 linears partitioned `rows` (output) or `cols` (input) when divisible by 8.
- **Qwen3-VL**: `embed_tokens` replicated (vocab-gather cost); `model.visual.*` replicated (never executed in text-only path); `q/k/v/gate/up` → rows, `o/down` → cols.
- **VAE**: fully replicated (small 3D conv / norm).
- 1-D biases co-partitioned with their row-partitioned weight when applicable.
- `validate_policy` checks divisibility; `ShardingReport` emitted JSON per artifact. `xla_backend` translates `replicated → None` + mesh axis for `torch_xla.distributed.spmd.mark_sharding`. `transfer_sharded` uses native `module.to(device)` (replacing `.data` with XLA storage crashes PJRT) and annotates every param/buffer.

### 2.8 Caching & fingerprinting

Inputs to `XLA_PERSISTENT_CACHE_PATH` fingerprint: `torch` + `torch_xla` versions, profile, dtype, mesh shape, policy version, tokenizer constants, latent shape (or `dynamic` for `krea2` so all `W×H` share one cache dir; `comfy/xla_backend.py:104`), per-artifact SHA-256. Path: `<cache>/executables/<32-hex>`. `write_cache_profile` dumps inputs alongside cache. On `torch-xla 2.8.0` executable deserialization is unsupported — cache is write-only/diagnostic; reuse across sizes is in-memory `CachedCompile` (e.g., `1024×1024` 75 s → 4 s, `1080×1920` 56 s → 4–8 s).

### 2.9 Deployment

| Artifact | Role |
|---|---|
| `deployment/requirements-tpu.txt` | Fully pinned env (torch 2.8.0+cpu, torch-xla 2.8.0, libtpu 0.0.17, transformers 5.12.1, …). |
| `deployment/tpu_env.sh` | `PJRT_DEVICE=TPU`, `TPU_SKIP_MDS_QUERY=1`, static topology `2,4,1`, clears multi-host vars. Sourced **before** first `torch_xla` import. |
| `deployment/stage_models.sh` | Symlinks 3 artifacts from `KREA2_MODEL_ROOT` into `models/{diffusion_models,text_encoders,vae}`. |
| `deployment/model_manifest.json` | Allowlist + pinned SHA-256 (drift → warm-up fails). |
| `deployment/hash_artifacts.py` | Pins digests after artifacts placed. |
| `deployment/launch.sh` | `python main.py --tpu --tpu-cache-dir … --tpu-profile krea2 --tpu-warmup` (also accepts `krea2-1920x1080` compat). |
| `deployment/healthcheck.sh` | Polls `/tpu/status`; exits 0 only on `ready`. |

---

## 3. Alternatives Considered

- **Per-layer offload / CPU staging per step:** rejected — XLA SPMD always wants resident sharded tensors; host↔device transfers per layer would add collectives and recompiles without HBM benefit after block partitioning.
- **Vocab-sharded embedding / attention-head sharding for non-divisible heads:** rejected — gather cost on vocab and GQA head reshapes outweighed tiny parameter saving; explicitly replicated instead.
- **FlashAttention / xFormers on TPU:** rejected — not available / not beneficial under SPMD; fixed to `SDPA math` path.
- **Single monolithic denoiser graph:** rejected (`73576271`) — OOM before step 2 on v5e; `mark_step` per block makes 8-block reuse of a small compiled program viable.

---

## 4. Consequences

**Positive:**
- CUDA / ROCm / MPS / etc. unchanged (all TPU branches gated, default adapter is no-op).
- Warm-up guarantees the first user request hits a compiled graph; readiness gate prevents serving while inconsistent.
- Sharding policy is named, versioned, tested, emitted, and fingerprinted — mismatches fail loudly.
- Dynamic `W×H` shares one profile/cache dir (`latent=dynamic`) and one sharding policy — validated across `1024×1024` / `1080×1920` / `1152×896` / `1280×720` / `1024×768` without per-size profile churn.

**Constraints accepted:**
- One slice, BF16 only, no LoRA/ControlNet/patches, no custom device selection; sampler/scheduler/CFG/batch/save-prefix remain frozen. Dimensions are now validated within `512–2048` step 8, area `262k–2.1M` rather than fixed `1920×1080`. Relaxing remaining frozen values still requires a new profile version + policy revision + fingerprint bump.

## 5. Verification

- `python -m pytest tests/tpu -q` → 101 passed headless (real Qwen tokenizer contract included via `tests/tpu/conftest.py` stub accelerator + `--tpu` flag; validator now covers `is_valid_krea2_size`).
- End-to-end on v5e-8 — fixed: `deployment/healthcheck.sh` → prompt `workflows/Krea2-turbo-tpu.json` via `/prompt` → `output/krea2_automatic_00001_.png` RGB `1920×1080` (verified 2026-08-16; `docs/benchmark/krea2_1920x1080.md`).
- End-to-end on v5e-8 — dynamic: `krea2` cold 152 s + first at new size 50–75 s then warm `3–8 s` cached / `9–13 s` varying prompt, revisits stay warm; 30 gens across 5 sizes verified RGB (`docs/benchmark/krea2_dynamic.md`, `docs/benchmark/run_dynamic_benchmark.py`).

---

## 6. Follow-ups

- Ideogram 4 / MiniMax H3 / PiD upscaler via same adapter boundary (new profiles, new policies, same warm-up/readiness contract).
- Multi-slice / other accelerators: generalize `DEVICE_COUNT` / `MESH_SHAPE` / topology, keep fingerprint invariant.

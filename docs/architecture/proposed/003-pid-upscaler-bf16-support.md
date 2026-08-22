# Proposal 003 — PiD Upscaler TPU Support (BF16)

*Status: Proposed — not yet implemented. Becomes ADR 003 on acceptance. Artifacts verified 2026-08-22 at `/kaggle/input/models/helltester2/pid-upscaler-bf16/transformers/default/1` (see §2).*
*Scope: TPU v5e-8 (`v5litepod-8`) · new profile `pid` (alias `upscaler`) alongside `krea2` / `ideogram4` · BF16 diffusion+TE, F32 Flux VAE (CPU). Phase A = standalone upscaler (`Upscaler.json`); Phase B = fused `krea2→pid` pipeline (deferred per user 2026-08-22).*
*Upstream base: PiD already in-tree (`comfy/ldm/pixeldit/{model.py,pid.py,modules.py}`, `comfy/text_encoders/pixeldit.py`, `comfy_extras/nodes_pid.py`, `comfy/supported_models.py: PiD`, `comfy/model_detection.py: pid`). This proposal covers TPU integration, not model support itself.*
*Reference workflow: `workflows/Upscaler.json` (parent checkout `../workflows/Upscaler.json`) — aspect-ratio-preserving longest-edge 1024 → 4× pixel-space output. Plan builds a frozen TPU-canonical derivative `workflows/Upscaler-tpu.json` (fixed geometry, native loaders).*

---

## 1. Summary

Add a `pid` TPU profile following the Krea2 playbook: pinned BF16 artifacts with SHA manifest, fixed-shape warm-up workflow, per-artifact sharding policy, pre-queue validator. The structural differences from Krea2 are **(a) two VAE roles** (Flux VAE for encode of the low-res input, `pixel_space` passthrough VAE for decode — no weights), **(b) pixel-space diffusion** (`ChromaRadiance/PixelDiTPixel` latent: `(B,3,H,W)` not `(B,C,H/8,W/8)`), **(c) LQ conditioning branch** (`PiDConditioning` attaches `lq_latent` + `degrade_sigma` to conditioning, consumed by `PidNet`'s `LQProjection2D` + `SigmaAwareGate`), and **(d) 4K activation budget** (Phase A pins 1024×576 → 4096×2304, the Krea 1920×1080 → 16:9 composition; a square 1024×1024 → 4096×4096 bucket is ~1.33× larger and deferred).

Phase A pins one resolution bucket. Phase B (follow-up) lifts to additional longest-edge-1024 buckets using the `krea2` dynamic mechanism (`is_valid_*_size()`-style gate + `latent=dynamic` fingerprint sharing).

## 2. Artifacts

PiD 1.5 INT8/ConvRot checkpoints exist but require CUDA ConvRot kernels. For TPU Phase A use offline BF16 (dequantized if needed, like Ideogram proposal). Artifacts staged to `models/{diffusion_models,text_encoders,vae}` — `pixel_space` decode needs no file (`comfy/pixel_space_convert.py`).

**Verified staging source (2026-08-22):** `/kaggle/input/models/helltester2/pid-upscaler-bf16/transformers/default/1/models`

| Staged name | On-disk source | Workflow source key | Role | Verified |
|---|---|---|---|---|
| `pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors` | `diffusion_models/pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors` — 2 800 450 070 B, SHA256 `18931256e97822dc31db10b1e7399c73e7ee2c897f6d461eb1d1cf5e1d2de049`, 461 keys, all `BF16`, `net.*` (`patch_blocks` 336 / `lq_proj` 73 / `pixel_blocks` 34) — ~1.40 B params / 2.80 GB BF16 | `275:295.unet_name = pid_1.5_flux1_1024_to_4096_4step_int8_convrot.safetensors` | PiD diffusion (`PidNet` wrapping `PixDiT_T2I`) — `patch_size=16`, `hidden_size=1536`, `patch_depth=14`, `pixel_depth=2`, `lq_interval=2` | `safetensors` header dtype set `{BF16}` only |
| `gemma_2_2b_it_elm_bf16.safetensors` | `text_encoders/gemma_2_2b_it_elm_bf16.safetensors` — 5 232 958 571 B, SHA256 `e7ae59c203c392db4aa4e27783e924ec3225eb563392260cf747e1130ffcdb88`, 289 keys (`BF16` ×288 + `U8`×1 `spiece_model` 4241003 B), 26 layers, hidden 2304, head split `q 2048 / k 1024 / v 1024 / o 2048+2304` | `275:272.clip_name = gemma_2_2b_it_elm_bf16.safetensors` | Text encoder `pixeldit` (Gemma2-2B) → 2304-dim, `chi_prompt` prepended, trimmed to 300 | Same tokenizer as Lumina Gemma2 (`comfy/text_encoders/lumina2.py: Gemma2BTokenizer`) |
| `flux1_vae.safetensors` *(symlink target `flux1-vae.safetensors`)* | `vae/flux1-vae.safetensors` — 335 304 388 B, SHA256 `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38`, 244 keys, **all `F32`** (Flux.1-AE, `modelspec.hash_sha256 0xddec9c…231d5b6d`) — note hyphen vs underscore | `258.vae_name = flux1_vae.safetensors` | Flux VAE for `VAEEncode` of low-res input (latent 16 ch, 8×) | `F32` → keep on CPU (Krea Qwen pattern) or cast on load; not BF16 — update dtype contract |
| *(virtual)* `pixel_space` | — | `275:262.vae_name = pixel_space` | Fake VAE `PixelspaceConversionVAE` (passthrough, single param `1.0`) | No file, no sharding; validated as allowed literal |

Former size estimates (BF16): PiD ~5–7 GB — **measured 2.80 GB BF16** (1.40 B params). Gemma2-2B ~4.5 GB — **measured 5.23 GB** (2.6 B BF16 + SPM). Flux VAE ~0.3 GB — **measured 0.335 GB F32**. Total ≲8.37 GB host RAM; still fits replicated. Sharded PiD over 8 chips ≈ **350 MB/chip** (2.80 GB /8); Gemma stays **CPU-resident** like Krea Qwen (`model_management.text_encoder_device()=CPU` for TPU, 5.23 GB host) so zero HBM; VAE replicated only if sharded (335 MB) otherwise CPU. Weight HBM therefore <0.4 GB/chip — activations dominate (see §5).

**Deployment:** extend `deployment/model_manifest.json` with 3 entries (pid, gemma, flux VAE) including verified SHA256/length/`dtype` (vae `F32`), `deployment/stage_models.sh` with `PID_MODEL_ROOT` (or reuse `FLUX_MODEL_ROOT`) and symlink `flux1-vae.safetensors` → `flux1_vae.safetensors` plus `flux1_vae.safetensors` → `models/vae/`, `deployment/hash_artifacts.py` covers new digests. Fingerprint includes all digests + `vae_dtype=F32`.

## 3. Target profile (Phase A)

Pinned bucket derived from Krea 1920×1080 composition (longest-edge 1024, 16:9):

```
input_img:  1024×576  (W×H, longest edge 1024, aspect preserved from 1920×1080)
                 ↓ flux1_vae encode (×8)
lq_latent:  (1, 16, 72, 128)   # (B, 16, H/8, W/8) for 576×1024
latent_noise: (1, 3, 2304, 4096) # EmptyChromaRadianceLatentImage 4096×2304 (B,3,H,W)
output_img: (1, 3, 2304, 4096)  # pixel-space decoded, passthrough VAE, saved as RGB

Steps: 4  Sampler: lcm  Scheduler: simple  cfg: 1  denoise: 1.0  degrade_sigma: 0  latent_format: flux
Text: "" (empty) → chi_prompt + Gemma2 → (1, 300, 2304) trimmed to BOS + last 299
 conditioning LoQ: lq_latent (flux-normalized via Flux.process_in) + degrade_sigma tensor [0.0]
VAE encode: flux1_vae on CPU (mirrors Krea TE placement) — verify tracing feasibility
VAE decode: pixel_space (no-op, host-side tensor → PNG)
Save prefix: PiD
```

- **Tokenizer constants:** `_PIXELDIT_MAX_LENGTH=300`, `_PIXELDIT_CHI_PROMPT` (~80 tokens) + `Gemma2BTokenizer` (`comfy/text_encoders/pixeldit.py:60`). Like Krea's 512→478, PiD's fixed contract is `300` after `BOS + last 299` slice (`pixeldit.py:86`). No prefix-strip like Krea; conditioning is always `(B,300,2304)`. Measure `chi_token_count` against pinned `transformers==5.12.1` and reuse existing `tpu_profile` measurement shim.
- **Latent contract:** `PixelDiTPixel/ChromaRadiance` — `EmptyChromaRadianceLatentImage` emits `(B,3,H,W)` at full pixel resolution (`nodes_chroma_radiance.py:26`). PiD pads to patch size 16 (`model.py:205`) so W×H must be multiple of **16** (Krea steps by 8 via `latent_formats.Flux`, PiD steps by patch).
- **LQ branch:** `PiDConditioning` (`nodes_pid.py:34-51`) wraps `lq_latent = Flux().process_in(VAEEncode_out)` with optional `Wann` `Wan21` format fallback, but workflow pins `flux`. `degrade_sigma` is always scalar tensor `[0.0]` in Phase A. `PidNet._forward` (`pid.py:234`) validates `lq_latent.shape[1]==16` (flux) else raises, reshapes `LQProjection2D` gates.
- **Scheduler:** `BasicScheduler` 4 steps, `SamplerCustom` `cfg=1`, `lcm` sampler. Sigmas host-side (`comfy/samplers.py` style).
- **Math nodes:** `GetImageSize`, `ComfyMathExpression`, `MultiplyNode`, `ImageScale` compute dynamic geometry in the generic workflow. TPU canonical `Upscaler-tpu.json` **removes** them and hardcodes `256/257: 1024×576` scale + `275:261: 4096×2304` + `257: VAEEncode` on the pre-scaled image. Dynamic math is reintroduced only in Phase B via `is_valid_pid_size()` gate.

Phase B follow-up: lift to other longest-edge-1024 aspect ratios (e.g., square 1024×1024 → 4096×4096, 576×1024 portrait, 768×1024) using Krea's `latent=dynamic` fingerprint bucket + on-demand compile per distinct `(H,W)` patch grid. Operational cost is one warm compile per bucket, in-memory `CachedCompile` thereafter.

## 4. Changes

Reuse without modification: mesh/env/cache machinery (`xla_backend.py`), `mark_step` boundaries (`comfy/k_diffusion/sampling.py` & `comfy/ldm/pixeldit`), whole-model sharded transfer (`model_patcher._load_tpu`), artifact-driven loaders, CPU text-encoder placement (`model_management.text_encoder_device()`), BF16 dtype pins, host-side sigma path.

| File | Δ lines | Change |
|---|---|---|
| `deployment/model_manifest.json` | — | Pin 3 new digests (pid, gemma, flux VAE) under new profile key |
| `deployment/stage_models.sh` | +~20 | Stage `flux1_vae.safetensors`→`vae/`, `gemma_2_2b_it_elm_bf16.safetensors`→`text_encoders/`, `pid_*_bf16.safetensors`→`diffusion_models/` (env `PID_MODEL_ROOT`, symlink) |
| `deployment/hash_artifacts.py` | +~2 | No code change — manifest growth covered; add comment for pid profile |
| `comfy/tpu_profile.py` | +~150 | `PROFILE_PID=upscaler` (+ alias `pid`), `PROFILE_PID_WIDTH_IN=1024`, `HEIGHT_IN=576`, `WIDTH_OUT=4096`, `HEIGHT_OUT=2304`, `UPSCALE_FACTOR=4`, `PIXELDIT_FIXED_LEN=300`, `PIXELDIT_CHI_PREFIX_LEN` (measured), `DYNAMIC_STEP=16`, `is_valid_pid_size()`, `latent_shape_for_pid()`, validator: allow `PiDConditioning` (flux, degrade_sigma==0), `CLIPLoader type=pixeldit`, `EmptyChromaRadianceLatentImage` step-16 gate, `VAELoader` `pixel_space` literal, `VAEEncode` flux, `SamplerCustom`+`BasicScheduler` (steps 4, lcm/simple/cfg1), `UNETLoader` pid artifact, `SaveImage PiD`. Analogous `krea2` dynamic alias pattern. |
| `comfy/cli_args.py` | +1 | add `"pid"` / `"upscaler"` to `--tpu-profile` choices (keep `krea2` compat) |
| `comfy/xla_backend.py` | +~15 | Fingerprint parametrized per active profile: `dtype=bf16` stays, `latent=1024x576→4096x2304` or `dynamic`, `tokens=300:chi_len`, `vae=flux1+pixels_space` string. Reuse dynamic branch (`args.tpu_profile == PROFILE_PID`). |
| `comfy/tpu_sharding.py` | +~90 | `PROFILE_PID_RULES` (PixDiT MMDiT + PiT + LQ), `_GEMMA2B_RULES` (clone of qwen pattern but vocab 256k, heads 8, dim 2304), Flux VAE `replicate` (existing), `policy_for_artifact` branches for 3 artifacts, `POLICY_VERSION` bump → `krea2-pid-tpu-v2` or `pid-tpu-v1`. Divisibility checked against 8 (see notes). |
| `comfy/text_encoders/pixeldit.py` | +~60 | Fixed-length `tokenize_with_weights` path mirroring `krea2.py:36-58` → ensure even empty prompt pads to `chi+300` and TPU `encode_token_weights` keeps `(1,300,2304)` shape-stable + `mark_step` boundary before PiD consume. Keep `pixeldit_te` BF16 contract. |
| `comfy/ldm/pixeldit/{model.py,pid.py,modules.py}` | +~12 | Per-block `mark_step` in `PixDiT_T2I._forward` patch-loop (14) + `PiTBlock` pixel loop (2) — mirrors `krea2/model.py:372`. Without this, 4096×2304 joint attention at `L=36864` (256×144 patches) + pixel stage 2× `L×256` is monolithic. Gate/LQ `LQProjection2D` stays on device; `SigmaAwareGate` scalar broadcast kept as XLA scalar. |
| `comfy/ldm/pixeldit/pid.py` | +~5 | Ensure `log_alpha` cast mirrors upstream (`to(dtype=torch.float32)`) — already XLA-friendly; add comment guard. |
| `comfy/supported_models.py` | +0 | Already supports `PiD` (diffusers Convert path not needed for TPU). |
| `comfy/pixel_space_convert.py` | +0 | No change — tiny passthrough VAE stays on CPU/host. |
| `main.py` | +~15 | `run_tpu_warmup`: resolve `workflows/Upscaler-tpu.json` when `profile==pid`, enumerate output nodes (`SaveImage`/`PreviewImage` + `VAEDecode` 275:259), temp output dir, same readiness FSM. |
| `workflows/Upscaler-tpu.json` | new | Canonical graph: `VAELoader(flux1_vae)` → `VAEEncode` (pre-upscaled 1024×576 image) → `PiDConditioning` (flux, 0.0) → `CLIPLoader(pixeldit, gemma)` → `CLIPTextEncode("")` → `UNETLoader(pid_bf16)` + `EmptyChromaRadianceLatentImage(4096×2304)` + `KSamplerSelect(lcm)` + `BasicScheduler(simple,4)` → `SamplerCustom(270)` → `VAEDecode(pixel_space)` → `SaveImage(PiD)` + `LoadImage` (1024×576) + `ImageCompare`. No `GetImageSize`/`MathExpression`/`MultiplyNode`/`ImageScale`/`CLIPLoaderMultiGPU`/`UNETLoaderDisTorch2MultiGPU`. |
| `tests/tpu/*` | +~14 cases | profile constants, `is_valid_pid_size` step-16, validator paths (good canonical, missing PiDConditioning, wrong latent_format, wrong clip type, `degrade_sigma!=0`, `pixel_space` vs real VAE, scheduler/sampler mismatches), sharding divisibility against dumped `pid_params.json`/`gemma_params.json`/`flux_vae_params.json`, tokenizer chi+300 length, warmup resolves correct workflow per profile. |

**Sharding notes (divisibility over 8) — verified 2026-08-22 against real safetensors:**

- MMDiT core (checked 14× `patch_blocks`): `hidden_size=1536` → `qkv_x/y` `4608=3×1536` rows ✓ (`r%8=0 c%8=0`), `proj_x/y` `1536` cols ✓, `adaLN_modulation_{img,txt}` `9216=6×1536` rows ✓, `mlp_x/y` SwiGLU `w1 4096×1536 / w2 1536×4096 / w3 4096×1536` all ✓, norms `1536` ✓. `t_embedder 1536×256`, `s_embedder 1536×768` both divisible.
- PiT branch (2× `pixel_blocks`): `PixelTokenEmbedder.proj 16×3` replicated (tiny), `compress_to_attn 1152×4096` (`1152%8=0 4096%8=0`) ✓, `expand 4096×1152` ✓, `qkv 3456=3×1152` ✓, `proj 1152` ✓, `adaLN_modulation 24576=16×1536` ✓, `mlp 64×16 / 16×64` tiny replicated.
- LQ branch: `latent_proj` `Conv2d 1024×16` + `1024×1024` + 4×Res `1024×1024` all out `1024%8=0` ✓ (conv replicated), `output_heads` 7× `1536×1024` ✓, `pit_head 1536×1024` ✓, `SigmaAwareGate.content_proj (1,3072)` rows `1` → **replicate** (tiny, GQA-like exception `r%8=1` — same as Krea `wk/wv` pattern), `log_alpha ()` scalar replicate.
- Gemma2-2B (26 layers, dumped `text_encoders/gemma_2_2b_it_elm_bf16.safetensors`): `embed_tokens 256000×2304` (`r%8=0 c%8=0`) → replicate (gather), `q 2048×2304 / k 1024×2304 / v 1024×2304 / o 2304×2048` all ✓, `gate/up 9216×2304 / down 2304×9216` all ✓, norms `2304` ✓, `spiece_model (4241003,) U8` → replicate/host. GQA `k/v 1024` is divisible (8×128), not the Krea 1280 exception, so only `embed_tokens` + norms + SPM are replicated.
- Flux VAE encode: 244 F32 conv kernels → replicated (existing `_VAE_RULES`), but **F32 dtype** must be kept on CPU or cast via `cast_to` — do not shard as BF16.

## 5. Risks and open questions

| Risk | Mitigation | Status / remaining question |
|---|---|---|
| 4096×2304 activation HBM — PixDiT joint attention `L = Hs×Ws = 256×144 = 36864` patch tokens + text 300, pixel stage `L×256 = 9.4M` pixel tokens chunked | Per-block `mark_step` + pixel MLP chunking (`pixel_mlp_chunks=2` already in ckpt; `comfy/model_detection.py:553` may set `pixel_mlp_chunks` differently for quant variants — Pin BF16 to `2`). Measure `L×D` per device: `36864×1536` bf16 ≈ 108 MB replicated → sharded 13.5 MB/chip before attention temps (now **350 MB/chip weights** leaving >90% HBM for activations). If OOM, increase `mark_step` granularity to per-MMDiT + per-PiT split and/or bump `pixel_mlp_chunks` to 4. | Confirm BF16 PiD `pixel_mlp_chunks` value and whether it can be forced without reloading. |
| Compile time 2–3× Krea warmup (pixel + LQ branch + 4 denoising steps) | Accept for Phase A; warmup reports `compile_counters_delta`. Cache fingerprint separates `krea` vs `pid` executables so mixed runs do not thrash. | Expected startup budget? `krea2` warm 105 s; pid 4K may be 180–260 s cold. |
| Flux VAE encode under XLA — **verified F32 dtype** (`vae/flux1-vae.safetensors` all `F32`, not BF16 as assumed) | **Decision:** keep `flux1_vae` on CPU (`vae_device()` already CPU in TPU mode `model_management.py:1310`) and transfer `lq_latent (1,16,72,128)` bf16 to XLA — weightless transfer vs recompiling F32 graph and mixed-dtype shard. `comfy/model_management.cast_to` at use covers dtype. M0 probe still includes `VAEEncode`→`PidNet`→`pixel_space` but VAE outside compiled region. | Closed for Phase A (CPU VAE). Phase B fused `krea2→pid` must chain two VAEs (Qwen + Flux) on CPU without host PIL round-trip — confirm ordering. |
| `PiDConditioning` `degrade_sigma` scalar tensor device mismatch | Upstream `PidNet` does `degrade_sigma.to(device=x.device)` (`pid.py:248`) — keep host scalar, cast at use. | Pin `degrade_sigma=0.0` for Phase A? Allow range later. |
| Dynamic math removal breaks generic upscaler UX | Validator rejects workflows containing `GetImageSize`/`MathExpression`/`MultiplyNode` upstream of `EmptyChromaRadianceLatentImage` for `pid` profile; error suggests using the fixed `Upscaler-tpu.json`. Host-side resize remains allowed (`ImageScale` before `VAEEncode`) but TPU canonical prefilters to 1024 longest edge on CPU before XLA section. | Should Phase A support only 1024×576 via hardcode, or also 1024×1024 square as second warmup bucket? |
| Tokenizer `chi_prompt` truncation drift across transformers versions | Pin `transformers==5.12.1` (already deployment) and measure `chi_token_count` at startup; include `chi_prefix_len` in fingerprint. | Confirm empty prompt is canonical warmup or a sample prompt like `""` vs `"a photo"`? |
| Multi-profile cache explosion | Each `(Hs,Ws)` yields distinct `precompute_freqs_cis_2d` RoPE table (`modules.py:13`). Keep fingerprint `latent=1024x576:4096x2304` pinned; Phase B adds `dynamic` entry whose cache dir holds per-shape executables (one per distinct patch grid). | **Decided 2026-08-22:** Phase A standalone `upscaler` only; fused `krea2→pid` (Krea 1920×1080 → PiD 4096×2304 in one queued request) is Phase B follow-up. |
| Checkpoint `lq_latent_channels` mismatch (Flux=16 vs Flux2=128) | Validator + `PiD._forward` raise with explicit channel message (`pid.py:238`). Phase A pins 16 (Flux1). Verified artifact `lq_proj.latent_proj.0.weight (1024,16,3,3)` confirms 16-ch. | Will provider ever ship Flux2-tuned PiD (128-ch `lq_proj` with `latent_fold_factor`)? If so, need second artifact family. |
| `UNETLoaderDisTorch2MultiGPU` / `CLIPLoaderMultiGPU` drift | Replaced by native loaders in TPU canonical workflow; validator rejects `*MultiGPU` variants (`tpu_profile._UNSUPPORTED_LOADERS`). | **Closed:** staging renames `flux1-vae.safetensors` (hyphen on disk) → `flux1_vae.safetensors` (underscore expected by `VAELoader`), ditto for TPU warmup. |
| Disk hyphen vs Comfy underscore | Staging symlinks `flux1-vae.safetensors` (verified on disk) to `flux1_vae.safetensors` (expected by `258.vae_name`) and `deployment/stage_models.sh` handles both. | Needs fix in `stage_models.sh` alias. |

## 6. Milestones

- **M0 — trace probe (now partially verified, no code yet):** headers verified: PiD `BF16` divisibility 8 ✓ (all but `gate (1,3072)` replicate), Gemma GQA `k/v 1024` divisible ✓ (no Krea 1280 exception), VAE `F32` → CPU path. Remaining: run `validate_policy` against new `PROFILE_PID_RULES` and one SPMD forward at `(1,3,2304,4096)` pixel + `(1,300,2304)` context + `(1,16,72,128)` lq_latent with per-block `mark_step`; confirms HBM fit, verifies `pixel_mlp_chunks` memory, confirms `s_embedder 1536×768` path.
- **M1 — sharded residency:** stage from verified path `transformers/default/1/models/{diffusion_models,text_encoders,vae}` via `stage_models.sh` alias (`flux1-vae` hyphen → underscore), `verify_artifacts` clean (pid SHA `1893125…`, vae SHA `0xddec9c…`), `test_policy_tpu.py` fixtures from dumped headers pass.
- **M2 — fixed profile validator:** `validate_prompt` clean on `workflows/Upscaler-tpu.json`, rejects wrong clip type, wrong `degrade_sigma`, step-16 violations, `GetImageSize` math upstream.
- **M3 — warmup compile (standalone):** `--tpu-profile pid` (or `upscaler`) `--tpu-cache-dir /tmp/pid --tpu-warmup` compiles canonical `Upscaler-tpu.json`; `/prompt` returns RGB `4096×2304`; `tpu_request` stages + `compile_counters_delta` logged; `sharding_report.json` + `cache_profile.json` written. *Per user 2026-08-22: Phase A is standalone upscaler only.*
- **M4 — benchmark doc** `docs/benchmark/pid_1024_576_to_4096_2304.md` (cold compile, warm steady-state, HBM `xla_current_bytes`, per-stage ms), then **Phase B** fused `krea2→pid` (Krea 1920×1080 Qwen VAE → Flux VAE → PiD) without host PIL round-trip.

Estimated total: ~380–480 lines across ~11 files plus new workflow JSON and 3 artifact fixtures. Estimated HBM after verified sizes: ~350 MB/chip PiD sharded + CPU TE — well under v5e-8 budget.

---

## Appendix A — Upscaler.json walkthrough (for validator)

Existing `../workflows/Upscaler.json` (`workflows/Upscaler.json` in parent checkout):

- `278 LoadImage` → `279 GetImageSize` → `284/287 ComfyMathExpression round(a*1024/max(a,b))` → `288 ImageScale(nearest-exact)` (to 1024 longest), `289/291 MultiplyNode ×4` → `275:261 EmptyChromaRadianceLatentImage (4× dims)` is dynamic.
- `257 VAEEncode(flux1_vae)` consumes `288` scaled image; `275:263 PiDConditioning(flux,0.0)` consumes `257` latent + `275:266 positive`; `275:272 CLIPLoader pixeldit/gemma`.
- Sampling stack `275:264 KSamplerSelect(lcm)`, `275:265 BasicScheduler(simple,4)`, `275:269 SamplerCustom(cfg1)` on top of diffusion `275:295 UNETLoaderDisTorch2MultiGPU`.
- `275:259 VAEDecode(pixel_space)` → `274 SaveImage(PiD)`.

Validator for Phase A will accept **only** the frozen derivative where `278/279/284/287/288/289/291` are replaced by constant `1024×576` scale image and `4096×2304` empty latent, and loaders are native (`UNETLoader`, `CLIPLoader`, `VAELoader`). Dynamic nodes upstream of `EmptyChromaRadianceLatentImage` or `VAEEncode` are rejected with `tpu_profile_dynamic_geometry`.

## Appendix B — deferred: INT8 / ConvRot

Like Ideogram 002, `int8_convrot` PiD artifacts can be probed after BF16 ships: verify `torch._int_mm` lowering vs XLA `quantized_dot` after patching `comfy-kitchen-TPU`. Rows-preferred sharding for ConvRot K-groups of 256 must not straddle devices. Deferred for same reason: BF16 HBM for 4K is high but still fits; INT8 saves ~0.4 GB/chip diffusion but adds second codepath. Revisit if Phase B wants >16 Mpx or multi-bucket residency.

## Appendix C — rejected alternatives

- **Per-request dynamic math on CPU + XLA encode:** Would keep one TPU graph per expanded size but reintroduces host `F.interpolate` during warmup; instead bake CPU `ImageScale` outside the compiled region and make TPU section fully static.
- **Reuse Krea `qwen_image_vae` for LQ encode:** Wrong latent channels (`Qwen VAE` vs Flux `16`); `PidNet` explicitly validates 16/128. Must use `flux1_vae`.
- **Sharding Flux VAE over 8 devices:** VAE is ~0.3 GB and conv-heavy with small kernels — replicate per `tpu_sharding._VAE_RULES` precedent; sharding adds collectives with zero HBM win.

---

*Remaining questions for reviewer (most closed by verification 2026-08-22):*

1. **~~Artifact names & provenance~~ — verified:** `pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors` (SHA `1893125…`, 2.80 GB BF16) / `gemma_2_2b_it_elm_bf16.safetensors` (5.23 GB) / `flux1-vae.safetensors` hyphen on disk (335 MB F32, Flux.1-AE `0xddec9c…`) staged from `…/transformers/default/1/models`. Need HF repo/revision to record in `model_manifest.json` `source` field. Should `flux1_vae` manifest entry keep `F32` dtype tag?
2. **Canonical geometry:** is pinning 1024×576 → 4096×2304 (Krea 16:9 composition) sufficient for acceptance, or must the warmup also cover the square 1024×1024 → 4096×4096 bucket? Both share one fingerprint under `latent=dynamic`, but the first compile of the second bucket will be on-demand (50–75 s).
3. **~~Profile composition~~ — decided 2026-08-22:** Phase A standalone `upscaler` only (client uploads any 1024-long-edge image, as in current `Upscaler.json`); fused `krea2→pid` (single queue item generates 1920×1080 then immediately upscales to 4096×2304) is Phase B. Propose to enforce this ordering in `tpu_profile.py` validator (`krea2` and `pid` separate).
4. **Text conditioning:** is empty prompt the only accepted TPU value (allows fixed `chi`+`BOS+last299` graph), or must arbitrary prompts be accepted (still shape-stable 300, but vocab coverage matters for acceptance image)?
5. **~~VAE placement~~ — decided:** `flux1_vae` F32 stays on CPU (`vae_device()=CPU` in TPU mode) → `lq_latent (1,16,72,128)` transfer to XLA. Sharding F32 VAE rejected (mixed dtype + no HBM win).
6. **Loader replacement scope:** may `workflows/Upscaler-tpu.json` freely drop `*MultiGPU` loaders (process-wide SPMD is canonical), or must `CLIPLoaderMultiGPU`/`UNETLoaderDisTorch2MultiGPU` be shimmed for compat?
7. **Profile naming:** prefer `--tpu-profile pid`, `upscaler`, or `pid-upscaler-4096x2304`? This proposal uses `pid` (+ alias `upscaler`) mirroring `krea2` convention. User confirmed `pid-upscaler-bf16` on disk — keep `pid` alias for CLI brevity.
8. **HBM headroom target:** is 10–15 % per-chip headroom (Krea gate) still the acceptance threshold at 4096×2304, or is a tighter budget acceptable? Verified weights now only ~350 MB/chip, so activation headroom is generous.
9. **Test inputs:** path to the reference input image for warmup (current workflow uses `Gemini_Generated_Image_a63al3a63al3a63a.png` from `input/`; TPU warmup uses a synthetic zero/hint image via `VAEEncode` of the `ImageScale` output — should it be checked-in or generated)?

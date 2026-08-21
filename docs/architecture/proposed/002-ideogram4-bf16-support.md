# Proposal 002 — Ideogram 4 TPU Support (BF16)

*Status: Proposed — not yet implemented. Becomes ADR 002 on acceptance.*
*Scope: TPU v5e-8 (`v5litepod-8`) · new profile `ideogram4` alongside `krea2` / `krea2-1920x1080` · BF16 artifacts only (int8 alternative analyzed in Appendix A).*
*Upstream base: Ideogram4 already exists in-tree (`comfy/ldm/ideogram4/model.py`, `comfy/text_encoders/ideogram4.py`, `comfy_extras/nodes_ideogram4.py`, PR CORE-208). This proposal covers the TPU integration, not model support itself.*

---

## 1. Summary

Add an `ideogram4` TPU profile following the proven Krea2 playbook: pinned BF16 artifacts with SHA manifest, fixed-shape warm-up workflow, per-artifact sharding policy, pre-queue validator. The structural difference from Krea2 is **dual diffusion models**: Ideogram 4 runs a conditional DiT (text+image) and a separate unconditional DiT (image-only) joined by `DualModelGuider`. Both must be resident and sharded simultaneously.

Phase A below pins one resolution (1024×1024). Phase B lifts dimensions using the exact `krea2` dynamic mechanism (`is_valid_krea2_size()`-style gate + `latent=dynamic` cache fingerprint).

## 2. Artifacts

No official BF16 exists (only fp8 weight-only e4m3 and CUDA-only nf4; confirmed in the upstream HF discussion). Plan: take the Comfy-Org repackaged fp8 files and **dequantize to BF16 offline** (`w_bf16 = w_fp8.to(bf16) × scale`). This is lossless relative to what the official release provides — fp8 weight-only dequantizes at compute time anyway.

| Artifact (staged name) | Source file | Source size | BF16 size | Role |
|---|---|---|---|---|
| `ideogram4_bf16.safetensors` | [diffusion_models/ideogram4_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_fp8_scaled.safetensors) | ~13.8 GB | ~18.6 GB | conditional DiT |
| `ideogram4_unconditional_bf16.safetensors` | [diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors) | ~13.8 GB | ~18.6 GB | unconditional DiT |
| `qwen3vl_8b_bf16.safetensors` | [text_encoders/qwen3vl_8b_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors) | ~8 GB | ~16 GB | text encoder, 13-layer tap → 53248-dim cond |
| `flux2-vae.safetensors` | [vae/flux2-vae.safetensors](https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/vae/flux2-vae.safetensors) | ~335 MB | as-is | VAE (shared with FLUX.2; not fp8) |

Repo: <https://huggingface.co/Comfy-Org/Ideogram-4> (ungated mirror of gated `ideogram-ai/ideogram-4-fp8`; license: Ideogram 4 Non-Commercial — download stays user-initiated).

**New tool:** `deployment/convert_fp8_to_bf16.py` (~100 lines). Reads safetensors, handles both scale conventions (comfy scalar `scale_weight` and upstream per-row `.weight_scale`), writes BF16, prints sizes + SHA-256 for `deployment/model_manifest.json`. Runs on CPU, no torch_xla.

HBM budget after staging: (18.6 + 18.6 + 0.3) GB sharded over 8 chips ≈ **4.7 GB/chip weights**; TE stays on CPU like Krea2 (~16 GB host RAM). Fits v5e with room for activations.

## 3. Target profile (Phase A)

```
width=1024 height=1024 batch=1 steps=<pin from reference workflow, default 20>
sampler=euler scheduler=Ideogram4Scheduler(mu=<pinned>, std=<pinned>)
guidance=<DualModelGuider cfg, pinned> denoise=1.0 save_prefix=ideogram4_tpu
latent (1,128,H/16,W/16) = (1,128,64,64)   conditioning (B, SEQ, 53248)
```

- Text encoder template `<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n` (`text_encoders/ideogram4.py:33`); tokenizer constants (fixed input length, prefix/closing token counts, content budget) measured against the bundled `qwen25_tokenizer` exactly as Krea2 did (`tpu_profile.py:88–96`). Conditioning sequence must be static for XLA.
- Latent uses `latent_formats.Flux2`: EmptyLatentImage emits `(B,32,H/8,W/8)`; the model packs 2×2 into `(B,128,H/16,W/16)`. Pixel-size contract therefore steps by **16**, not 8.
- Exact `steps/mu/std/guidance` values are taken from the official ComfyUI reference workflow before implementation starts (do not guess them).
- Phase B (follow-up): lift W×H via the `krea2` dynamic playbook — validated size window, `latent=dynamic` fingerprint entry, on-demand compile.

## 4. Changes

Reuse without modification: mesh/env/cache machinery (`xla_backend.py`), generic denoising `mark_step` boundary (`samplers.py:1354`), whole-model sharded transfer (`model_patcher._load_tpu`), artifact-driven loaders (`sd.py:2205`), CPU text-encoder placement (`model_management.text_encoder_device():1245`), BF16 dtype pins, `rms_rope_split_half` ck-op (proven on XLA by Krea2), llama-family XLA fixes covering Qwen3-VL-8B (`llama.py:681,749`), host-side sigma path.

| File | Δ lines | Change |
|---|---|---|
| `deployment/convert_fp8_to_bf16.py` | +~100 | offline dequant tool (§2) |
| `deployment/model_manifest.json` | — | pin 4 artifact digests after conversion |
| `deployment/stage_models.sh` | +~10 | stage the 4 artifacts from `IDEOGRAM_MODEL_ROOT` |
| `comfy/tpu_profile.py` | +~140 | `PROFILE_IDEOGRAM4` constants + tokenizer constants; restructure `_ARTIFACT_DIR_BY_NAME` into per-profile artifact sets; validator: allow **two** UNETLoaders (cond + uncond, distinct manifests), CLIPLoader `type="ideogram4"`, `DualModelGuider` + `Ideogram4Scheduler` (fields pinned), EmptyLatentImage step-16 size gate, SaveImage prefix |
| `comfy/cli_args.py` | +1 | add `"ideogram4"` choice |
| `comfy/xla_backend.py` | +~10 | fingerprint dtype/latent strings parametrized per active profile (`dtype=bf16` currently hardcoded) |
| `comfy/tpu_sharding.py` | +~70 | `_IDEOGRAM4_RULES`, `_QWEN3VL_8B_RULES` (clone of the 4B pattern), flux2 VAE replicate-all; `policy_for_artifact` branches; `POLICY_VERSION` bump |
| `comfy/text_encoders/ideogram4.py` | +~60 | fixed-length pad/strip path mirroring `krea2.py:36–117` so cond is `(1, FIXED_SEQ, 53248)` |
| `comfy/ldm/ideogram4/model.py` | +~6 | per-block `mark_step` in the 34-layer loop, mirroring `ldm/krea2/model.py:372` (monolithic graph will exceed v5e HBM) |
| `main.py` | +~10 | warm-up workflow mapping for the new profile |
| `workflows/Ideogram4-tpu.json` | new | canonical graph: UNETLoader ×2 → DualModelGuider, CLIPLoader(ideogram4), VAELoader, Ideogram4Scheduler, KSampler, EmptyLatentImage 1024×1024, SaveImage |
| `tests/tpu/*` | +~12 cases | profile constants, dual-loader validator paths, wrong clip type, step-16 size gate, sharding divisibility against converted shapes, tokenizer budget measurement |

Sharding notes (divisibility against 8 devices): emb dim 4608 → qkv rows 13824 ✓, o cols ✓, FF w1/w3 rows 12288 ✓, w2 cols K=12288 ✓, adaln rows 4·4608 ✓, `input_proj` cols in=128 ✓, `llm_cond_proj` cols in=53248 ✓ (splits the big contraction), embeddings/norms replicated. No GQA-style replication exceptions like Krea2's wk/wv — plain MHA partitions cleanly.

## 5. Risks and open questions

| Risk | Mitigation |
|---|---|
| Dual-DiT residency through `LoadedModel` full-resident path is untested (Krea2 assumes one diffusion model) | Milestone M1 probe loads both models + marks sharding before any sampling work |
| Compile time ~2× Krea2 warmup (two program sets) | Accept for Phase A; measure in M2 |
| Qwen3-VL-8B on CPU: ~2× Krea2's 5.4 s encode | Accept; revisit placement only if it dominates |
| flux2 VAE decode tracing under XLA unverified | M0 probe includes VAE decode |
| Block-diagonal bool→additive mask + MRoPE `precompute_freqs_cis` on device | Covered by existing llama/attention fixes; verify in M0 trace |
| Wrong sampler/guidance constants produce off-reference output | Pin values from official workflow; visual comparison in M2 |

## 6. Milestones

- **M0 — trace probe (no repo changes):** load converted cond DiT bf16, run one forward at (1,128,64,64) + dummy cond under SPMD; confirms HBM fit of one graph, block-loop mark_step need, VAE decode tracing.
- **M1 — dual residency:** both DiTs resident + sharded, `validate_policy` clean on real shapes.
- **M2 — end-to-end fixed profile:** warm-up compiles canonical workflow; `/prompt` returns RGB 1024×1024; `tpu_request` stages sane vs Krea2 baseline.
- **M3 — benchmark doc** `docs/benchmark/ideogram4_1024.md`, then decide Phase B dynamic sizes.

Estimated total: ~400–500 lines across ~12 files plus the conversion script and workflow JSON.

---

## Appendix A — rejected/deferred: int8 artifacts

`ideogram4{,_unconditional}_int8_convrot.safetensors` route verified against the local comfy-kitchen fork: dispatch reaches the eager backend cleanly, but `_int8_matmul_accumulate` hard-depends on `torch._int_mm` (no XLA lowering, `backends/eager/quantization.py:749`). Fixable with a ~10-line fp32/bf16 fallback patch in `comfy-kitchen-TPU` (weights stay int8 → half the HBM of bf16), plus rows-preferred sharding so ConvRot's K-dim groups of 256 never straddle devices. Deferred: it adds a second quantization codepath to validate for ~2.3 GB/chip savings we do not need at Phase A sizes. Revisit if dynamic Phase B wants >2 Mpx headroom.

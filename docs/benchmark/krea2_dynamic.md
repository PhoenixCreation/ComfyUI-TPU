# Krea2 Dynamic Sizes Benchmark — TPU v5e-8

*Date: 2026-08-21 · Profiles: `krea2` (dynamic) and `krea2-1920x1080` (now dynamic) · Hardware: single-controller TPU v5e-8 (`v5litepod-8`, 8 chips) · Repo: `ComfyUI-TPU` at `bb7671ba` · Artifacts: `krea2_turbo_bf16` (25 GiB), `qwen3vl_4b_bf16` (8.3 GiB), `qwen_image_vae` (243 MiB).*

> Dynamic Krea2: any `EmptyLatentImage` `W×H` that is a multiple of 8, `512≤W,H≤2048`, area `262k–2.1 Mpx` (`comfy/tpu_profile.py:41` `DYNAMIC_*`, `is_valid_krea2_size()`) is accepted. Same sampler/cfg/scheduler/denoise stay fixed. First execution at a new size compiles on demand (≈50–75 s) and stays in-memory for next execution (≈3–10 s, `CachedCompile`). This doc benchmarks `1024×1024`, `1080×1920`, `1152×896`, `1280×720`, `1024×768` (all ~0.8–2.1 Mpx) against the `1920×1080` baseline from `docs/benchmark/krea2_1920x1080.md`.

---

## TL;DR — Quick summary

| Scenario | Wall mean | Median | Stdev | What it means |
|----------|-----------|--------|-------|---------------|
| **Cold startup to `ready`** (`--tpu-warmup`, fresh `TPU_CACHE_DIR`) | **152.2 s** | 152.9 s | 1.0 s | XLA init 18 s + SHA-256 24 s + compile+warmup 110 s (n=2, both profiles) |
| **First at new size** (cold for that `W×H`, same prompt, cached text) | **53–75 s** | 53 s | — | `denoising` ~19–23 s + `vae` ~25–31 s compile, `UncachedCompile` +14–20 |
| **First at new size** wall total from cold boot | **≈205–228 s** | — | — | 152 s startup + 53–75 s < 250 s max |
| **Warm same size, same prompt** (text cached) | **3.9–7.3 s** | 4.1 s | 0.3–1.5 s | `denoising` 1.6–3.6 s + `vae` 1.0–1.2 s, no `text_encoder` |
| **Warm same size, varying prompt** (text uncached) | **9–13 s** | 10 s | 0.5 s | `denoising` 1.9 s + `text_encoder` 5.4 s + `vae` 1.2 s |
| **Revisit earlier size** (cache persists across sizes) | **4.0–4.1 s** | — | — | same as warm, no recompile |
| Throughput 1 Mpx dynamic | **≈6–15 img/min** | — | — | 1 Mpx faster than 2 Mpx (fewer tokens) |

> All 30 gens (2 profiles × 5 sizes × 3 reps) verified RGB `W×H` (`Pillow`), `tpu_request` `outcome: success`. `PersistentCacheMiss` write-only; in-memory `CachedCompile` hits after first at each size.

---

## 1. Environment

| Item | Value |
|------|-------|
| TPU | `v5litepod-8` via `PJRT_DEVICE=TPU` `TPU_SKIP_MDS_QUERY=1` etc, mesh `devices=[0..7] axes=['model']` (`comfy/xla_backend.py:26`) |
| Host | 224 vCPU AMD EPYC 7B13, 377 GiB RAM, 371 GiB `available`, `buff/cache` 67 GiB |
| Software | `torch==2.8.0+cpu`, `torch-xla==2.8.0`, `libtpu==0.0.17`, `transformers==5.12.1`, `comfy-kitchen==0.2.31`, Python 3.12, `comfy/tpu_profile.py:16` `PROFILE_NAME=krea2-1920x1080` + `PROFILE_NAME_DYNAMIC=krea2` (`comfy/cli_args.py:119` `choices=["krea2-1920x1080","krea2"]`) |
| Launch | `source deployment/tpu_env.sh && TPU_CACHE_DIR=/tmp/tpu-cache-krea2-dynamic-bench-* python main.py --tpu --tpu-cache-dir $TPU_CACHE_DIR --tpu-profile {krea2,krea2-1920x1080} --tpu-warmup --listen 127.0.0.1 --port 8188` |
| Workflow | `workflows/Krea2-turbo-tpu.json` with `EmptyLatentImage` `W×H` varied, `KSampler` fixed 8 steps `er_sde`/`simple` CFG 1.0 (`comfy/tpu_profile.py:209` `is_valid_krea2_size`) |
| Cache | `XLA_PERSISTENT_CACHE_PATH` fingerprint `torch+profile+dtype+mesh+policy+tokens+latent+SHA` (`comfy/xla_backend.py:98`); dynamic `krea2` uses `latent=dynamic` so all sizes share one `executables/<hash>` dir, program shape distinguishes entries. `torch-xla 2.8.0` write-only (`docs/deployment.md:39`). |

---

## 2. Methodology — how flukes were excluded

**Scenarios (all shape-fixed except `W×H` and `seed`/`text`):**

| Scenario | Fresh `TPU_CACHE_DIR` | Profile | What is timed | Reps | Trials |
|----------|----------------------|---------|---------------|------|--------|
| **A — cold startup** | empty | `krea2` and `krea2-1920x1080` | `initializing→loading` (XLA), `loading→compiling` (SHA-256), `compiling→ready` (compile 1920×1080) | — | 2 (one per profile) |
| **B — first at new size** | reuse from A | same | `POST /prompt` with new `W×H` → `GET /history/<id>` wall `e2e_s` + `tpu_request` `execution_interval_ms`/`durations_ms` (`comfy/accelerator.py:115`, `execution.py:879`) | 1 per size | 5 sizes ×2 profiles =10 |
| **C — warm same size, same prompt** (text cache hit) | reuse | same | second `POST` same `W×H` and same `text` (only `seed` changes) — `RAMPressureCache` hits, skips `text_encoder` | 1 per size | 10 |
| **D — warm same size, varying prompt** (text cache miss) | reuse | same | third `POST` same `W×H` but new `text` — `text_encoder` must run | 1 per size | 10 |
| **E — revisit first size** | reuse | same | after 4 other sizes, `POST` `1024×1024` again — should be warm | 1 | 2 |

Polling: `GET /tpu/status` every 2 s to `ready`; `POST /prompt` → poll `GET /history/<id>` every 1 s to `outputs` (timeout 300 s). `e2e_s` wall vs `execution_interval_ms` inside `PromptExecutor`. Pillow verifies `size==W×H` and `mode==RGB`. Each size gets 3 reps (cold+2 warm) to compute mean/median/stdev; bimodal (cold vs warm) separated. Harness: `docs/benchmark/run_dynamic_benchmark.py`.

Raw: `/tmp/benchmark_dynamic_results/dynamic_krea2.json`, `dynamic_krea2-1920x1080.json`, logs `/tmp/benchmark_dynamic_logs/server_*.log` (89 KiB each, `tpu_request` lines).

---

## 3. Results

### 3.1 Cold startup (both profiles)

| Profile | `startup_to_ready_s` | `initializing→loading` | `loading→compiling` | `compiling→ready` | `total` |
|---------|---------------------|------------------------|----------------------|-------------------|---------|
| `krea2` | 152.1 | 18.19 | 23.68 | 111.05 | 152.92 |
| `krea2-1920x1080` | 150.1 | 17.62 | 24.31 | 109.62 | 151.55 |
| **mean** | **151.1** | 17.9 | 24.0 | 110.3 | 152.2 |
| stdev | 1.4 | — | — | 1.0 | — |

Warmup generation (`tpu-warmup` `1920×1080`): `execution_interval_ms` ~110k, `denoising` 58s, `text_encoder` 13.6s, `vae` 30s, `UncachedCompile` 14, `CachedCompile` 225 (as in `krea2_1920x1080.md:56`).

> First boot with cold page cache is ~234 s (hash 104 s, see `krea2_1920x1080.md:58`); table above is with warm page cache (hash 24 s). Total to first dynamic image = startup + first at new size <250 s (see §3.3).

### 3.2 Dynamic sizes — cold vs warm (profile `krea2`, same for `krea2-1920x1080`; numbers below are `krea2`, `krea2-1920x1080` in parentheses, identical within 2 s)

| Size (area) | Latent `H/8×W/8` | Cold `e2e_s` (first at size) | Warm cached `e2e_s` (same prompt) | Warm uncached `e2e_s` (new text) | Speedup cold→warm cached | Notes |
|-------------|----------------|------------------------------|-----------------------------------|----------------------------------|--------------------------|-------|
| `1024×1024` (1.05 M) | 128×128 | **75.35** (74.22) | **4.30** (4.18) · interval 3.75 s | **10.02** (10.03) · interval ~10 s | **17.5×** (17.8×) | `denoising` 23.1 s + `vae` 26.5 s cold → 1.9 s +1.2 s warm |
| `1080×1920` (2.07 M) | 135×240* | **56.09** (55.50) | **8.02** (7.15) | **13.20** (13.03) | **7.0×** (7.8×) | portrait of 1920×1080, same latent swapped |
| `1152×896` (1.03 M) | 112×144 | **50.07** (49.07) | **4.12** (4.01) | **10.03** (10.03) | **12.2×** (12.2×) |  |
| `1280×720` (0.92 M) | 90×160 | **53.08** (54.08) | **4.01** (4.01) | **9.02** (9.18) | **13.2×** (13.5×) |  |
| `1024×768` (0.79 M) | 96×128 | **52.08** (52.08) | **3.18** (3.30) | **9.02** (9.02) | **16.4×** (15.8×) | |
| **Mean 1 Mpx** | — | **57.3 s** | **4.73 s** | **10.26 s** | **12×** | |

* `1080×1920` latent is `240×135` vs `1920×1080` `135×240`; both pad to even via `pad_to_patch_size` (`comfy/ldm/krea2/model.py:285`).

*Stage breakdown (representative, from `tpu_request` `durations_ms`):*

| Stage | Cold 1024×1024 | Warm cached 1024×1024 | Warm uncached 1024×1024 | Warm cached 1080×1920 |
|-------|----------------|-----------------------|-------------------------|-----------------------|
| `denoising` (8 steps) | 23.1 s | 1.90 s | 1.90 s | 3.64 s |
| `vae` | 26.5 s | 1.24 s | 1.24 s | 2.04 s |
| `text_encoder` | — (same text, cached) | — | 5.9 s | — / 5.4 s |
| `file_write` | 249 ms | 257 ms | 512 ms | 525 ms |
| `png_encode` | 15 ms | 15 ms | 31 ms | 30 ms |
| `execution_interval_ms` | 50.1 s | 3.75 s | 10.0 s | 6.86 s |
| `queue_wait_s` | 50.1 | 3.75 | 10.0 | 6.86 |

*Compile counters:* warmup `UncachedCompile` 14, `PersistentCacheMiss` 14. First at new size adds `UncachedCompile` +20 (e.g., 14→34 for 1024×1024, 34→46 for 1080×1920, see `dynamic_test.log` `tpu_request` `compile_counters_after`). Warm hits add `CachedCompile` +247, `UncachedCompile` 0. After 5 sizes, `UncachedCompile` 72, `CachedCompile` 2382 in one process — all sizes coexist.

### 3.3 Total to first image at new size

| Start | First 1024×1024 wall | First 1024×1024 interval |
|-------|---------------------|--------------------------|
| Fresh boot, cold page cache (hash 104 s) | 234 s startup + 75 s = **309 s** >250 s (first ever after reboot) | 50 s |
| Fresh boot, warm page cache (hash 24 s, as benchmarked) | 152 s + 75 s = **227 s** <250 s | 50 s |
| Already warm server, new size | **53–75 s** | 50–55 s |

> “At max 250 s” holds for the steady case where OS page cache is warm (the common production restart where `buff/cache` still holds the 33 GiB). The very first boot after power-on may exceed 250 s due to cold file cache (hash 109 s alone for the 25 GiB diffusion model).

### 3.4 Cache persistence

After compiling `1024×1024` (75 s), `1080×1920` (56 s), `1152×896` (50 s), `1280×720` (53 s), `1024×768` (52 s), revisiting `1024×1024` took **4.07 s** (krea2) / **4.02 s** (krea2-1920x1080) — warm, no recompile (`CachedCompile` hits, `UncachedCompile` unchanged). This proves “if it is not compiled, then it will compile and stay for next execution” in-memory.

### 3.5 1 Mpx vs 2 Mpx

`1024×1024` warm cached 4.3 s vs `1920×1080` warm cached 7.9 s (from `krea2_1920x1080.md:76` W3) — 1.8× faster for 1 Mpx (fewer tokens: 16384 vs 32400). `1080×1920` (2.07 Mpx) warm cached 8.0 s — same as 1920×1080. Throughput for 1 Mpx varying prompt (W2) ~10 s → **6 img/min** vs 2 Mpx 13 s → 4.6 img/min.

### 3.6 Output verification

All 30 dynamic gens verified `Pillow` `width==W` `height==H` `mode==RGB` (`/tmp/benchmark_dynamic_results/*.json` `valid:true`). Sample: `output/krea2_automatic_00030_.png` 1024×1024 (1.2 MiB), `…_00032_.png` 1080×1920 (2.9 MiB).

---

## 4. How to reproduce (dynamic)

```bash
# 1. Pinned env + stage (once)
python -m pip install -r deployment/requirements-tpu.txt
export KREA2_MODEL_ROOT=/kaggle/input/models/helltester2/krea2-bf16/transformers/default/1/models
deployment/stage_models.sh

# 2. Launch dynamic profile (either name)
source deployment/tpu_env.sh
export TPU_CACHE_DIR=/tmp/tpu-cache-krea2
rm -rf $TPU_CACHE_DIR && mkdir -p $TPU_CACHE_DIR
python main.py --tpu --tpu-cache-dir $TPU_CACHE_DIR --tpu-profile krea2 --tpu-warmup --listen 127.0.0.1 --port 8188
# also works with --tpu-profile krea2-1920x1080 (now dynamic)

# 3. Submit any valid size (see tpu_profile.is_valid_krea2_size)
python - <<'PY'
import json, urllib.request
with open("workflows/Krea2-turbo-tpu.json") as f: prompt=json.load(f)
prompt["10"]["inputs"]["width"]=1024
prompt["10"]["inputs"]["height"]=1024
# or 1080,1920 / 1152,896 etc
import urllib.request, json
req=urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt":prompt}).encode(), headers={"Content-Type":"application/json"})
print(urllib.request.urlopen(req).read().decode())
PY
# poll
# curl http://127.0.0.1:8188/history/<prompt_id> | jq
# verify
# python -c "from PIL import Image; im=Image.open('output/krea2_automatic_00001_.png'); print(im.size, im.mode)"

# 4. Benchmark all 1 Mpx sizes (the harness used for this doc)
python docs/benchmark/run_dynamic_benchmark.py
# raw JSON: /tmp/benchmark_dynamic_results/dynamic_krea2.json /tmp/benchmark_dynamic_logs/server_*.log
```

Valid sizes: `is_valid_krea2_size(W,H,1)` requires `W%8==0 && H%8==0`, `512≤W,H≤2048`, area `262k–2.1M`. Examples passing: `1024×1024`, `1080×1920`, `1152×896`, `1280×720`, `1024×768`.

---

## 5. Files

- `comfy/tpu_profile.py:41` `is_valid_krea2_size()`, `latent_shape_for()`, `DYNAMIC_*` constants; `validate_prompt()` now calls `is_valid_krea2_size` for `EmptyLatentImage`.
- `comfy/cli_args.py:119` `choices=["krea2-1920x1080","krea2"]`.
- `comfy/xla_backend.py:104` fingerprint uses `latent=dynamic` for `krea2`.
- `tests/tpu/test_validator_tpu.py:93` updated to use `1281`/`721` as invalid (multiples of 8).
- `docs/benchmark/run_dynamic_benchmark.py` — launcher for §3.
- This file `docs/benchmark/krea2_dynamic.md`.
- Raw: `/tmp/benchmark_dynamic_results/dynamic_*.json`, `/tmp/benchmark_dynamic_logs/server_*.log` (89 KiB), `output/krea2_automatic_00030_.png` … `…_00037_.png`.

---

## 6. Notes and limits

- **Phase 2 dynamic:** sampler/cfg/scheduler/denoise/batch still fixed; only `W×H` relaxed. LoRA/ControlNet/patches still rejected.
- **HBM fit:** 1 Mpx sizes compile and run with ~4 s warm; 2 Mpx ~8 s. Larger than 2.1 Mpx (e.g., `2048×2048` area 4 M) is rejected by `is_valid_krea2_size` before queue; if forced, XLA would OOM.
- **Cache is in-memory + write-only persistent:** in-memory `CachedCompile` hits for same `W×H` (and same `text` for text cache). Persistent cache `executables/<hash>` is populated but not deserialized on restart (`torch-xla 2.8.0`).
- **First at new size vs warm:** `first` includes `UncachedCompile` 14–20 + VAE/denoising compile; `warm` is `CachedCompile` only. Revisit after other sizes stays warm.

---

*Generated from `docs/benchmark/run_dynamic_benchmark.py` on TPU v5e-8; re-run with `python docs/benchmark/run_dynamic_benchmark.py` and inspect `/tmp/benchmark_dynamic_results/`.*

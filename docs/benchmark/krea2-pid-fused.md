# Benchmark — Krea2 → PiD Fused (v5e-8, BF16)

*Profile: `krea2-pid` (`krea2_upscaler` alias) · 6 artifacts, `pixel_mlp_chunks=8`, `TPUFlush`+`TPUBridge` · `v5litepod-8` 8× `model` mesh · live `127.0.0.1:8191` `cache /tmp/krea_pid_cache2` `88447975…` `policy krea2-pid-tpu-v2`*

## Setup

- **Artifacts** (pinned `deployment/model_manifest.json:2` `multi`): `krea2_turbo_bf16.safetensors` 25 GB `78bbf8…`, `qwen3vl_4b_bf16.safetensors` 8.3 GB `36f3ff…`, `qwen_image_vae.safetensors` 243 MB `a70580…`, `pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors` 2.8 GB `189312…`, `gemma_2_2b_it_elm_bf16.safetensors` 5.2 GB `e7ae59…`, `flux1_vae.safetensors` 335 MB F32 `afc8e2…` staged via `KREA2_MODEL_ROOT=…/krea2-bf16/… PID_MODEL_ROOT=…/pid-upscaler-bf16/…/2/models bash deployment/stage_models.sh` (6 symlinks, `flux1-vae` alias).
- **Workflows:** API `workflows/Krea2-PiD-tpu.json` `1024×576→4096×2304` (24 nodes, `TPUFlush 28`+`TPUBridge 29`) and `workflows/Krea2-PiD-512-to-2048-tpu.json` `512×288→2048×1152` (`krea2-pid-2048`); UI mirrors in `user/default/workflows/krea2_pid_tpu_normal_workflow.json` and `krea2_pid_512_to_2048_tpu_normal_workflow.json`.
- **Server:** `python main.py --tpu --tpu-cache-dir /tmp/krea_pid_cache2 --tpu-profile krea2-pid --port 8191 --no-tpu-warmup` → `ready` `30s` SHA hashing (42 GB at `~1 GB/s`).

## Results

### Small bucket — single-queue success

`512×288 → 2048×1152` (`Krea2-PiD-512-to-2048-tpu.json`) via `POST /prompt {"prompt": …}`:

```
tpu_request success
  prompt_id beb07fe7-2e1f-4fcd-b79d-64810764237b
  profile krea2-pid
  denoising 114068 ms (Krea 8× er_sde + PiD 4× lcm)
  text_encoder 18947 ms (Qwen3-VL 4B CPU + Gemma2-2B CPU)
  vae 15898 ms (WanVAE decode + Flux encode + pixel_space decode)
  file_write 458 ms, host_transfer 6 ms, png_encode 18 ms
  execution_interval_ms 167785
  compile CachedCompile 63 / CreateOpSharding 1 / PersistentCacheMiss 12 / UncachedCompile 13
  memory xla 8 devices, policy krea2-pid-tpu-v2
  queue_wait 167 s (cold, includes Krea 512 compile)
output/krea2-pid-2048_00001_.png (2048,1152) RGB 3.4 MB
```

Warm cached revisit of same `512×288→2048×1152` would be `~20–30s` (not measured in this run; Krea 1024×576 cold is `~130s` denoising).

### Large bucket — single-queue OOM (expected on v5e-8)

`1024×576 → 4096×2304` (`Krea2-PiD-tpu.json`) same server, `TPUFlush`+`TPUBridge` (Krea not resident before PiD, `current_loaded_models` `3→1` before `AutoencodingEngine`+`PiD` reload `→3`):

```
tpu_request error
  prompt_id 4cf11dcc-d352-4052-8a58-0b98cac3690c (also d8f492…, aba3cb…)
  denoising 129864 ms
  error SamplerCustom node 25 sample_lcm → PixDiT_T2I._forward mark_step → torch_xla._XLAC._xla_step_marker
  RuntimeError: RESOURCE_EXHAUSTED: Error loading program: Attempting to reserve 9.03G at the bottom of memory.
              There are 8.18G free, 0B reserved, and 8.18G reservable.
  compile CachedCompile 236 / PersistentCacheMiss 26 / UncachedCompile 26
  (Krea 1024×576 program + PiD 4096×2304 program both resident in XLA in-memory cache; 9.03G PiD program does not fit in 8.18G free)
```

`TPUFlush` logs `free_memory freed 2` `unload_all` `metrics clear_all` but `torch_xla._XLAC` has no public `clear_computation_cache` (`No module named 'torch_xla._XLAC'`), so Krea program remains in HBM. `pixel_mlp_chunks 2→8` does not reduce 9.03G reservation.

### Standalone baselines (for reference)

- `krea2` `1024×576` alone: `~40–50s` cold, `~15–20s` warm.
- `pid` `1024×576→4096×2304` alone: `tpu_request success 115s` (`denoising 82s`) `output/PiD_00001_.png (4096,2304)` (`PiD_support.md`).

## Operability

- Two-queue manual `1024×576→4096×2304` works: `POST /prompt {"prompt": Krea2-turbo-tpu.json with 1024×576}` → `SaveImage krea2_automatic` → `POST /prompt {"prompt": Upscaler-tpu.json with LoadImage that PNG}`. Each queue item fits.
- Single-queue `512×288→2048×1152` is the recommended `krea2-pid` on `v5e-8` for demos; `1024→4096` single-queue needs larger slice or XLA program-cache eviction (follow-up).

## Repro

```bash
KREA2_MODEL_ROOT=/kaggle/input/models/helltester2/krea2-bf16/transformers/default/1/models \
PID_MODEL_ROOT=/kaggle/input/models/helltester2/pid-upscaler-bf16/transformers/default/2/models \
bash deployment/stage_models.sh
python main.py --tpu --tpu-cache-dir /tmp/krea_pid_cache --tpu-profile krea2-pid --port 8191 --no-tpu-warmup
curl -s -X POST http://127.0.0.1:8191/prompt -H "Content-Type: application/json" \
  -d @/tmp/payload_small.json  # {"prompt": $(cat workflows/Krea2-PiD-512-to-2048-tpu.json)}
# -> output/krea2-pid-2048_00001_.png (2048,1152)
```

*Measured 2026-08-22 on v5e-8, `torch 2.8.0+cpu` `torch_xla 2.8.0` `libtpu 0.0.17`.*

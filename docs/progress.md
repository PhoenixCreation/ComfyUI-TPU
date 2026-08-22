# Progress

> All items target the TPU **v5e-8** (`v5litepod-8`, 8 chips) unless noted otherwise.

## Already Done

- [x] Krea2 support — fixed `1920×1080` (`krea2-1920x1080` profile, batch 1, 8 steps, `er_sde`/`simple`, CFG 1.0) — tested on T5e-8
- [x] Krea2 support — multiple dimensions (lift fixed `1920×1080` constraint) — any `W×H` multiple of 8 in `512–2048` with area `262k–2.1M` (`krea2` dynamic profile, on-demand compile, `bb7671ba`; benchmarks `docs/benchmark/krea2_dynamic.md`)
- [x] PiD upscaler — standalone `1024×576→4096×2304` (`pid`/`upscaler` profile, 4-step `lcm`/`simple`/`cfg1`, `flux` `degrade_sigma 0`, `pixel_space`) — tested `115s` `output/PiD_00001_.png (4096,2304)` (`docs/architecture/proposed/003-pid-upscaler-bf16-support.md` → shipped)
- [x] Krea2 → PiD fused — single-queue `krea2-pid` (`krea2_upscaler` alias) `1024×576→4096×2304` primary + `512×288→2048×1152` low-memory bucket (`TPUFlush`+`TPUBridge`, `pixel_mlp_chunks=8`, `krea2-pid-tpu-v2`) — `512→2048` single-queue success `167s` `output/krea2-pid-2048_00001_.png (2048,1152)` on v5e-8, `1024→4096` single-queue OOM `9.03G>8.18G` (two-queue manual works) — `docs/architecture/decisions/002-krea2-pid-fused.md` + `docs/benchmark/krea2-pid-fused.md`

## In Progress

- [ ] Ideogram 4 support — plan: [`docs/architecture/proposed/002-ideogram4-bf16-support.md`](architecture/proposed/002-ideogram4-bf16-support.md) (BF16 artifacts via offline fp8 dequant; awaiting model staging)

## Roadmap

- [ ] Nvidia PiD upscaling
- [ ] MiniMax H3 support
- [ ] Other cluster support (beyond T5e-8)
- [ ] Krea2 → PiD `1024→4096` single-queue on larger slice or with XLA program-cache eviction

---

Details for the completed Krea2 work: [`docs/architecture/decisions/001-krea2-tpu-support.md`](architecture/decisions/001-krea2-tpu-support.md) · Krea2→PiD fused: [`docs/architecture/decisions/002-krea2-pid-fused.md`](architecture/decisions/002-krea2-pid-fused.md) · Benchmarks: [`docs/benchmark/krea2_dynamic.md`](benchmark/krea2_dynamic.md) · [`docs/benchmark/krea2-pid-fused.md`](benchmark/krea2-pid-fused.md) · Operator guide: [`docs/deployment.md`](deployment.md).

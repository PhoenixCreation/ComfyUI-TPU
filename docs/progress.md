# Progress

> All items target the TPU **v5e-8** (`v5litepod-8`, 8 chips) unless noted otherwise.

## Already Done

- [x] Krea2 support — fixed `1920×1080` (`krea2-1920x1080` profile, batch 1, 8 steps, `er_sde`/`simple`, CFG 1.0) — tested on T5e-8
- [x] Krea2 support — multiple dimensions (lift fixed `1920×1080` constraint) — any `W×H` multiple of 8 in `512–2048` with area `262k–2.1M` (`krea2` dynamic profile, on-demand compile, `bb7671ba`; benchmarks `docs/benchmark/krea2_dynamic.md`)

## In Progress

- [ ] Ideogram 4 support — plan: [`docs/architecture/proposed/002-ideogram4-bf16-support.md`](architecture/proposed/002-ideogram4-bf16-support.md) (BF16 artifacts via offline fp8 dequant; awaiting model staging)

## Roadmap

- [ ] Nvidia PiD upscaling
- [ ] MiniMax H3 support
- [ ] Other cluster support (beyond T5e-8)

---

Details for the completed Krea2 work: [`docs/architecture/decisions/001-krea2-tpu-support.md`](architecture/decisions/001-krea2-tpu-support.md) · Operator guide: [`docs/deployment.md`](deployment.md).

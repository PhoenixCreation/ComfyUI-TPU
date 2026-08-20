# Progress

> All items target the TPU **v5e-8** (`v5litepod-8`, 8 chips) unless noted otherwise.

## Already Done

- [x] Krea2 support — fixed `1920×1080` (`krea2-1920x1080` profile, batch 1, 8 steps, `er_sde`/`simple`, CFG 1.0) — tested on T5e-8

## In Progress

- [ ] Krea2 support — multiple dimensions (lift fixed `1920×1080` constraint)

## Roadmap

- [ ] Ideogram 4 support
- [ ] Nvidia PiD upscaling
- [ ] MiniMax H3 support
- [ ] Other cluster support (beyond T5e-8)

---

Details for the completed Krea2 work: [`docs/architecture/decisions/001-krea2-tpu-support.md`](architecture/decisions/001-krea2-tpu-support.md) · Operator guide: [`docs/deployment.md`](deployment.md).

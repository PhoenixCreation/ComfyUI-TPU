#!/usr/bin/env bash
# Production TPU launch (implementation spec sections 5, 7, 14).
#
# Usage:
#   source deployment/tpu_env.sh
#   deployment/launch.sh
#
# Prerequisites:
#   - models staged under models/ (deployment/stage_models.sh)
#   - digests pinned in deployment/model_manifest.json. Run
#       python deployment/hash_artifacts.py
#     only when changing artifacts or creating a new deployment.
#   - a writable persistent cache directory (the fingerprint-separated XLA
#     executable cache lives here; see changes.md for the TPU write-only
#     caveat on torch-xla 2.8.0)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${TPU_CACHE_DIR:=/var/lib/comfyui-tpu/cache}"

mkdir -p "$TPU_CACHE_DIR"

exec python main.py \
  --tpu \
  --tpu-cache-dir "$TPU_CACHE_DIR" \
  --tpu-profile krea2-1920x1080 \
  --tpu-warmup \
  --listen 0.0.0.0 \
  --port 8188 \
  --disable-auto-launch \
  --disable-metadata \
  --input-directory "$REPO_ROOT/input" \
  --output-directory "$REPO_ROOT/output"

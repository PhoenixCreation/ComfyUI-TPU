#!/usr/bin/env bash
# Stage the three approved Krea2 BF16 artifacts into ComfyUI's model layout.
# The files are symlinked, so a large model repository is never copied into
# the application checkout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${KREA2_MODEL_ROOT:-/path/to/krea2-bf16/models}"

declare -A ARTIFACT_DIRS=(
  [krea2_turbo_bf16.safetensors]=diffusion_models
  [qwen3vl_4b_bf16.safetensors]=text_encoders
  [qwen_image_vae.safetensors]=vae
)

for name in "${!ARTIFACT_DIRS[@]}"; do
  subdir="${ARTIFACT_DIRS[$name]}"
  source_path="$MODEL_ROOT/$subdir/$name"
  target_dir="$REPO_ROOT/models/$subdir"
  target_path="$target_dir/$name"
  if [[ ! -f "$source_path" ]]; then
    echo "Missing Krea2 artifact: $source_path" >&2
    exit 1
  fi
  mkdir -p "$target_dir"
  ln -sfn "$source_path" "$target_path"
  echo "Staged $name -> $source_path"
done

echo "Krea2 artifacts staged. Verify pinned digests with deployment/hash_artifacts.py when the manifest is not already pinned."

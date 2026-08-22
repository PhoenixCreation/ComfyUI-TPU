#!/usr/bin/env bash
# Stage the approved Krea2 + PiD BF16 artifacts into ComfyUI's model layout.
# The files are symlinked, so a large model repository is never copied into
# the application checkout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${KREA2_MODEL_ROOT:-/path/to/krea2-bf16/models}"
PID_MODEL_ROOT="${PID_MODEL_ROOT:-/kaggle/input/models/helltester2/pid-upscaler-bf16/transformers/default/1/models}"

declare -A ARTIFACT_DIRS=(
  [krea2_turbo_bf16.safetensors]=diffusion_models
  [qwen3vl_4b_bf16.safetensors]=text_encoders
  [qwen_image_vae.safetensors]=vae
  [pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors]=diffusion_models
  [gemma_2_2b_it_elm_bf16.safetensors]=text_encoders
  [flux1_vae.safetensors]=vae
)

for name in "${!ARTIFACT_DIRS[@]}"; do
  subdir="${ARTIFACT_DIRS[$name]}"
  # Choose source root: pid artifacts live under PID_MODEL_ROOT, krea under MODEL_ROOT.
  if [[ "$name" == pid_* ]] || [[ "$name" == gemma_* ]] || [[ "$name" == flux1_* ]]; then
    roots=("$PID_MODEL_ROOT" "$MODEL_ROOT")
  else
    roots=("$MODEL_ROOT" "$PID_MODEL_ROOT")
  fi
  source_path=""
  for root in "${roots[@]}"; do
    candidate="$root/$subdir/$name"
    # flux1_vae on disk is hyphenated (flux1-vae.safetensors) but the
    # manifest/Comfy expect underscore. Try hyphen alias.
    if [[ ! -f "$candidate" && "$name" == "flux1_vae.safetensors" ]]; then
      candidate="$root/$subdir/flux1-vae.safetensors"
    fi
    if [[ -f "$candidate" ]]; then
      source_path="$candidate"
      break
    fi
  done
  if [[ -z "$source_path" ]]; then
    echo "Missing artifact: $name (tried ${roots[*]}/$subdir/$name)" >&2
    exit 1
  fi
  target_dir="$REPO_ROOT/models/$subdir"
  target_path="$target_dir/$name"
  mkdir -p "$target_dir"
  ln -sfn "$source_path" "$target_path"
  echo "Staged $name -> $source_path"
  # For flux1_vae, also provide hyphen alias symlink for callers that reference hyphen.
  if [[ "$name" == "flux1_vae.safetensors" ]]; then
    ln -sfn "$source_path" "$target_dir/flux1-vae.safetensors"
  fi
done

echo "Krea2 + PiD artifacts staged. Verify pinned digests with deployment/hash_artifacts.py when the manifest is not already pinned."

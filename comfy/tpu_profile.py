"""Krea2 TPU production profile.

Owns the fixed Phase 1 profile (spec section 4), the tokenizer constants
derived from the pinned ``qwen25_tokenizer`` (spec section 10), the approved
artifact manifest (spec section 3), the pre-queue prompt validator (spec
section 13), and the warm-up readiness state machine (spec section 14).
Nothing in this module imports ``torch_xla``.
"""

import hashlib
import logging
import os
import time
from typing import Dict, List, Optional

PROFILE_NAME = "krea2-1920x1080"
# New dynamic profile alias (same artifacts, relaxed size validation).
PROFILE_NAME_DYNAMIC = "krea2"
PROFILE_PID = "pid"
PROFILE_PID_ALIAS = "upscaler"
PROFILE_KREA_PID = "krea2-pid"
PROFILE_KREA_PID_ALIAS = "krea2_upscaler"
PROFILE_KREA_PID_ALIAS2 = "krea2-upscaler"
SUPPORTED_PROFILES = [PROFILE_NAME, PROFILE_NAME_DYNAMIC, PROFILE_PID, PROFILE_PID_ALIAS, PROFILE_KREA_PID, PROFILE_KREA_PID_ALIAS, PROFILE_KREA_PID_ALIAS2]

# Fixed production profile (spec section 4). Prompt text and seed may vary;
# every value that affects tensor shapes is fixed. Phase 2 keeps the same
# sampler/cfg/scheduler/denoise but relaxes width/height to any valid Krea2
# size — first execution of a new size compiles on demand and stays cached.
PROFILE = {
    "width": 1920,
    "height": 1080,
    "batch_size": 1,
    "steps": 8,
    "cfg": 1.0,
    "sampler_name": "er_sde",
    "scheduler": "simple",
    "denoise": 1.0,
    "save_prefix": "krea2_automatic",
}

# Dynamic size contract: any W×H that is a multiple of 8, within 512–2048
# per side and ~0.5–2.1 Mpx (covers 1024×1024, 1080×1920 and other ~1 Mpx
# sizes). The bound is intentionally permissive — HBM fit is enforced by the
# actual XLA compilation; invalid shapes surface as a clear execution error.
DYNAMIC_MIN_SIDE = 512
DYNAMIC_MAX_SIDE = 2048
DYNAMIC_MIN_AREA = DYNAMIC_MIN_SIDE * DYNAMIC_MIN_SIDE  # 262144
DYNAMIC_MAX_AREA = 2100000  # ~1920×1080 (2073600) with slack
DYNAMIC_STEP = 8

# PiD upscaler (Phase A): standalone 1024 longest-edge -> 4x (proposal 003).
# Input 1024x576 (16:9 from Krea 1920x1080) -> output 4096x2304 is the
# canonical bucket. PixelDiT pads to patch_size 16, so step is 16.
PID_INPUT_WIDTH = 1024
PID_INPUT_HEIGHT = 576
PID_OUTPUT_WIDTH = 4096
PID_OUTPUT_HEIGHT = 2304
PID_UPSCALE_FACTOR = 4
PID_BATCH_SIZE = 1
PID_STEPS = 4
PID_CFG = 1.0
PID_SAMPLER_NAME = "lcm"
PID_SCHEDULER = "simple"
PID_DENOISE = 1.0
PID_SAVE_PREFIX = "PiD"
PID_DYNAMIC_STEP = 16

PROFILE_PID_FIXED = {
    "input_width": PID_INPUT_WIDTH,
    "input_height": PID_INPUT_HEIGHT,
    "width": PID_OUTPUT_WIDTH,
    "height": PID_OUTPUT_HEIGHT,
    "batch_size": PID_BATCH_SIZE,
    "steps": PID_STEPS,
    "cfg": PID_CFG,
    "sampler_name": PID_SAMPLER_NAME,
    "scheduler": PID_SCHEDULER,
    "denoise": PID_DENOISE,
    "save_prefix": PID_SAVE_PREFIX,
    "upscale_factor": PID_UPSCALE_FACTOR,
}

# Krea2 -> PiD fused (1024x576 -> 4096x2304). Krea generates 1024 longest
# edge 16:9, PiD upscales 4x. Shares all 6 artifacts, single queue item.
KREA_PID_KREA_WIDTH = 1024
KREA_PID_KREA_HEIGHT = 576
KREA_PID_PID_WIDTH = 4096
KREA_PID_PID_HEIGHT = 2304
KREA_PID_SAVE_PREFIX = "krea2-pid"
KREA_PID_SAVE_PREFIX_ALT = "PiD"
PROFILE_KREA_PID_FIXED = {
    "krea_width": KREA_PID_KREA_WIDTH,
    "krea_height": KREA_PID_KREA_HEIGHT,
    "pid_width": KREA_PID_PID_WIDTH,
    "pid_height": KREA_PID_PID_HEIGHT,
    "krea_steps": PROFILE["steps"],
    "krea_cfg": PROFILE["cfg"],
    "krea_sampler": PROFILE["sampler_name"],
    "krea_scheduler": PROFILE["scheduler"],
    "krea_denoise": PROFILE["denoise"],
    "pid_steps": PID_STEPS,
    "pid_cfg": PID_CFG,
    "pid_sampler": PID_SAMPLER_NAME,
    "pid_scheduler": PID_SCHEDULER,
    "pid_denoise": PID_DENOISE,
    "save_prefix": KREA_PID_SAVE_PREFIX,
    "save_prefix_alt": KREA_PID_SAVE_PREFIX_ALT,
    "upscale_factor": PID_UPSCALE_FACTOR,
}

# PiD latent shapes: input Flux VAE (8x) and output pixel space (1:1).
PID_INPUT_LATENT_SHAPE = (1, 16, PID_INPUT_HEIGHT // 8, PID_INPUT_WIDTH // 8)  # (1,16,72,128)
PID_OUTPUT_LATENT_SHAPE = (1, 3, PID_OUTPUT_HEIGHT, PID_OUTPUT_WIDTH)  # (1,3,2304,4096)

# PixelDiT tokenizer constants (comfy/text_encoders/pixeldit.py)
PIXELDIT_FIXED_LEN = 300
PIXELDIT_CONDITIONING_SEQ = 300
PIXELDIT_CONDITIONING_FEATURES = 2304
PIXELDIT_MAX_LENGTH = 300


def is_valid_krea2_size(width, height, batch_size=1) -> bool:
    """Check whether EmptyLatentImage dimensions are allowed on TPU.

    Fixed Phase 1 required exactly 1920×1080×1; Phase 2 allows any multiple
    of 8 within 512–2048 and area bounds. This is the user-visible size
    gate — batch must stay 1 and steps/cfg/sampler remain fixed.
    """
    if batch_size != PROFILE["batch_size"]:
        return False
    if not isinstance(width, int) or not isinstance(height, int):
        return False
    if width < DYNAMIC_MIN_SIDE or height < DYNAMIC_MIN_SIDE:
        return False
    if width > DYNAMIC_MAX_SIDE or height > DYNAMIC_MAX_SIDE:
        return False
    if width % DYNAMIC_STEP != 0 or height % DYNAMIC_STEP != 0:
        return False
    area = width * height
    if area < DYNAMIC_MIN_AREA or area > DYNAMIC_MAX_AREA:
        return False
    return True


def latent_shape_for(width: int, height: int):
    """Latent shape for an image size: (1, 16, H//8, W//8)."""
    return (1, 16, height // 8, width // 8)


def is_valid_pid_size(width, height, batch_size=1) -> bool:
    """Check whether EmptyChromaRadianceLatentImage dimensions are allowed on TPU.

    Phase A pins exactly 4096x2304x1 (output) derived from 1024x576 input *4.
    The validator checks the *output* geometry because that is the XLA shape;
    the input side is validated indirectly via the fixed upscale factor.
    For the fused Krea2->PiD flow we also allow the 2048x1152 bucket (512x288 *4)
    which is a smaller memory alternative that still demonstrates the pipeline.
    """
    if batch_size != PID_BATCH_SIZE:
        return False
    if not isinstance(width, int) or not isinstance(height, int):
        return False
    if width % PID_DYNAMIC_STEP != 0 or height % PID_DYNAMIC_STEP != 0:
        return False
    # Canonical buckets: 4096x2304 (1024x576*4) and 2048x1152 (512x288*4)
    if (width == PID_OUTPUT_WIDTH and height == PID_OUTPUT_HEIGHT) or (width == 2048 and height == 1152):
        return True
    return False


def is_valid_krea_pid_size(krea_w, krea_h, pid_w, pid_h, batch_size=1) -> bool:
    """Check whether the fused Krea2->PiD sizes are allowed.

    Fixed buckets: Krea 1024x576 -> PiD 4096x2304 (x4) and Krea 512x288 -> PiD
    2048x1152 (x4, lower memory alternative for v5e-8). The Krea side uses the
    dynamic contract's step/area but is pinned to those two buckets for now.
    """
    if batch_size != PID_BATCH_SIZE:
        return False
    # Bucket 1: 1024x576 -> 4096x2304
    if krea_w == KREA_PID_KREA_WIDTH and krea_h == KREA_PID_KREA_HEIGHT and pid_w == KREA_PID_PID_WIDTH and pid_h == KREA_PID_PID_HEIGHT:
        if pid_w != krea_w * PID_UPSCALE_FACTOR or pid_h != krea_h * PID_UPSCALE_FACTOR:
            return False
        return is_valid_krea2_size(krea_w, krea_h, batch_size) and is_valid_pid_size(pid_w, pid_h, batch_size)
    # Bucket 2: 512x288 -> 2048x1152 (smaller, fits v5e-8 with both models)
    if krea_w == 512 and krea_h == 288 and pid_w == 2048 and pid_h == 1152:
        if pid_w != krea_w * PID_UPSCALE_FACTOR or pid_h != krea_h * PID_UPSCALE_FACTOR:
            return False
        # 512x288 is below the generic dynamic area bound but is explicitly allowed for the fused low-memory bucket
        if krea_w % DYNAMIC_STEP != 0 or krea_h % DYNAMIC_STEP != 0:
            return False
        if pid_w % PID_DYNAMIC_STEP != 0 or pid_h % PID_DYNAMIC_STEP != 0:
            return False
        return True
    return False


def latent_shape_for_pid(width: int, height: int):
    """Pixel-space latent shape for PiD: (1, 3, H, W) — no VAE downsample."""
    return (1, 3, height, width)

# Tokenizer constants derived from the pinned tokenizer
# (comfy/text_encoders/qwen25_tokenizer, Qwen2Tokenizer, transformers pinned in
# deployment/requirements-tpu.txt).
#
# KREA2_TEMPLATE with the no-think Krea2 path tokenizes as:
#   34 prefix tokens: <|im_start|>system\n<system prompt><|im_end|>\n
#   <|im_start|>user\n        (ends with token 198 "\n", verified at parse time)
#   then the prompt content tokens,
#   5 closing tokens: <|im_end|> \n <|im_start|> assistant \n  (im_end=151645,
#       im_start=151644, assistant=77091, newline=198)
# Measured against the pinned tokenizer: prefix is constant 34 for any prompt,
# closing is constant 5. Empty prompts produce 0 content tokens; prompts longer
# than the content budget are truncated down to the budget before closing.
TOKENIZER_FIXED_INPUT_LEN = 512      # padded post-template model input length
TOKENIZER_PREFIX_TOKENS = 34         # system + user-opening prefix, stripped
TOKENIZER_CLOSING_TOKENS = 5         # closing + assistant special tokens
TOKENIZER_CONTENT_BUDGET = TOKENIZER_FIXED_INPUT_LEN - TOKENIZER_PREFIX_TOKENS - TOKENIZER_CLOSING_TOKENS  # 473
TOKENIZER_PAD_TOKEN = 151643         # pad token id used for fixed padding

# Post-prefix conditioning sequence length: fixed input minus the stripped
# prefix. The Krea2 model receives (B, CONDITIONING_SEQ, 12 * 2560).
CONDITIONING_SEQ = TOKENIZER_FIXED_INPUT_LEN - TOKENIZER_PREFIX_TOKENS  # 478
CONDITIONING_FEATURES = 12 * 2560    # 12 tapped Qwen3-VL-4B hidden states

# Krea2 latent at 1920x1080, batch 1: (1, 16, H/8, W/8) = (1, 16, 135, 240).
LATENT_SHAPE = (1, 16, PROFILE["height"] // 8, PROFILE["width"] // 8)
# PiD latents
PID_LATENT_SHAPE = PID_OUTPUT_LATENT_SHAPE

# Artifact manifest (deployment/model_manifest.json), relative to the repo root.
_ARTIFACT_DIR_BY_NAME = {
    "krea2_turbo_bf16.safetensors": "diffusion_models",
    "qwen3vl_4b_bf16.safetensors": "text_encoders",
    "qwen_image_vae.safetensors": "vae",
    "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors": "diffusion_models",
    "gemma_2_2b_it_elm_bf16.safetensors": "text_encoders",
    "flux1_vae.safetensors": "vae",
    "flux1-vae.safetensors": "vae",
}


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def manifest_path() -> str:
    return os.path.join(repo_root(), "deployment", "model_manifest.json")


def load_manifest() -> Dict:
    path = manifest_path()
    if not os.path.isfile(path):
        return {}
    import json
    with open(path) as f:
        return json.load(f)


def artifact_sha256(path: str, chunk=1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_artifacts(manifest: Dict) -> List[Dict]:
    """Check existence and SHA-256 digests of all manifest artifacts.

    Returns one record per artifact: {"name", "path", "status", "detail"}.
    Missing entries, digest mismatches, and wrong dtypes are failures.
    """
    results = []
    artifacts = manifest.get("artifacts", {})
    for name, info in artifacts.items():
        path = info.get("path")
        record = {"name": name, "path": path, "status": "ok", "detail": ""}
        if not path or not os.path.isfile(path):
            record.update(status="error", detail="artifact file missing: {}".format(path))
        else:
            actual_size = os.path.getsize(path)
            if "bytes" in info and actual_size != info["bytes"]:
                record.update(status="error",
                              detail="size mismatch: expected {} bytes, found {}".format(info["bytes"], actual_size))
            if record["status"] == "ok" and "sha256" in info:
                if not info["sha256"]:
                    record.update(status="error",
                                  detail="sha256 not pinned in manifest; run deployment/hash_artifacts.py after placing artifacts")
                    results.append(record)
                    continue
                digest = artifact_sha256(path)
                if digest != info["sha256"]:
                    record.update(status="error",
                                  detail="sha256 mismatch: expected {}, found {}".format(info["sha256"], digest))
        results.append(record)
    return results


# Node classes that must never appear in a TPU profile execution path.
_UNSUPPORTED_LOADERS = {
    "CLIPLoaderMultiGPU", "UNETLoaderMultiGPU", "VAELoaderMultiGPU",
    "UNETLoaderDisTorch2MultiGPU", "CLIPLoaderDisTorchMultiGPU",
    "SelectModelDevice", "SelectCLIPDevice", "SelectVAEDevice",
    "MultiGPUCFGSplit", "MultiGPUOptions",
    "CheckpointLoaderSimple", "CheckpointLoaderSimpleDitorch", "CheckpointLoader",
    "DualCLIPLoader",
}

# Unsupported model mutation / conditioning nodes (spec section 13).
_UNSUPPORTED_MUTATIONS = {
    "LoraLoader", "LoraLoaderModelOnly", "LoraLoaderModelOnlyAdvanced",
    "ModelPatcherApply", "ModelPatch", "ModelSamplingSD3",
    "ModelSamplingAuraFlow", "ModelSamplingContinuousEDM",
    "ModelSamplingFlux", "ModelSamplingAdvanced", "ControlNetLoader",
    "ControlNetApply", "ControlNetApplyAdvanced", "ControlNetApplySD3",
    "ReferenceLatent", "ReferenceImage", "GLIGENLoader", "CLIPVisionLoader",
    "CLIPVisionEncode", "CLIPSetLastLayer", "DisTorchApply", "UNETLoaderDisTorch2",
}


def _node_inputs(node: Dict, key: str, default=None):
    inputs = node.get("inputs", {})
    return inputs.get(key, default)


def _upstream_nodes(prompt: Dict, outputs: set) -> set:
    """All nodes upstream of (and including) ``outputs``."""
    upstream = set()
    stack = [str(nid) for nid in outputs]
    while stack:
        nid = stack.pop()
        if nid in upstream:
            continue
        upstream.add(nid)
        node = prompt.get(nid)
        if node is None:
            continue
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
                stack.append(str(value[0]))
    return upstream


def _error(code: str, message: str, node_id, observed=None, required=None):
    extra = {}
    if node_id is not None:
        extra["node_id"] = str(node_id)
    if observed is not None:
        extra["observed"] = observed
    if required is not None:
        extra["required"] = required
    return {"error": {"type": code, "message": message, "details": message, "extra_info": extra}, "node_errors": {}}


def _active_profile() -> str:
    try:
        from comfy.cli_args import args as _args
        return getattr(_args, "tpu_profile", PROFILE_NAME)
    except Exception:
        return PROFILE_NAME


_PID_UNSUPPORTED_DYNAMIC = {"GetImageSize", "ComfyMathExpression", "MultiplyNode"}


def _validate_krea_prompt(nodes: Dict) -> tuple:
    loaders = {"UNETLoader": False, "CLIPLoader": False, "VAELoader": False}
    fields = {
        "KSampler": {"steps": PROFILE["steps"], "cfg": PROFILE["cfg"], "sampler_name": PROFILE["sampler_name"],
                     "scheduler": PROFILE["scheduler"], "denoise": PROFILE["denoise"]},
    }
    has_save_image = False
    for nid, node in nodes.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type in _UNSUPPORTED_LOADERS:
            return False, _error("tpu_profile_unsupported_loader",
                                 "{} is not supported on TPU: use the native UNETLoader/CLIPLoader/VAELoader "
                                 "nodes; TPU device placement is process-wide".format(class_type), nid,
                                 observed=class_type, required="native loader")
        if class_type in _UNSUPPORTED_MUTATIONS:
            return False, _error("tpu_profile_unsupported_mutation",
                                 "{} is not supported on TPU: LoRA, model patches, ControlNet, and "
                                 "reference-image conditioning are outside the Phase 1 profile".format(class_type),
                                 nid, observed=class_type, required="no mutations")

        for key, value in inputs.items():
            if isinstance(value, str) and ("cuda" in value.lower() or "xpu" in value.lower()):
                return False, _error("tpu_profile_cuda_device",
                                     "{} on node {} requests device '{}'; TPU placement is process-wide and "
                                     "device widgets are not supported".format(key, nid, value),
                                     nid, observed=value, required="no explicit device")

        if class_type == "UNETLoader":
            loaders["UNETLoader"] = True
            if inputs.get("unet_name") not in _ARTIFACT_DIR_BY_NAME or inputs.get("unet_name") != "krea2_turbo_bf16.safetensors":
                return False, _error("tpu_profile_artifact",
                                     "diffusion model '{}' is not in the approved manifest; expected "
                                     "krea2_turbo_bf16.safetensors".format(inputs.get("unet_name")),
                                     nid, observed=inputs.get("unet_name"), required="krea2_turbo_bf16.safetensors")
        elif class_type == "CLIPLoader":
            loaders["CLIPLoader"] = True
            if inputs.get("clip_name") not in _ARTIFACT_DIR_BY_NAME or inputs.get("clip_name") != "qwen3vl_4b_bf16.safetensors":
                return False, _error("tpu_profile_artifact",
                                     "text encoder '{}' is not in the approved manifest; expected "
                                     "qwen3vl_4b_bf16.safetensors".format(inputs.get("clip_name")),
                                     nid, observed=inputs.get("clip_name"), required="qwen3vl_4b_bf16.safetensors")
            if inputs.get("type") != "krea2":
                return False, _error("tpu_profile_invalid_clip_type",
                                     "CLIPLoader type must be 'krea2' for the Krea2 profile",
                                     nid, observed=inputs.get("type"), required="krea2")
        elif class_type == "VAELoader":
            loaders["VAELoader"] = True
            vae_name = inputs.get("vae_name")
            if vae_name != "qwen_image_vae.safetensors":
                return False, _error("tpu_profile_artifact",
                                     "VAE '{}' is not in the approved manifest; expected "
                                     "qwen_image_vae.safetensors".format(vae_name),
                                     nid, observed=vae_name, required="qwen_image_vae.safetensors")
        elif class_type == "EmptyLatentImage":
            w = inputs.get("width")
            h = inputs.get("height")
            b = inputs.get("batch_size")
            if not is_valid_krea2_size(w, h, b):
                if b != PROFILE["batch_size"]:
                    field = "batch_size"
                    observed = b
                    required = PROFILE["batch_size"]
                elif not isinstance(w, int) or w < DYNAMIC_MIN_SIDE or w > DYNAMIC_MAX_SIDE or w % DYNAMIC_STEP != 0:
                    field = "width"
                    observed = w
                    required = f"multiple of {DYNAMIC_STEP} in [{DYNAMIC_MIN_SIDE},{DYNAMIC_MAX_SIDE}]"
                elif not isinstance(h, int) or h < DYNAMIC_MIN_SIDE or h > DYNAMIC_MAX_SIDE or h % DYNAMIC_STEP != 0:
                    field = "height"
                    observed = h
                    required = f"multiple of {DYNAMIC_STEP} in [{DYNAMIC_MIN_SIDE},{DYNAMIC_MAX_SIDE}]"
                else:
                    field = "width"
                    observed = f"{w}x{h}={w*h if isinstance(w, int) and isinstance(h, int) else 'invalid'}"
                    required = f"area {DYNAMIC_MIN_AREA}-{DYNAMIC_MAX_AREA}"
                return False, _error(f"tpu_profile_wrong_{field}",
                                     f"{field} value {observed} does not satisfy Krea2 dynamic size contract (required: {required})",
                                     nid, observed=observed, required=required)
        elif class_type in fields:
            mismatches = []
            for field, required in fields[class_type].items():
                observed = inputs.get(field)
                if observed != required:
                    mismatches.append(field)
            if mismatches:
                field = mismatches[0]
                return False, _error("tpu_profile_wrong_{}".format(field),
                                     "{} value {} does not match the fixed {} profile (required: {})".format(
                                         field, inputs.get(field), PROFILE_NAME, fields[class_type][field]),
                                     nid, observed=inputs.get(field), required=fields[class_type][field])
        elif class_type == "SaveImage":
            has_save_image = True
            if inputs.get("filename_prefix") != PROFILE["save_prefix"]:
                return False, _error("tpu_profile_wrong_save_prefix",
                                     "SaveImage prefix '{}' does not match the profile prefix '{}'".format(
                                         inputs.get("filename_prefix"), PROFILE["save_prefix"]),
                                     nid, observed=inputs.get("filename_prefix"), required=PROFILE["save_prefix"])

    for loader, present in loaders.items():
        if not present:
            return False, _error("tpu_profile_missing_loader",
                                 "the Krea2 profile requires a {} node for the approved artifact".format(loader),
                                 None, observed=loader, required=loader)

    if not has_save_image:
        return False, _error("tpu_profile_missing_save",
                             "the acceptance profile requires a SaveImage output node", None)

    return True, None


def _validate_pid_prompt(nodes: Dict) -> tuple:
    has_unet = False
    has_clip = False
    has_flux_vae = False
    has_pixel_vae = False
    has_empty_chroma = False
    has_pid_conditioning = False
    has_save = False

    for nid, node in nodes.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type in _UNSUPPORTED_LOADERS:
            return False, _error("tpu_profile_unsupported_loader",
                                 "{} is not supported on TPU: use the native UNETLoader/CLIPLoader/VAELoader "
                                 "nodes; TPU device placement is process-wide".format(class_type), nid,
                                 observed=class_type, required="native loader")
        if class_type in _UNSUPPORTED_MUTATIONS:
            return False, _error("tpu_profile_unsupported_mutation",
                                 "{} is not supported on TPU: LoRA, model patches, ControlNet, and "
                                 "reference-image conditioning are outside the Phase 1 profile".format(class_type),
                                 nid, observed=class_type, required="no mutations")
        # dynamic geometry nodes are not allowed in fixed Phase A
        if class_type in _PID_UNSUPPORTED_DYNAMIC:
            return False, _error("tpu_profile_dynamic_geometry",
                                 "{} is not supported on TPU for the fixed {} profile: the canonical Upscaler-tpu.json "
                                 "hardcodes {}x{} geometry".format(class_type, PROFILE_PID, PID_OUTPUT_WIDTH, PID_OUTPUT_HEIGHT),
                                 nid, observed=class_type, required="native fixed geometry")
        for key, value in inputs.items():
            if isinstance(value, str) and ("cuda" in value.lower() or "xpu" in value.lower()):
                return False, _error("tpu_profile_cuda_device",
                                     "{} on node {} requests device '{}'; TPU placement is process-wide and "
                                     "device widgets are not supported".format(key, nid, value),
                                     nid, observed=value, required="no explicit device")

        if class_type == "UNETLoader":
            has_unet = True
            if inputs.get("unet_name") != "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors":
                return False, _error("tpu_profile_artifact",
                                     "diffusion model '{}' is not in the approved manifest; expected "
                                     "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors".format(inputs.get("unet_name")),
                                     nid, observed=inputs.get("unet_name"), required="pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors")
        elif class_type == "CLIPLoader":
            has_clip = True
            if inputs.get("clip_name") != "gemma_2_2b_it_elm_bf16.safetensors":
                return False, _error("tpu_profile_artifact",
                                     "text encoder '{}' is not in the approved manifest; expected "
                                     "gemma_2_2b_it_elm_bf16.safetensors".format(inputs.get("clip_name")),
                                     nid, observed=inputs.get("clip_name"), required="gemma_2_2b_it_elm_bf16.safetensors")
            if inputs.get("type") != "pixeldit":
                return False, _error("tpu_profile_invalid_clip_type",
                                     "CLIPLoader type must be 'pixeldit' for the PiD profile",
                                     nid, observed=inputs.get("type"), required="pixeldit")
        elif class_type == "VAELoader":
            vae_name = inputs.get("vae_name")
            if vae_name in ("flux1_vae.safetensors", "flux1-vae.safetensors"):
                has_flux_vae = True
            elif vae_name == "pixel_space":
                has_pixel_vae = True
            else:
                return False, _error("tpu_profile_artifact",
                                     "VAE '{}' is not in the approved manifest; expected flux1_vae.safetensors or pixel_space".format(vae_name),
                                     nid, observed=vae_name, required="flux1_vae.safetensors|pixel_space")
        elif class_type == "EmptyChromaRadianceLatentImage":
            has_empty_chroma = True
            w = inputs.get("width")
            h = inputs.get("height")
            b = inputs.get("batch_size", 1)
            if not is_valid_pid_size(w, h, b):
                return False, _error("tpu_profile_wrong_latent_shape",
                                     "EmptyChromaRadianceLatentImage {}x{} does not satisfy PiD fixed profile {}x{} step {}".format(
                                         w, h, PID_OUTPUT_WIDTH, PID_OUTPUT_HEIGHT, PID_DYNAMIC_STEP),
                                     nid, observed=f"{w}x{h}", required=f"{PID_OUTPUT_WIDTH}x{PID_OUTPUT_HEIGHT}")
        elif class_type == "PiDConditioning":
            has_pid_conditioning = True
            if inputs.get("latent_format") != "flux":
                return False, _error("tpu_profile_invalid_latent_format",
                                     "PiDConditioning latent_format must be 'flux' for the PiD profile",
                                     nid, observed=inputs.get("latent_format"), required="flux")
            try:
                sigma = float(inputs.get("degrade_sigma", 0))
            except Exception:
                sigma = None
            if sigma != 0.0:
                return False, _error("tpu_profile_wrong_degrade_sigma",
                                     "PiDConditioning degrade_sigma must be 0 for the fixed PiD profile",
                                     nid, observed=inputs.get("degrade_sigma"), required=0.0)
        elif class_type == "BasicScheduler":
            if inputs.get("scheduler") != PID_SCHEDULER:
                return False, _error("tpu_profile_wrong_scheduler",
                                     "scheduler value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("scheduler"), PROFILE_PID, PID_SCHEDULER),
                                     nid, observed=inputs.get("scheduler"), required=PID_SCHEDULER)
            if inputs.get("steps") != PID_STEPS:
                return False, _error("tpu_profile_wrong_steps",
                                     "steps value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("steps"), PROFILE_PID, PID_STEPS),
                                     nid, observed=inputs.get("steps"), required=PID_STEPS)
            if inputs.get("denoise") != PID_DENOISE:
                return False, _error("tpu_profile_wrong_denoise",
                                     "denoise value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("denoise"), PROFILE_PID, PID_DENOISE),
                                     nid, observed=inputs.get("denoise"), required=PID_DENOISE)
        elif class_type == "KSamplerSelect":
            if inputs.get("sampler_name") != PID_SAMPLER_NAME:
                return False, _error("tpu_profile_wrong_sampler_name",
                                     "sampler_name value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("sampler_name"), PROFILE_PID, PID_SAMPLER_NAME),
                                     nid, observed=inputs.get("sampler_name"), required=PID_SAMPLER_NAME)
        elif class_type == "SamplerCustom":
            if inputs.get("cfg") != PID_CFG:
                return False, _error("tpu_profile_wrong_cfg",
                                     "cfg value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("cfg"), PROFILE_PID, PID_CFG),
                                     nid, observed=inputs.get("cfg"), required=PID_CFG)
        elif class_type == "ImageScale":
            w = inputs.get("width")
            h = inputs.get("height")
            # only allow fixed integer geometry, not graph-wired expressions
            if isinstance(w, list) or isinstance(h, list):
                return False, _error("tpu_profile_dynamic_geometry",
                                     "ImageScale with wired width/height is not supported on TPU for the fixed {} profile".format(PROFILE_PID),
                                     nid, observed=str(w if isinstance(w, list) else h), required=f"{PID_INPUT_WIDTH}x{PID_INPUT_HEIGHT}")
        elif class_type == "SaveImage":
            has_save = True
            if inputs.get("filename_prefix") != PID_SAVE_PREFIX:
                return False, _error("tpu_profile_wrong_save_prefix",
                                     "SaveImage prefix '{}' does not match the profile prefix '{}'".format(
                                         inputs.get("filename_prefix"), PID_SAVE_PREFIX),
                                     nid, observed=inputs.get("filename_prefix"), required=PID_SAVE_PREFIX)

    if not has_unet:
        return False, _error("tpu_profile_missing_loader", "the PiD profile requires a UNETLoader for pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors", None, observed="UNETLoader", required="UNETLoader")
    if not has_clip:
        return False, _error("tpu_profile_missing_loader", "the PiD profile requires a CLIPLoader for gemma_2_2b_it_elm_bf16.safetensors", None, observed="CLIPLoader", required="CLIPLoader")
    if not has_flux_vae:
        return False, _error("tpu_profile_missing_loader", "the PiD profile requires a VAELoader for flux1_vae.safetensors", None, observed="VAELoader", required="flux1_vae.safetensors")
    if not has_pixel_vae:
        return False, _error("tpu_profile_missing_loader", "the PiD profile requires a VAELoader for pixel_space", None, observed="VAELoader", required="pixel_space")
    if not has_empty_chroma:
        return False, _error("tpu_profile_missing_loader", "the PiD profile requires an EmptyChromaRadianceLatentImage", None, observed="EmptyChromaRadianceLatentImage", required="EmptyChromaRadianceLatentImage")
    if not has_pid_conditioning:
        return False, _error("tpu_profile_missing_loader", "the PiD profile requires a PiDConditioning node", None, observed="PiDConditioning", required="PiDConditioning")
    if not has_save:
        return False, _error("tpu_profile_missing_save", "the acceptance profile requires a SaveImage output node", None)
    return True, None


def _validate_krea_pid_prompt(nodes: Dict) -> tuple:
    """Validate the fused Krea2 -> PiD pipeline.

    Requires both Krea2 and PiD loaders, fixed geometry 1024x576 -> 4096x2304,
    Krea sampler (er_sde/simple 8) + PiD sampler (lcm/simple 4) and no host
    dynamic geometry nodes. VAE chain is qwen_image_vae (decode) -> flux1_vae
    (encode) -> pixel_space (final decode). Single queue item without host PIL.
    """
    has_krea_unet = False
    has_pid_unet = False
    has_krea_clip = False
    has_pid_clip = False
    has_qwen_vae = False
    has_flux_vae = False
    has_pixel_vae = False
    has_empty_latent = False
    has_empty_chroma = False
    has_pid_conditioning = False
    has_krea_sampler = False
    has_pid_sampler_select = False
    has_pid_scheduler = False
    has_pid_sampler_custom = False
    has_save = False
    krea_w = krea_h = None
    pid_w = pid_h = None

    for nid, node in nodes.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type in _UNSUPPORTED_LOADERS:
            return False, _error("tpu_profile_unsupported_loader",
                                 "{} is not supported on TPU: use the native UNETLoader/CLIPLoader/VAELoader "
                                 "nodes; TPU device placement is process-wide".format(class_type), nid,
                                 observed=class_type, required="native loader")
        if class_type in _UNSUPPORTED_MUTATIONS:
            return False, _error("tpu_profile_unsupported_mutation",
                                 "{} is not supported on TPU: LoRA, model patches, ControlNet, and "
                                 "reference-image conditioning are outside the profile".format(class_type),
                                 nid, observed=class_type, required="no mutations")
        if class_type in _PID_UNSUPPORTED_DYNAMIC:
            return False, _error("tpu_profile_dynamic_geometry",
                                 "{} is not supported on TPU for the fixed {} profile: the canonical fused graph "
                                 "hardcodes {}x{} -> {}x{}".format(class_type, PROFILE_KREA_PID, KREA_PID_KREA_WIDTH, KREA_PID_KREA_HEIGHT, KREA_PID_PID_WIDTH, KREA_PID_PID_HEIGHT),
                                 nid, observed=class_type, required="native fixed geometry")
        for key, value in inputs.items():
            if isinstance(value, str) and ("cuda" in value.lower() or "xpu" in value.lower()):
                return False, _error("tpu_profile_cuda_device",
                                     "{} on node {} requests device '{}'; TPU placement is process-wide and "
                                     "device widgets are not supported".format(key, nid, value),
                                     nid, observed=value, required="no explicit device")

        if class_type == "UNETLoader":
            unet_name = inputs.get("unet_name")
            if unet_name == "krea2_turbo_bf16.safetensors":
                has_krea_unet = True
            elif unet_name == "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors":
                has_pid_unet = True
            else:
                return False, _error("tpu_profile_artifact",
                                     "diffusion model '{}' is not in the approved manifest; expected "
                                     "krea2_turbo_bf16.safetensors or pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors".format(unet_name),
                                     nid, observed=unet_name, required="krea2_turbo_bf16.safetensors|pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors")
        elif class_type == "CLIPLoader":
            clip_name = inputs.get("clip_name")
            clip_type = inputs.get("type")
            if clip_name == "qwen3vl_4b_bf16.safetensors":
                has_krea_clip = True
                if clip_type != "krea2":
                    return False, _error("tpu_profile_invalid_clip_type",
                                         "CLIPLoader type must be 'krea2' for qwen3vl in the fused profile",
                                         nid, observed=clip_type, required="krea2")
            elif clip_name == "gemma_2_2b_it_elm_bf16.safetensors":
                has_pid_clip = True
                if clip_type != "pixeldit":
                    return False, _error("tpu_profile_invalid_clip_type",
                                         "CLIPLoader type must be 'pixeldit' for gemma in the fused profile",
                                         nid, observed=clip_type, required="pixeldit")
            else:
                return False, _error("tpu_profile_artifact",
                                     "text encoder '{}' is not in the approved manifest; expected "
                                     "qwen3vl_4b_bf16.safetensors or gemma_2_2b_it_elm_bf16.safetensors".format(clip_name),
                                     nid, observed=clip_name, required="qwen3vl_4b_bf16.safetensors|gemma_2_2b_it_elm_bf16.safetensors")
        elif class_type == "VAELoader":
            vae_name = inputs.get("vae_name")
            if vae_name == "qwen_image_vae.safetensors":
                has_qwen_vae = True
            elif vae_name in ("flux1_vae.safetensors", "flux1-vae.safetensors"):
                has_flux_vae = True
            elif vae_name == "pixel_space":
                has_pixel_vae = True
            else:
                return False, _error("tpu_profile_artifact",
                                     "VAE '{}' is not in the approved manifest; expected qwen_image_vae.safetensors, flux1_vae.safetensors or pixel_space".format(vae_name),
                                     nid, observed=vae_name, required="qwen_image_vae.safetensors|flux1_vae.safetensors|pixel_space")
        elif class_type == "EmptyLatentImage":
            has_empty_latent = True
            w = inputs.get("width")
            h = inputs.get("height")
            b = inputs.get("batch_size")
            krea_w, krea_h = w, h
            # Allow both 1024x576 and 512x288 (low-memory) buckets
            if not ((w == KREA_PID_KREA_WIDTH and h == KREA_PID_KREA_HEIGHT) or (w == 512 and h == 288)):
                return False, _error("tpu_profile_wrong_latent_shape",
                                     "EmptyLatentImage {}x{} does not satisfy fused profile 1024x576 or 512x288".format(
                                         w, h),
                                     nid, observed=f"{w}x{h}", required="1024x576|512x288")
            if b != PROFILE["batch_size"]:
                return False, _error("tpu_profile_wrong_batch_size",
                                     "EmptyLatentImage batch {} does not satisfy fused profile".format(b),
                                     nid, observed=b, required=PROFILE["batch_size"])
        elif class_type == "EmptyChromaRadianceLatentImage":
            has_empty_chroma = True
            w = inputs.get("width")
            h = inputs.get("height")
            b = inputs.get("batch_size", 1)
            pid_w, pid_h = w, h
            if not is_valid_pid_size(w, h, b):
                return False, _error("tpu_profile_wrong_latent_shape",
                                     "EmptyChromaRadianceLatentImage {}x{} does not satisfy PiD profile 4096x2304 or 2048x1152".format(
                                         w, h),
                                     nid, observed=f"{w}x{h}", required="4096x2304|2048x1152")
        elif class_type == "PiDConditioning":
            has_pid_conditioning = True
            if inputs.get("latent_format") != "flux":
                return False, _error("tpu_profile_invalid_latent_format",
                                     "PiDConditioning latent_format must be 'flux' for the fused profile",
                                     nid, observed=inputs.get("latent_format"), required="flux")
            try:
                sigma = float(inputs.get("degrade_sigma", 0))
            except Exception:
                sigma = None
            if sigma != 0.0:
                return False, _error("tpu_profile_wrong_degrade_sigma",
                                     "PiDConditioning degrade_sigma must be 0 for the fixed fused profile",
                                     nid, observed=inputs.get("degrade_sigma"), required=0.0)
        elif class_type == "KSampler":
            has_krea_sampler = True
            for field, required in [("steps", PROFILE["steps"]), ("cfg", PROFILE["cfg"]), ("sampler_name", PROFILE["sampler_name"]), ("scheduler", PROFILE["scheduler"]), ("denoise", PROFILE["denoise"])]:
                observed = inputs.get(field)
                if observed != required:
                    return False, _error("tpu_profile_wrong_{}".format(field),
                                         "{} value {} does not match the fused Krea profile (required: {})".format(
                                             field, observed, PROFILE_KREA_PID, required),
                                         nid, observed=observed, required=required)
        elif class_type == "BasicScheduler":
            has_pid_scheduler = True
            if inputs.get("scheduler") != PID_SCHEDULER:
                return False, _error("tpu_profile_wrong_scheduler",
                                     "scheduler value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("scheduler"), PROFILE_KREA_PID, PID_SCHEDULER),
                                     nid, observed=inputs.get("scheduler"), required=PID_SCHEDULER)
            if inputs.get("steps") != PID_STEPS:
                return False, _error("tpu_profile_wrong_steps",
                                     "steps value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("steps"), PROFILE_KREA_PID, PID_STEPS),
                                     nid, observed=inputs.get("steps"), required=PID_STEPS)
            if inputs.get("denoise") != PID_DENOISE:
                return False, _error("tpu_profile_wrong_denoise",
                                     "denoise value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("denoise"), PROFILE_KREA_PID, PID_DENOISE),
                                     nid, observed=inputs.get("denoise"), required=PID_DENOISE)
        elif class_type == "KSamplerSelect":
            has_pid_sampler_select = True
            if inputs.get("sampler_name") != PID_SAMPLER_NAME:
                return False, _error("tpu_profile_wrong_sampler_name",
                                     "sampler_name value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("sampler_name"), PROFILE_KREA_PID, PID_SAMPLER_NAME),
                                     nid, observed=inputs.get("sampler_name"), required=PID_SAMPLER_NAME)
        elif class_type == "SamplerCustom":
            has_pid_sampler_custom = True
            if inputs.get("cfg") != PID_CFG:
                return False, _error("tpu_profile_wrong_cfg",
                                     "cfg value {} does not match the fixed {} profile (required: {})".format(
                                         inputs.get("cfg"), PROFILE_KREA_PID, PID_CFG),
                                     nid, observed=inputs.get("cfg"), required=PID_CFG)
        elif class_type == "ImageScale":
            w = inputs.get("width")
            h = inputs.get("height")
            if isinstance(w, list) or isinstance(h, list):
                return False, _error("tpu_profile_dynamic_geometry",
                                     "ImageScale with wired width/height is not supported on TPU for the fixed {} profile".format(PROFILE_KREA_PID),
                                     nid, observed=str(w if isinstance(w, list) else h), required=f"{PID_INPUT_WIDTH}x{PID_INPUT_HEIGHT}")
        elif class_type == "SaveImage":
            has_save = True
            prefix = inputs.get("filename_prefix")
            # Allow any krea2-pid* prefix (covers krea2-pid, krea2-pid-2048 etc) and PiD variants
            allowed_fused = prefix in (KREA_PID_SAVE_PREFIX, KREA_PID_SAVE_PREFIX_ALT, KREA_PID_SAVE_PREFIX_ALT.lower(), "krea2_upscaler", "krea2-upscaler") or (isinstance(prefix, str) and prefix.startswith("krea2-pid"))
            allowed_legacy = prefix in (PROFILE["save_prefix"], PID_SAVE_PREFIX, KREA_PID_SAVE_PREFIX, KREA_PID_SAVE_PREFIX_ALT)
            if not (allowed_fused or allowed_legacy):
                return False, _error("tpu_profile_wrong_save_prefix",
                                     "SaveImage prefix '{}' does not match the fused profile prefix '{}' or '{}'".format(
                                         prefix, KREA_PID_SAVE_PREFIX, KREA_PID_SAVE_PREFIX_ALT),
                                     nid, observed=prefix, required=f"{KREA_PID_SAVE_PREFIX}|{KREA_PID_SAVE_PREFIX_ALT}|krea2-pid*")

    if not has_krea_unet:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a UNETLoader for krea2_turbo_bf16.safetensors", None, observed="UNETLoader(krea2)", required="krea2_turbo_bf16.safetensors")
    if not has_pid_unet:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a UNETLoader for pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors", None, observed="UNETLoader(pid)", required="pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors")
    if not has_krea_clip:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a CLIPLoader for qwen3vl_4b_bf16.safetensors (krea2)", None, observed="CLIPLoader(krea2)", required="qwen3vl_4b_bf16.safetensors")
    if not has_pid_clip:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a CLIPLoader for gemma_2_2b_it_elm_bf16.safetensors (pixeldit)", None, observed="CLIPLoader(pixeldit)", required="gemma_2_2b_it_elm_bf16.safetensors")
    if not has_qwen_vae:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a VAELoader for qwen_image_vae.safetensors", None, observed="VAELoader", required="qwen_image_vae.safetensors")
    if not has_flux_vae:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a VAELoader for flux1_vae.safetensors", None, observed="VAELoader", required="flux1_vae.safetensors")
    if not has_pixel_vae:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a VAELoader for pixel_space", None, observed="VAELoader", required="pixel_space")
    if not has_empty_latent:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires an EmptyLatentImage 1024x576", None, observed="EmptyLatentImage", required="EmptyLatentImage")
    if not has_empty_chroma:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires an EmptyChromaRadianceLatentImage 4096x2304", None, observed="EmptyChromaRadianceLatentImage", required="EmptyChromaRadianceLatentImage")
    if not has_pid_conditioning:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a PiDConditioning node", None, observed="PiDConditioning", required="PiDConditioning")
    if not has_krea_sampler:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a KSampler for Krea2 (er_sde simple 8)", None, observed="KSampler", required="KSampler")
    if not has_pid_scheduler:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a BasicScheduler for PiD (simple 4)", None, observed="BasicScheduler", required="BasicScheduler")
    if not has_pid_sampler_select:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a KSamplerSelect for PiD (lcm)", None, observed="KSamplerSelect", required="KSamplerSelect")
    if not has_pid_sampler_custom:
        return False, _error("tpu_profile_missing_loader", "the fused profile requires a SamplerCustom for PiD (cfg1)", None, observed="SamplerCustom", required="SamplerCustom")
    if not has_save:
        return False, _error("tpu_profile_missing_save", "the fused profile requires a SaveImage output node", None)
    # cross-check upscale factor and bucket
    if krea_w is not None and pid_w is not None:
        if not is_valid_krea_pid_size(krea_w, krea_h, pid_w, pid_h):
            return False, _error("tpu_profile_wrong_latent_shape",
                                 "fused sizes mismatch: Krea {}x{} *{} != PiD {}x{} ".format(krea_w, krea_h, PID_UPSCALE_FACTOR, pid_w, pid_h),
                                 None, observed=f"{krea_w}x{krea_h}->{pid_w}x{pid_h}", required="1024x576->4096x2304|512x288->2048x1152")
    return True, None


def validate_prompt(prompt: Dict, outputs_to_execute: Optional[List[str]]) -> tuple:
    """Validate a submitted prompt against the fixed TPU profile.

    Runs on the prompt graph after node replacement and normal validation, and
    only over nodes upstream of the requested outputs. Returns
    (True, None) or (False, error_dict) with a ``tpu_profile_`` error code.
    """
    if not prompt:
        return False, _error("tpu_profile_no_prompt", "TPU profile requires a prompt graph", None)

    if outputs_to_execute:
        outputs = set(str(nid) for nid in outputs_to_execute)
    else:
        consumed = set()
        for node in prompt.values():
            for value in node.get("inputs", {}).values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
                    consumed.add(str(value[0]))
        outputs = set(prompt.keys()) - consumed

    nodes = {nid: prompt[nid] for nid in _upstream_nodes(prompt, outputs) if nid in prompt}
    if not nodes:
        return False, _error("tpu_profile_no_output_nodes", "TPU profile requires at least one output node", None)

    active = _active_profile()
    is_krea_pid = active in (PROFILE_KREA_PID, PROFILE_KREA_PID_ALIAS, PROFILE_KREA_PID_ALIAS2)
    is_pid = active in (PROFILE_PID, PROFILE_PID_ALIAS)

    # Auto-detect when profile is not yet configured (e.g. tests that do not
    # set args.tpu_profile): fused graph has both Krea UNET + PiD marker,
    # pure PiD has only PiD marker.
    if not is_pid and not is_krea_pid:
        has_pid_marker = any(n.get("class_type") in ("PiDConditioning", "EmptyChromaRadianceLatentImage") for n in nodes.values())
        has_krea_unet = any(n.get("class_type") == "UNETLoader" and n.get("inputs", {}).get("unet_name") == "krea2_turbo_bf16.safetensors" for n in nodes.values())
        has_pid_unet = any(n.get("class_type") == "UNETLoader" and n.get("inputs", {}).get("unet_name") == "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors" for n in nodes.values())
        # fused needs both unets + pid marker
        if has_pid_marker and has_krea_unet and has_pid_unet:
            is_krea_pid = True
        elif has_pid_marker:
            is_pid = True

    if is_krea_pid:
        return _validate_krea_pid_prompt(nodes)
    if is_pid:
        return _validate_pid_prompt(nodes)
    return _validate_krea_prompt(nodes)


class ReadinessTracker:
    """Warm-up/readiness state exposed by the status endpoint (spec section 14)."""

    states = ("initializing", "loading", "compiling", "ready", "failed")

    def __init__(self, profile: str = PROFILE_NAME):
        self.profile = profile
        self.state = "initializing"
        self.mesh = ""
        self.cache_dir = ""
        self.artifact_hashes = []
        self.last_error = ""
        self.fields = {}
        self.timestamps = {s: None for s in self.states}
        self.timestamps["initializing"] = time.monotonic()

    def transition(self, state: str):
        if state not in self.states:
            raise ValueError("unknown readiness state: {}".format(state))
        self.state = state
        self.timestamps[state] = time.monotonic()
        logging.info("TPU readiness state: %s", state)

    def fail(self, error: str):
        self.last_error = error
        self.transition("failed")

    def snapshot(self) -> Dict[str, object]:
        return {
            "state": self.state,
            "profile": self.profile,
            "mesh": self.mesh,
            "cache_dir": self.cache_dir,
            "artifact_hashes": list(self.artifact_hashes),
            "last_error": self.last_error,
            "fields": dict(self.fields),
            "warmup_timestamps": {k: v for k, v in self.timestamps.items() if v is not None},
        }


# Process-wide readiness state; main.py drives it, server.py reads it for the
# status endpoint and the pre-queue gate.
readiness = ReadinessTracker()
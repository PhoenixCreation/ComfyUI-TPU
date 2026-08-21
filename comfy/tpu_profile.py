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
SUPPORTED_PROFILES = [PROFILE_NAME, PROFILE_NAME_DYNAMIC]

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

# Artifact manifest (deployment/model_manifest.json), relative to the repo root.
_ARTIFACT_DIR_BY_NAME = {
    "krea2_turbo_bf16.safetensors": "diffusion_models",
    "qwen3vl_4b_bf16.safetensors": "text_encoders",
    "qwen_image_vae.safetensors": "vae",
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

    loaders = {"UNETLoader": False, "CLIPLoader": False, "VAELoader": False}
    # KSampler and other shape-affecting fields remain fixed; EmptyLatentImage
    # is dynamic in Phase 2 — any valid Krea2 size compiles on demand.
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
            if inputs.get("unet_name") not in _ARTIFACT_DIR_BY_NAME:
                return False, _error("tpu_profile_artifact",
                                     "diffusion model '{}' is not in the approved manifest; expected "
                                     "krea2_turbo_bf16.safetensors".format(inputs.get("unet_name")),
                                     nid, observed=inputs.get("unet_name"), required="krea2_turbo_bf16.safetensors")
        elif class_type == "CLIPLoader":
            loaders["CLIPLoader"] = True
            if inputs.get("clip_name") not in _ARTIFACT_DIR_BY_NAME:
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
            if inputs.get("vae_name") not in _ARTIFACT_DIR_BY_NAME:
                return False, _error("tpu_profile_artifact",
                                     "VAE '{}' is not in the approved manifest; expected "
                                     "qwen_image_vae.safetensors".format(inputs.get("vae_name")),
                                     nid, observed=inputs.get("vae_name"), required="qwen_image_vae.safetensors")
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
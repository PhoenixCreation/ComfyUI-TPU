"""Named Krea2 TPU sharding policy.

Spec section 9.2: one named policy, covered by tests and emitted in
diagnostic output, validated against the real parameter names and shapes of
the three Phase 1 artifacts. Partitioning follows the dominant matmul
dimension; explicit exceptions are replicated with justification rather than
left accidentally unannotated.

Policy version is part of the persistent-cache fingerprint (spec section 12).
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

MESH_AXIS = "model"
POLICY_VERSION = "krea2-pid-tpu-v2"

# Partition dimension: "rows" = output rows of the weight matrix (x @ W^T),
# "cols" = input/contraction columns of the weight matrix.
_ROW_PARTITION = "rows"
_COL_PARTITION = "cols"
_REPLICATE = "replicate"

# (substring, action) rules evaluated in order. Shapes that reach the fallback
# use divisibility on the dominant dimension; anything not divisible by the
# eight-device mesh is replicated explicitly.
#
# Krea2 DiT exceptions:
# - attn wk/wv (1536 = 12 kv-heads x 128): 12 kv-heads do not divide over the
#   8-device mesh; GQA head replication keeps q/k/v shapes valid. Replicated
#   explicitly (justified deviation, spec section 9.2).
# - txtfusion attention (2560 = 20 heads x 128): 20 heads do not divide over
#   8 devices; the whole txtfusion attention stack is small (12 x 33M) and
#   replicated. txtfusion MLPs (6912 dims, divisible by 8) stay partitioned.
# - last.modulation.lin / txtfusion.projector: tiny F32 parameters, no matmul
#   benefit; replicated explicitly.
_KREA2_RULES = [
    (".attn.wk.weight", _REPLICATE),
    (".attn.wv.weight", _REPLICATE),
    ("txtfusion.", _REPLICATE),
    ("last.modulation.lin", _REPLICATE),
    (".attn.gate.weight", _ROW_PARTITION),
    (".attn.wq.weight", _ROW_PARTITION),
    (".attn.wo.weight", _COL_PARTITION),
    (".mlp.gate.weight", _ROW_PARTITION),
    (".mlp.up.weight", _ROW_PARTITION),
    (".mlp.down.weight", _COL_PARTITION),
    ("first.weight", _ROW_PARTITION),
    ("tmlp.0.weight", _ROW_PARTITION),
    ("tmlp.2.weight", _ROW_PARTITION),
    ("tproj.1.weight", _ROW_PARTITION),
    ("txtmlp.1.weight", _ROW_PARTITION),
    ("last.linear.weight", _COL_PARTITION),
]

# Qwen3-VL-4B exceptions:
# - embed_tokens (151936 x 2560): vocabulary lookup; a vocab-sharded embedding
#   needs cross-device gather on every lookup in this profile and its output
#   contract is feed-forward only. Replicated explicitly.
# - model.visual.* (vision tower, deepstack mergers, pos_embed): never
#   executed by the text-only Krea2 conditioning path; partitioning adds
#   collectives with zero execution benefit. Replicated explicitly.
_QWEN3VL_RULES = [
    ("embed_tokens.weight", _REPLICATE),
    ("model.visual.", _REPLICATE),
    (".self_attn.q_proj.weight", _ROW_PARTITION),
    (".self_attn.k_proj.weight", _ROW_PARTITION),
    (".self_attn.v_proj.weight", _ROW_PARTITION),
    (".self_attn.o_proj.weight", _COL_PARTITION),
    (".mlp.gate_proj.weight", _ROW_PARTITION),
    (".mlp.up_proj.weight", _ROW_PARTITION),
    (".mlp.down_proj.weight", _COL_PARTITION),
]

# Qwen Image VAE: all weights are small 3D convolution kernels or norm/bias
# vectors; replicated explicitly (spec section 9.2: small convolution kernels
# are replicated).
_VAE_RULES = []

# PiD diffusion (PixDiT_T2I + PiT + LQ branch). All large matmuls are
# divisible by the 8-device mesh (verified 2026-08-22: 4608/9216/4096/1536 etc
# all %8==0). Tiny SigmaAwareGate content_proj (1,3072) and log_alpha scalars
# plus PiT's small norm/mlpfc are replicated explicitly.
_PID_RULES = [
    ("content_proj.weight", _REPLICATE),
    ("log_alpha", _REPLICATE),
    ("spiece_model", _REPLICATE),
    (".norm", _REPLICATE),
    (".mlp.fc", _REPLICATE),
    (".pixel_embedder.proj.weight", _REPLICATE),
    (".attn.qkv", _ROW_PARTITION),
    (".attn.qkv_x.weight", _ROW_PARTITION),
    (".attn.qkv_y.weight", _ROW_PARTITION),
    (".attn.qkv.weight", _ROW_PARTITION),
    (".attn.proj", _COL_PARTITION),
    (".attn.proj_x.weight", _COL_PARTITION),
    (".attn.proj_y.weight", _COL_PARTITION),
    (".compress_to_attn.weight", _ROW_PARTITION),
    (".expand_from_attn.weight", _COL_PARTITION),
    (".adaLN_modulation", _ROW_PARTITION),
    (".mlp_x.w", _ROW_PARTITION),
    (".mlp_y.w", _ROW_PARTITION),
    (".mlp_x.w2.weight", _COL_PARTITION),
    (".mlp_y.w2.weight", _COL_PARTITION),
    (".output_heads", _ROW_PARTITION),
    (".pit_head", _ROW_PARTITION),
    (".latent_proj", _REPLICATE),
    (".s_embedder", _ROW_PARTITION),
    (".y_embedder.proj.weight", _ROW_PARTITION),
    (".t_embedder.mlp", _ROW_PARTITION),
]

# Gemma2-2B (pixeldit): vocab lookup and SPM replicated, attention/MLP sharded.
# GQA k/v 1024 are divisible by 8 (8*128) so no Krea-style 1280 exception.
_GEMMA2B_RULES = [
    ("embed_tokens.weight", _REPLICATE),
    ("spiece_model", _REPLICATE),
    ("input_layernorm.weight", _REPLICATE),
    ("post_attention_layernorm.weight", _REPLICATE),
    ("post_feedforward_layernorm.weight", _REPLICATE),
    ("pre_feedforward_layernorm.weight", _REPLICATE),
    ("model.norm.weight", _REPLICATE),
    (".self_attn.q_proj.weight", _ROW_PARTITION),
    (".self_attn.k_proj.weight", _ROW_PARTITION),
    (".self_attn.v_proj.weight", _ROW_PARTITION),
    (".self_attn.o_proj.weight", _COL_PARTITION),
    (".mlp.gate_proj.weight", _ROW_PARTITION),
    (".mlp.up_proj.weight", _ROW_PARTITION),
    (".mlp.down_proj.weight", _COL_PARTITION),
]

# Flux VAE (for PiD LQ encode): F32 conv kernels — replicate (same as Qwen VAE).
_FLUX_VAE_RULES = []


def _fallback_spec(shape: Tuple[int, ...]) -> str:
    if len(shape) == 2:
        rows, cols = shape
        if rows % 8 == 0 and rows >= cols:
            return _ROW_PARTITION
        if cols % 8 == 0:
            return _COL_PARTITION
    return _REPLICATE


class ShardingPolicy:
    """Resolves a partition spec for a parameter name and global shape."""

    def __init__(self, artifact: str, rules: List[Tuple[str, str]]):
        self.artifact = artifact
        self.rules = rules

    def spec_for(self, name: str, shape: Tuple[int, ...]) -> str:
        for pattern, action in self.rules:
            if pattern in name:
                return action
        if name.endswith(".bias"):
            weight_name = name[:-len(".bias")] + ".weight"
            weight_spec = self.spec_for(weight_name, ())
            if weight_spec == _ROW_PARTITION and len(shape) == 1:
                return _ROW_PARTITION
        return _fallback_spec(shape)

    def mesh_spec(self, spec: str, ndim: int) -> Tuple[str, ...]:
        """Convert a policy action into a mark_sharding axis spec."""
        if spec == _REPLICATE:
            return tuple("replicated" for _ in range(ndim))
        if ndim == 1 and spec == _ROW_PARTITION:
            # 1-D bias co-partitioned with its row-partitioned weight.
            return (MESH_AXIS,)
        if ndim == 2:
            if spec == _ROW_PARTITION:
                return (MESH_AXIS, "replicated")
            return ("replicated", MESH_AXIS)
        return tuple("replicated" for _ in range(ndim))

    def partition_dim(self, spec: str, shape: Tuple[int, ...]) -> Optional[int]:
        """Dimension split when the policy action partitions; None if replicated."""
        if spec == _ROW_PARTITION and len(shape) == 1:
            # 1-D bias co-partitioned with its row-partitioned weight: the
            # single dim is split over the mesh, so it must be divisible too.
            return 0
        if spec == _ROW_PARTITION and len(shape) == 2:
            return 0
        if spec == _COL_PARTITION and len(shape) == 2:
            return 1
        return None


def policy_for_artifact(artifact: str) -> ShardingPolicy:
    if artifact == "krea2_turbo_bf16.safetensors":
        return ShardingPolicy(artifact, _KREA2_RULES)
    if artifact == "qwen3vl_4b_bf16.safetensors":
        return ShardingPolicy(artifact, _QWEN3VL_RULES)
    if artifact == "qwen_image_vae.safetensors":
        return ShardingPolicy(artifact, _VAE_RULES)
    if artifact == "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors":
        return ShardingPolicy(artifact, _PID_RULES)
    if artifact == "gemma_2_2b_it_elm_bf16.safetensors":
        return ShardingPolicy(artifact, _GEMMA2B_RULES)
    if artifact in ("flux1_vae.safetensors", "flux1-vae.safetensors"):
        return ShardingPolicy(artifact, _FLUX_VAE_RULES)
    if artifact == "pixel_space":
        # Virtual VAE (comfy/pixel_space_convert.py) – single param, no sharding benefit.
        return ShardingPolicy(artifact, [])
    raise ValueError("no sharding policy for artifact: {}".format(artifact))


def validate_policy(policy: ShardingPolicy, params: Dict[str, Tuple[int, ...]]) -> List[Dict]:
    """Check the policy against actual parameter names and shapes.

    ``params`` maps parameter name to global shape. Returns a list of issue
    records; an empty list means every parameter resolved to a valid spec
    (partition dims divisible by the eight-device mesh).
    """
    issues = []
    for name, shape in sorted(params.items()):
        spec = policy.spec_for(name, shape)
        dim = policy.partition_dim(spec, shape)
        if dim is not None and shape[dim] % 8 != 0:
            issues.append({
                "name": name,
                "shape": list(shape),
                "spec": spec,
                "issue": "partition dim {} (size {}) is not divisible by the 8-device mesh".format(dim, shape[dim]),
            })
    return issues


class ShardingReport:
    """Machine-readable sharding report (spec section 9.2)."""

    def __init__(self, policy: ShardingPolicy):
        self.policy = policy
        self.records: List[Dict] = []

    def add(self, name: str, shape: Tuple[int, ...], dtype: str):
        spec = self.policy.spec_for(name, shape)
        self.records.append({
            "name": name,
            "shape": list(shape),
            "dtype": dtype,
            "spec": spec,
            "status": "sharded" if self.policy.partition_dim(spec, shape) is not None else "replicated",
        })

    def to_json(self) -> str:
        payload = {
            "policy_version": POLICY_VERSION,
            "artifact": self.policy.artifact,
            "mesh_axis": MESH_AXIS,
            "parameters": self.records,
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def write_all(reports: Dict[str, "ShardingReport"], path: str):
        payload = {
            "policy_version": POLICY_VERSION,
            "mesh_axis": MESH_AXIS,
            "artifacts": {name: json.loads(r.to_json()) for name, r in reports.items()},
        }
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        sharded = sum(1 for r in reports.values() for rec in r.records if rec["status"] == "sharded")
        replicated = sum(1 for r in reports.values() for rec in r.records if rec["status"] == "replicated")
        logging.info("Sharding report written to %s (%d sharded, %d replicated parameters)",
                     path, sharded, replicated)
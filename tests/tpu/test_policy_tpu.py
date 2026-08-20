"""Sharding policy validated against the real parameter names and shapes of
all three Phase 1 artifacts (spec section 9.2)."""

import json
import os

from comfy import tpu_sharding

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _params(name):
    with open(os.path.join(FIXTURES, name)) as f:
        raw = json.load(f)
    return {k: tuple(v[0]) for k, v in raw.items()}


def test_policy_validates_all_three_artifacts():
    for artifact, fixture in [
        ("krea2_turbo_bf16.safetensors", "krea2_params.json"),
        ("qwen3vl_4b_bf16.safetensors", "qwen3vl_params.json"),
        ("qwen_image_vae.safetensors", "vae_params.json"),
    ]:
        policy = tpu_sharding.policy_for_artifact(artifact)
        issues = tpu_sharding.validate_policy(policy, _params(fixture))
        assert issues == [], "{}: {}".format(artifact, issues)


def test_partitioned_parameters_are_mesh_divisible():
    for artifact, fixture in [
        ("krea2_turbo_bf16.safetensors", "krea2_params.json"),
        ("qwen3vl_4b_bf16.safetensors", "qwen3vl_params.json"),
    ]:
        policy = tpu_sharding.policy_for_artifact(artifact)
        for name, shape in _params(fixture).items():
            spec = policy.spec_for(name, shape)
            dim = policy.partition_dim(spec, shape)
            if dim is not None:
                assert shape[dim] % 8 == 0, "{} {} not divisible".format(artifact, name)


def test_known_policy_decisions():
    policy = tpu_sharding.policy_for_artifact("krea2_turbo_bf16.safetensors")
    assert policy.spec_for("blocks.0.attn.wq.weight", (6144, 6144)) == "rows"
    assert policy.spec_for("blocks.0.attn.wo.weight", (6144, 6144)) == "cols"
    assert policy.spec_for("blocks.0.attn.wk.weight", (1536, 6144)) == "replicate"
    assert policy.spec_for("blocks.0.mlp.gate.weight", (20480, 6144)) == "rows"
    assert policy.spec_for("blocks.0.mlp.down.weight", (6144, 20480)) == "cols"
    assert policy.spec_for("blocks.3.attn.qknorm.qnorm.scale", (6144,)) == "replicate"


def test_known_qwen_decisions():
    policy = tpu_sharding.policy_for_artifact("qwen3vl_4b_bf16.safetensors")
    assert policy.spec_for("model.language_model.embed_tokens.weight", (151936, 2560)) == "replicate"
    assert policy.spec_for("model.language_model.layers.0.mlp.down_proj.weight", (2560, 9728)) == "cols"
    assert policy.spec_for("model.language_model.layers.0.input_layernorm.weight", (2560,)) == "replicate"
    assert policy.spec_for("model.language_model.layers.0.self_attn.o_proj.weight", (2560, 2560)) == "cols"


def test_vae_replicated():
    policy = tpu_sharding.policy_for_artifact("qwen_image_vae.safetensors")
    for name, shape in _params("vae_params.json").items():
        assert policy.spec_for(name, shape) == "replicate"


def test_mesh_spec_dialect():
    policy = tpu_sharding.policy_for_artifact("krea2_turbo_bf16.safetensors")
    assert policy.mesh_spec("rows", 2) == ("model", "replicated")
    assert policy.mesh_spec("cols", 2) == ("replicated", "model")
    assert policy.mesh_spec("rows", 1) == ("model",)
    assert policy.mesh_spec("replicate", 4) == ("replicated", "replicated", "replicated", "replicated")


def test_partition_dim_consistency():
    policy = tpu_sharding.policy_for_artifact("krea2_turbo_bf16.safetensors")
    assert policy.partition_dim("rows", (6144, 6144)) == 0
    assert policy.partition_dim("cols", (6144, 6144)) == 1
    assert policy.partition_dim("rows", (6144,)) == 0
    assert policy.partition_dim("replicate", (6144, 6144)) is None


def test_sharding_report_smoke(tmp_path):
    import torch
    from comfy.xla_backend import _runtime_mesh_spec

    report = tpu_sharding.ShardingReport(tpu_sharding.policy_for_artifact("krea2_turbo_bf16.safetensors"))
    report.add("blocks.0.attn.wq.weight", (6144, 6144), "torch.bfloat16")
    report.add("blocks.0.attn.wk.weight", (1536, 6144), "torch.bfloat16")
    report.add("blocks.0.attn.qknorm.qnorm.scale", (6144,), "torch.float32")
    assert report.records[0]["status"] == "sharded"
    assert report.records[1]["status"] == "replicated"
    assert report.records[2]["status"] == "replicated"

    out = str(tmp_path / "sharding.json")
    tpu_sharding.ShardingReport.write_all({"krea2_turbo_bf16.safetensors": report}, out)
    with open(out) as f:
        payload = json.load(f)
    assert payload["policy_version"] == tpu_sharding.POLICY_VERSION
    assert payload["mesh_axis"] == tpu_sharding.MESH_AXIS
    params = payload["artifacts"]["krea2_turbo_bf16.safetensors"]["parameters"]
    assert len(params) == 3
    assert params[0]["status"] == "sharded"

    assert torch.device("cpu")  # silence unused-import lint on cpu-only boxes
    assert _runtime_mesh_spec(("model", "replicated")) == ("model", None)
    assert _runtime_mesh_spec(("replicated", "replicated")) == (None, None)
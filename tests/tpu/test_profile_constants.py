"""Fixed-profile constants (spec section 4) and their internal consistency."""

from comfy import tpu_profile


def test_profile_fixed_values():
    p = tpu_profile.PROFILE
    assert tpu_profile.PROFILE_NAME == "krea2-1920x1080"
    assert p["width"] == 1920
    assert p["height"] == 1080
    assert p["batch_size"] == 1
    assert p["steps"] == 8
    assert p["cfg"] == 1.0
    assert p["sampler_name"] == "er_sde"
    assert p["scheduler"] == "simple"
    assert p["save_prefix"] == "krea2_automatic"


def test_tokenizer_constants_consistent():
    t = tpu_profile
    assert t.TOKENIZER_FIXED_INPUT_LEN == 512
    assert t.TOKENIZER_PREFIX_TOKENS == 34
    assert t.TOKENIZER_CLOSING_TOKENS == 5
    assert t.TOKENIZER_PAD_TOKEN == 151643
    assert (t.TOKENIZER_PREFIX_TOKENS + t.TOKENIZER_CLOSING_TOKENS
            + t.TOKENIZER_CONTENT_BUDGET) == t.TOKENIZER_FIXED_INPUT_LEN


def test_conditioning_shapes():
    t = tpu_profile
    assert t.CONDITIONING_SEQ == t.TOKENIZER_FIXED_INPUT_LEN - t.TOKENIZER_PREFIX_TOKENS == 478
    assert t.CONDITIONING_FEATURES == 12 * 2560 == 30720
    assert t.LATENT_SHAPE == (1, 16, 135, 240)


def test_manifest_covers_all_three_artifacts():
    m = tpu_profile.load_manifest()
    artifacts = m.get("artifacts", {})
    assert {
        "krea2_turbo_bf16.safetensors",
        "qwen3vl_4b_bf16.safetensors",
        "qwen_image_vae.safetensors",
    }.issubset(set(artifacts))
    for name, info in artifacts.items():
        assert "sha256" in info
        assert "path" in info
        assert info["dtype"] in ("bf16", "f32")


def test_manifest_covers_pid_artifacts():
    m = tpu_profile.load_manifest()
    artifacts = m.get("artifacts", {})
    assert {
        "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors",
        "gemma_2_2b_it_elm_bf16.safetensors",
        "flux1_vae.safetensors",
    }.issubset(set(artifacts))
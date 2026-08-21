"""Pre-queue prompt validation against the fixed profile (spec section 13)."""

import copy


def _code(err):
    return err["error"]["type"]


def test_canonical_workflow_accepted(canonical_workflow):
    ok, err = __import__("comfy.tpu_profile", fromlist=["x"]).validate_prompt(canonical_workflow, ["14"])
    assert ok, err


def test_canonical_workflow_accepted_without_output_selection(canonical_workflow):
    ok, err = __import__("comfy.tpu_profile", fromlist=["x"]).validate_prompt(canonical_workflow, None)
    assert ok, err


def test_multigpu_loaders_rejected(canonical_workflow):
    from comfy import tpu_profile

    wf = copy.deepcopy(canonical_workflow)
    for nid in ("11", "12", "13"):
        wf[nid]["class_type"] += "MultiGPU"
    ok, err = tpu_profile.validate_prompt(wf, None)
    assert not ok
    assert _code(err) == "tpu_profile_unsupported_loader"


def test_lora_rejected(canonical_workflow):
    from comfy import tpu_profile

    wf = copy.deepcopy(canonical_workflow)
    wf["9"] = {"class_type": "LoraLoader", "inputs": {}}
    wf["2"]["inputs"]["model"] = ["9", 0]
    ok, err = tpu_profile.validate_prompt(wf, None)
    assert not ok
    assert _code(err) == "tpu_profile_unsupported_mutation"


def test_controlnet_rejected(canonical_workflow):
    from comfy import tpu_profile

    wf = copy.deepcopy(canonical_workflow)
    wf["9"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {}}
    wf["2"]["inputs"]["positive"] = ["9", 0]
    ok, err = tpu_profile.validate_prompt(wf, None)
    assert not ok
    assert _code(err) == "tpu_profile_unsupported_mutation"


def test_explicit_cuda_device_rejected(canonical_workflow):
    from comfy import tpu_profile

    wf = copy.deepcopy(canonical_workflow)
    wf["13"]["inputs"]["device"] = "cuda:1"
    ok, err = tpu_profile.validate_prompt(wf, None)
    assert not ok
    assert _code(err) == "tpu_profile_cuda_device"


def test_wrong_artifacts_rejected(canonical_workflow):
    from comfy import tpu_profile

    wf = copy.deepcopy(canonical_workflow)
    wf["12"]["inputs"]["unet_name"] = "other_model.safetensors"
    ok, err = tpu_profile.validate_prompt(wf, None)
    assert not ok
    assert _code(err) == "tpu_profile_artifact"
    assert err["error"]["extra_info"]["required"] == "krea2_turbo_bf16.safetensors"

    wf2 = copy.deepcopy(canonical_workflow)
    wf2["13"]["inputs"]["vae_name"] = "other_vae.safetensors"
    ok, err = tpu_profile.validate_prompt(wf2, None)
    assert not ok
    assert _code(err) == "tpu_profile_artifact"


def test_wrong_clip_type_rejected(canonical_workflow):
    from comfy import tpu_profile

    wf = copy.deepcopy(canonical_workflow)
    wf["11"]["inputs"]["type"] = "stable_diffusion"
    ok, err = tpu_profile.validate_prompt(wf, None)
    assert not ok
    assert _code(err) == "tpu_profile_invalid_clip_type"


def test_wrong_profile_fields_rejected(canonical_workflow):
    from comfy import tpu_profile

    # Dynamic sizes: 1280 and 720 are now valid (multiples of 8, area <2.1M).
    # Use truly invalid sizes for width/height checks.
    cases = [
        ({"10": ("width", 1281)}, "tpu_profile_wrong_width"),  # not multiple of 8
        ({"10": ("height", 721)}, "tpu_profile_wrong_height"),
        ({"10": ("batch_size", 2)}, "tpu_profile_wrong_batch_size"),
        ({"2": ("steps", 4)}, "tpu_profile_wrong_steps"),
        ({"2": ("cfg", 2.0)}, "tpu_profile_wrong_cfg"),
        ({"2": ("sampler_name", "euler")}, "tpu_profile_wrong_sampler_name"),
        ({"2": ("scheduler", "karras")}, "tpu_profile_wrong_scheduler"),
        ({"2": ("denoise", 0.5)}, "tpu_profile_wrong_denoise"),
        ({"14": ("filename_prefix", "other")}, "tpu_profile_wrong_save_prefix"),
    ]
    for edits, code in cases:
        wf = copy.deepcopy(canonical_workflow)
        for nid, (field, value) in edits.items():
            wf[nid]["inputs"][field] = value
        ok, err = tpu_profile.validate_prompt(wf, None)
        assert not ok, (code, err)
        assert _code(err) == code, (code, err)


def test_missing_loader_rejected(canonical_workflow):
    from comfy import tpu_profile

    wf = copy.deepcopy(canonical_workflow)
    del wf["11"]
    ok, err = tpu_profile.validate_prompt(wf, None)
    assert not ok
    assert _code(err) == "tpu_profile_missing_loader"


def test_missing_save_image_rejected(canonical_workflow):
    from comfy import tpu_profile

    wf = copy.deepcopy(canonical_workflow)
    del wf["14"]
    ok, err = tpu_profile.validate_prompt(wf, None)
    assert not ok
    assert _code(err) == "tpu_profile_missing_save"


def test_no_prompt_rejected():
    from comfy import tpu_profile

    ok, err = tpu_profile.validate_prompt({}, None)
    assert not ok
    assert _code(err) == "tpu_profile_no_prompt"
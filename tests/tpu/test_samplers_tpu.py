"""Host-side sigmas and the denoising section boundary (spec sections 11-12)."""

import inspect
import re

import torch

from comfy import samplers
from comfy import tpu_profile


def test_set_steps_keeps_sigmas_on_cpu_in_tpu_mode(tpu_mode, monkeypatch):
    k = samplers.KSampler.__new__(samplers.KSampler)
    k.device = torch.device("xla:0")
    k.model = None
    expected = torch.linspace(1.0, 0.0, 9)
    monkeypatch.setattr(k, "calculate_sigmas", lambda steps: expected.clone())
    k.set_steps(8)
    assert k.sigmas.device.type == "cpu"
    assert len(k.sigmas) == 9
    k.set_steps(8, denoise=0.5)
    assert k.sigmas.device.type == "cpu"
    assert len(k.sigmas) == 9


def test_set_steps_puts_sigmas_on_model_device_without_tpu(monkeypatch):
    k = samplers.KSampler.__new__(samplers.KSampler)
    k.device = torch.device("cpu")
    k.model = None
    monkeypatch.setattr(k, "calculate_sigmas", lambda steps: torch.linspace(1.0, 0.0, 9))
    k.set_steps(8)
    assert k.sigmas.device.type == "cpu"


def test_er_sde_loop_has_no_host_sync_calls():
    """Structural guard: the er_sde hot loop must not extract host scalars or
    sync device tensors gated on runtime values (spec section 11.2)."""
    import comfy.k_diffusion.sampling as ksampling

    src = inspect.getsource(ksampling.sample_er_sde)
    forbidden = [r"\.item\(", r"\.numpy\(", r"\.cpu\(\)", r"\.tolist\(", r"bool\(", r"int\([^)]*device"]
    for pattern in forbidden:
        assert not re.search(pattern, src), "er_sde contains forbidden host sync: {}".format(pattern)
    assert "sigmas[i" in src or "sigmas[i + 1]" in src


def test_sample_function_has_denoising_boundary():
    src = inspect.getsource(samplers)
    assert "stage_timer(\"denoising\")" in src
    assert "mark_step" in src
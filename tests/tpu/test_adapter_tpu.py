"""Adapter contracts (spec section 6): default no-ops, init ordering, and
idempotent/conflicting initialization."""

import sys

import torch

import comfy.accelerator
import comfy.model_management


def test_default_adapter_is_noop():
    comfy.accelerator.set_current_tracker(None)
    from comfy.accelerator import Accelerator

    a = Accelerator()
    assert a.kind == "default"
    assert not a.is_xla()
    a.initialize()
    assert a.memory_info() == {}
    assert a.metrics_report() == ""
    assert a.compile_counters() == {}
    t = torch.zeros(4)
    assert a.mark_activation_sharding(t, ("model", "replicated")) is t


def test_tpu_mode_reports_xla(tpu_mode):
    assert comfy.accelerator.is_xla()
    assert comfy.accelerator.get_accelerator().world_size == 8
    assert comfy.model_management.xla_enabled()
    assert comfy.model_management.get_torch_device().type == "xla"


def test_tpu_mode_bf16_dtypes(tpu_mode):
    assert comfy.model_management.unet_dtype() == torch.bfloat16
    assert comfy.model_management.text_encoder_dtype() == torch.bfloat16
    assert comfy.model_management.vae_dtype() == torch.bfloat16


def test_initialize_accelerator_uses_xla_backend(tpu_mode, monkeypatch):
    import comfy.accelerator as acc

    calls = []

    class FakeXla:
        kind = "xla"
        world_size = 8
        device = None

        def __init__(self, tpu_cache_dir=None):
            calls.append(tpu_cache_dir)

        def initialize(self):
            pass

        def is_xla(self):
            return True

    monkeypatch.setattr(acc, "_initialized", False)
    monkeypatch.setattr("comfy.xla_backend.XlaAccelerator", FakeXla)
    adapter = acc.initialize_accelerator()
    assert adapter.is_xla()
    assert calls == [acc.args.tpu_cache_dir]


def test_initialize_accelerator_idempotent_and_conflict(tpu_mode, monkeypatch):
    import comfy.accelerator as acc

    monkeypatch.setattr(acc, "_accelerator", acc.Accelerator())
    monkeypatch.setattr(acc, "_initialized", True)
    with __import__("pytest").raises(RuntimeError):
        acc.initialize_accelerator()


def test_torch_xla_not_imported_by_default():
    """XLA is hidden behind the adapter even in this TPU-configured test
    process: nothing imports torch_xla unless initialize() runs for real."""
    assert "torch_xla" not in sys.modules
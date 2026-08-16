"""TPU CLI contracts (spec section 5.1): flag conflicts, BF16 defaults, and
the required --tpu-cache-dir. Parsing happens once at import, so each case is
run in a throwaway subprocess with the flag set on sys.argv."""

import subprocess
import sys

import pytest

PY = sys.executable

TPL = """
import sys
sys.argv = {argv!r}
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args
"""

CONFLICTS = [
    ("--cpu",),
    ("--directml",),
    ("--cuda-device", "0"),
    ("--default-device", "cpu"),
    ("--oneapi-device-selector", "0"),
    ("--lowvram",),
    ("--novram",),
    ("--gpu-only",),
    ("--highvram",),
]

REJECTED = [
    ("--force-fp32",),
    ("--force-fp16",),
    ("--fp16-unet",),
    ("--fp8_e4m3fn-unet",),
    ("--fp16-vae",),
    ("--fp32-text-enc",),
    ("--disable-cuda-malloc",),
    ("--enable-triton-backend",),
    ("--use-split-cross-attention",),
    ("--use-sage-attention",),
    ("--use-flash-attention",),
    ("--use-ck-attention",),
    ("--enable-dynamic-vram",),
    ("--fast",),
]


def _parse_ok(flags, tpu=True):
    argv = ["comfy", "--tpu", "--tpu-cache-dir", "/tmp/x"] + list(flags) if tpu else ["comfy"] + list(flags)
    code = TPL.format(argv=argv)
    result = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    return result.returncode


@pytest.mark.parametrize("flags", CONFLICTS, ids=lambda f: f[0])
def test_tpu_conflicts_rejected(flags):
    assert _parse_ok(flags) != 0


@pytest.mark.parametrize("flags", REJECTED, ids=lambda f: f[0])
def test_tpu_rejected_precision_attention_flags(flags):
    assert _parse_ok(flags) != 0


@pytest.mark.parametrize("flags", REJECTED, ids=lambda f: f[0])
def test_flags_are_fine_without_tpu(flags):
    assert _parse_ok(flags, tpu=False) == 0


def test_tpu_requires_cache_dir():
    code = """
import sys
sys.argv = ['comfy', '--tpu']
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args
"""
    result = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "--tpu-cache-dir" in result.stderr


def test_bf16_tpu_flags_accepted_as_redundant():
    assert _parse_ok(["--bf16-unet", "--bf16-text-enc", "--bf16-vae"]) == 0


def test_bf16_relevant_defaults_when_tpu_enabled():
    code = """
import sys
sys.argv = ['comfy', '--tpu', '--tpu-cache-dir', '/tmp/x']
import comfy.options
comfy.options.enable_args_parsing()
from comfy.cli_args import args
assert args.tpu
import torch, comfy.accelerator
class Stub:
    kind='xla'; world_size=8; device=torch.device('xla:0')
    def is_xla(self): return True
    def mark_step(self): pass
    def wait_device_ops(self): pass
    def mark_activation_sharding(self, t, s): return t
    def apply_parameter_sharding(self, m, p): return {}
    def transfer_sharded(self, module, source, policy): return {}
    def memory_info(self): return {}
    def metrics_report(self): return ''
    def compile_counters(self): return {}
comfy.accelerator._accelerator = Stub()
import comfy.model_management
assert comfy.model_management.unet_dtype() == torch.bfloat16
assert comfy.model_management.text_encoder_dtype() == torch.bfloat16
assert comfy.model_management.vae_dtype() == torch.bfloat16
assert comfy.model_management.get_torch_device().type == 'xla'
assert comfy.model_management.xla_enabled()
print('OK')
"""
    result = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-1200:]
    assert "OK" in result.stdout


def test_profile_choices_limited():
    code = """
import sys
sys.argv = ['comfy', '--tpu', '--tpu-cache-dir', '/tmp/x', '--tpu-profile', 'bogus']
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args
"""
    result = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
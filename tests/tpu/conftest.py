"""TPU-mode unit tests: everything here runs WITHOUT a TPU or torch_xla.

XLA APIs are hidden behind the XlaStub fakes (spec section 17.1). TPU mode is
simulated by setting comfy.cli_args.args.tpu on the process state, exactly as
the production --tpu flag would, and swapping the process-wide accelerator
with the stub.

Import contract: comfy.model_management runs a module-level device probe at
import time. With args.tpu set (and the stub accelerator installed) that probe
takes the XLA memory branch, so this conftest installs both BEFORE any test
module imports comfy. Run this suite in isolation: ``pytest tests/tpu``.
"""

import json
import os

import pytest

from comfy.cli_args import args

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_tmp")


class XlaStub:
    """Fake XLA adapter with the same surface as comfy.xla_backend.XlaAccelerator."""

    kind = "xla"
    world_size = 8
    device = None

    def __init__(self):
        import torch
        self.device = torch.device("xla:0")
        self.mark_step_calls = 0

    def is_xla(self):
        return True

    def mark_step(self):
        self.mark_step_calls += 1

    def wait_device_ops(self):
        pass

    def apply_parameter_sharding(self, module, policy):
        return {"sharded": 0, "replicated": 0}

    def mark_activation_sharding(self, tensor, spec):
        return tensor

    def transfer_sharded(self, module, source, policy):
        return {"sharded": 0, "replicated": 0}

    def memory_info(self):
        return {}

    def metrics_report(self):
        return ""

    def compile_counters(self):
        return {"MarkStep": self.mark_step_calls}


_saved = (args.tpu, args.tpu_profile, args.tpu_cache_dir, args.tpu_warmup)
args.tpu = True
args.tpu_profile = "krea2-1920x1080"
args.tpu_cache_dir = CACHE_DIR
args.tpu_warmup = True

import comfy.accelerator  # noqa: E402  (must follow the args.tpu set above)

comfy.accelerator._accelerator = XlaStub()
comfy.accelerator._initialized = False
comfy.accelerator.set_current_tracker(None)


def pytest_sessionfinish(session, exitstatus):
    args.tpu, args.tpu_profile, args.tpu_cache_dir, args.tpu_warmup = _saved


@pytest.fixture
def tpu_mode(monkeypatch):
    """Fresh stub accelerator + tracker state per test."""
    stub = XlaStub()
    monkeypatch.setattr(comfy.accelerator, "_accelerator", stub)
    comfy.accelerator.set_current_tracker(None)
    yield stub
    comfy.accelerator.set_current_tracker(None)


@pytest.fixture
def canonical_workflow():
    """The checked-in canonical TPU workflow graph."""
    path = os.path.join(REPO_ROOT, "workflows", "Krea2-turbo-tpu.json")
    with open(path) as f:
        return json.load(f)
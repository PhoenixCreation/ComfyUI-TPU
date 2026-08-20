"""Backend-neutral accelerator access point.

Generic ComfyUI code talks to this module; only ``comfy.xla_backend`` imports
``torch_xla``. The default adapter is a process-wide no-op wrapper: existing
CUDA/CPU/ROCm/MPS/XPU/NPU/MLU/DirectML behavior is unchanged when TPU mode is
disabled. In TPU mode the XLA adapter owns the three files in ``comfy/xla_backend.py``.

The same module also carries the per-request stage tracker used by the TPU
instrumentation contract (spec section 15). It is backend-neutral so the
layout is identical in non-TPU runs.
"""

import logging
import time
from typing import Dict, Optional

import torch

from comfy.cli_args import args


class Accelerator:
    """Process-wide accelerator state. Default adapter: no-ops that leave the
    existing model-management behavior untouched."""

    kind = "default"
    world_size = 1
    device: Optional[torch.device] = None

    def initialize(self) -> None:
        pass

    def is_xla(self) -> bool:
        return False

    def mark_step(self) -> None:
        pass

    def wait_device_ops(self) -> None:
        pass

    def apply_parameter_sharding(self, module, policy) -> None:
        pass

    def mark_activation_sharding(self, tensor, spec):
        return tensor

    def transfer_sharded(self, module, source, policy):
        raise RuntimeError("transfer_sharded is an XLA-only operation; it is never reached outside TPU mode")

    def memory_info(self) -> Dict[str, object]:
        return {}

    def metrics_report(self) -> str:
        return ""

    def compile_counters(self) -> Dict[str, int]:
        return {}

    def shutdown(self) -> None:
        pass


_accelerator = Accelerator()
_initialized = False


def get_accelerator() -> Accelerator:
    return _accelerator


def is_xla() -> bool:
    return _accelerator.is_xla()


def initialize_accelerator() -> Accelerator:
    """Build the process-wide adapter. Idempotent: a second call with the same
    configuration returns the existing adapter; conflicting configuration
    fails."""
    global _accelerator
    global _initialized
    if not _initialized:
        if args.tpu:
            from comfy import xla_backend
            adapter = xla_backend.XlaAccelerator(tpu_cache_dir=args.tpu_cache_dir)
            adapter.initialize()
            _accelerator = adapter
        _initialized = True
    else:
        if args.tpu and not _accelerator.is_xla():
            raise RuntimeError("Conflicting accelerator configuration: this process already initialized the default accelerator without --tpu")
    return _accelerator


def mark_step() -> None:
    _accelerator.mark_step()


def wait_device_ops() -> None:
    _accelerator.wait_device_ops()


def memory_info() -> Dict[str, object]:
    return _accelerator.memory_info()


def metrics_report() -> str:
    return _accelerator.metrics_report()


def compile_counters() -> Dict[str, int]:
    return _accelerator.compile_counters()


class StageTracker:
    """Per-request timing record for the TPU instrumentation contract.

    Timestamps use monotonic time; the record is emitted once per request as a
    structured log line keyed by ``prompt_id`` (spec section 15). Stages are
    registered by the owning code paths: model load, tokenization, text
    encoder, denoising, VAE, host transfer, PNG encode, file write.
    """

    def __init__(self, prompt_id: str, profile: str = ""):
        self.prompt_id = prompt_id
        self.profile = profile
        self.stages: Dict[str, float] = {}
        self.fields: Dict[str, object] = {}
        self._active: Dict[str, float] = {}
        self.interval_start: Optional[float] = None
        self.interval_end: Optional[float] = None

    def begin_interval(self):
        if self.interval_start is None:
            self.interval_start = time.monotonic()

    def begin(self, stage: str):
        if stage not in self._active:
            self._active[stage] = time.monotonic()

    def end(self, stage: str):
        start = self._active.pop(stage, None)
        if start is not None:
            self.stages[stage] = self.stages.get(stage, 0.0) + (time.monotonic() - start)

    def record(self, key: str, value: object):
        self.fields[key] = value

    def finalize(self):
        if self.interval_start is not None and self.interval_end is None:
            self.interval_end = time.monotonic()

    def to_log_record(self, outcome: str):
        self.finalize()
        record = {
            "event": "tpu_request",
            "prompt_id": self.prompt_id,
            "profile": self.profile,
            "outcome": outcome,
            "durations_ms": {k: round(v * 1000.0, 2) for k, v in sorted(self.stages.items())},
        }
        if self.interval_start is not None and self.interval_end is not None:
            record["execution_interval_ms"] = round((self.interval_end - self.interval_start) * 1000.0, 2)
        record.update(self.fields)
        return record

    def emit(self, outcome: str):
        logging.info("tpu_request %s", self.to_log_record(outcome), extra={"structured": True})


_current_tracker: Optional[StageTracker] = None


def set_current_tracker(tracker: Optional[StageTracker]):
    global _current_tracker
    _current_tracker = tracker


def get_current_tracker() -> Optional[StageTracker]:
    return _current_tracker


def stage_timer(stage: str):
    """Context manager recording ``stage`` on the current tracker, if any."""
    class _StageTimer:
        def __enter__(self):
            tracker = get_current_tracker()
            if tracker is not None:
                tracker.begin(stage)

        def __exit__(self, *exc):
            tracker = get_current_tracker()
            if tracker is not None:
                tracker.end(stage)
            return False

    return _StageTimer()
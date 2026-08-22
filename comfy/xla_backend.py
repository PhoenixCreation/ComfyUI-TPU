"""PyTorch/XLA SPMD backend for TPU mode.

This is the only core ComfyUI module that imports ``torch_xla`` (spec section
6.1). ``torch_xla`` is imported lazily inside :meth:`XlaAccelerator.initialize`
so that non-TPU installs never load it and CI without the runtime stays green.

Environment is fixed before the import: PJRT TPU is selected via
``PJRT_DEVICE``, the metadata-server lookup is skipped (``TPU_SKIP_MDS_QUERY``,
see docs/changes.md problem 1), and the legacy multi-host variables
``TPU_PROCESS_ADDRESSES`` / ``XRT_TPU_CONFIG`` are removed (they push the
runtime onto a multi-host bootstrap path inside the container).
"""

import hashlib
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch

from comfy import tpu_profile, tpu_sharding
from comfy.cli_args import args

MESH_SHAPE = (8,)
MESH_AXIS_NAMES = ("model",)
DEVICE_COUNT = 8


def _runtime_mesh_spec(mesh_spec) -> tuple:
    """Translate policy specs to the torch_xla partition-spec dialect.

    Policies write ``"replicated"`` for readability; the runtime represents a
    replicated dimension as ``None`` (any other element is a mesh axis name).
    """
    return tuple(None if element == "replicated" else element for element in mesh_spec)


def _safe_unset(name: str):
    if name in os.environ:
        logging.warning("TPU environment: removing %s=%s (legacy multi-host/bootstrap variable)", name, os.environ.pop(name))


class XlaAccelerator:
    kind = "xla"
    world_size = DEVICE_COUNT

    def __init__(self, tpu_cache_dir: Optional[str] = None):
        self.tpu_cache_dir = tpu_cache_dir
        self.device = None
        self.mesh = None
        self._runtime = None
        self._spmd = None
        self._xla_model = None
        self._initialized = False
        self._sharding_reports: Dict[str, tpu_sharding.ShardingReport] = {}

    # ------------------------------------------------------------------ init

    def initialize(self) -> None:
        if self._initialized:
            return
        if self.tpu_cache_dir is None:
            raise RuntimeError("TPU mode requires a --tpu-cache-dir for the persistent compilation cache")

        self._set_environment()
        cache_path = self._cache_fingerprint_path()
        os.environ["XLA_PERSISTENT_CACHE_PATH"] = cache_path
        logging.info("XLA persistent compilation cache: %s", cache_path)

        self._import_runtime()
        self._validate_devices()
        self._enable_spmd()
        self._log_startup()

    def _set_environment(self):
        # docs/changes.md problems 1/4: without these the runtime blocks on the
        # (unresolvable) GCE metadata server or takes the multi-host bootstrap
        # path; the values are deployment constants for the v5e-8 slice.
        environment = {
            "PJRT_DEVICE": "TPU",
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_ACCELERATOR_TYPE": "v5litepod-8",
            "TPU_CHIPS_PER_HOST_BOUNDS": "2,4,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "localhost",
        }
        for key, value in environment.items():
            previous = os.environ.get(key)
            os.environ[key] = value
            if previous != value:
                logging.info("TPU environment: %s=%s", key, value)
        _safe_unset("TPU_PROCESS_ADDRESSES")
        _safe_unset("XRT_TPU_CONFIG")

    def _cache_fingerprint_path(self) -> str:
        """Fingerprint-separated executable cache (spec section 12).

        The fingerprint covers runtime versions, artifact hashes, profile
        constants, dtype, mesh, and the sharding policy version, so changing
        any one of them cannot silently reuse an incompatible executable.
        """
        # Dynamic Krea2 sizes share one persistent-cache directory; the cache
        # key includes program shape, so per-size executables are distinct
        # inside the directory. Keep the fixed profile's hash stable.
        if args.tpu_profile in (getattr(tpu_profile, "PROFILE_PID", None), getattr(tpu_profile, "PROFILE_PID_ALIAS", None)):
            latent_str = f"{tpu_profile.PID_OUTPUT_WIDTH}x{tpu_profile.PID_OUTPUT_HEIGHT}"
            tokens_str = f"pixeldit={tpu_profile.PIXELDIT_FIXED_LEN}"
        elif args.tpu_profile == getattr(tpu_profile, "PROFILE_NAME_DYNAMIC", None):
            latent_str = "dynamic"
            tokens_str = "%d:%d:%d" % (
                tpu_profile.TOKENIZER_FIXED_INPUT_LEN,
                tpu_profile.TOKENIZER_PREFIX_TOKENS,
                tpu_profile.TOKENIZER_CLOSING_TOKENS,
            )
        else:
            latent_str = str(list(tpu_profile.LATENT_SHAPE))
            tokens_str = "%d:%d:%d" % (
                tpu_profile.TOKENIZER_FIXED_INPUT_LEN,
                tpu_profile.TOKENIZER_PREFIX_TOKENS,
                tpu_profile.TOKENIZER_CLOSING_TOKENS,
            )
        parts = [
            "torch=" + torch.__version__,
            "profile=" + args.tpu_profile,
            "dtype=bf16",
            "mesh=%s" % (MESH_SHAPE,),
            "policy=" + tpu_sharding.POLICY_VERSION,
            "tokens=%s" % tokens_str,
            "latent=%s" % latent_str,
        ]
        manifest = tpu_profile.load_manifest()
        for name, info in sorted(manifest.get("artifacts", {}).items()):
            parts.append("{}={}".format(name, info.get("sha256", "")))
        try:
            import torch_xla
            parts.append("torch_xla=" + torch_xla.__version__)
        except ImportError:
            pass
        fingerprint = hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]
        return os.path.join(self.tpu_cache_dir, "executables", fingerprint)

    def _import_runtime(self):
        try:
            from torch_xla import runtime as xr
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.spmd as xs
        except ImportError as e:
            raise RuntimeError(
                "TPU mode requires a torch-xla install matching the pinned PyTorch release line "
                "(see deployment/requirements-tpu.txt and deployment/tpu_env.sh)"
            ) from e
        self._runtime = xr
        self._xla_model = xm
        self._spmd = xs

    def _validate_devices(self):
        try:
            count = self._runtime.global_runtime_device_count()
        except AttributeError:
            count = self._xla_model.xrt_world_size()
        if count != DEVICE_COUNT:
            raise RuntimeError(
                "TPU mode requires exactly {} addressable TPU devices, found {}. v5e-8 slices expose all "
                "eight chips through PJRT. If another process holds the TPU (e.g. a notebook kernel after "
                "any jax/torch_xla call), the chips stay locked and PJRT sees fewer devices or fails with "
                "'Device or resource busy' on /dev/vfio/*; free the chips and retry.".format(DEVICE_COUNT, count)
            )

    def _enable_spmd(self):
        try:
            self._runtime.use_spmd()
        except AttributeError as e:
            raise RuntimeError("installed torch-xla does not expose runtime.use_spmd(); upgrade to the pinned release") from e
        # SPMD must be enabled before the first logical XLA device is created.
        # Initializing xla:0 first makes torch-xla migrate already-created
        # tensors into the virtual SPMD device and can crash PJRT when the
        # first large sharded transfer is executed.
        self.device = self._xla_model.xla_device()
        self._initialized = True
        device_ids = list(range(DEVICE_COUNT))
        self.mesh = self._spmd.Mesh(device_ids, MESH_SHAPE, MESH_AXIS_NAMES)
        self.mesh_device_ids = device_ids
        self.mesh_axis_names = list(MESH_AXIS_NAMES)

    def _log_startup(self):
        logging.info(
            "TPU accelerator initialized: logical device=%s devices=%d mesh_shape=%s axis_names=%s cache=%s",
            self.device, DEVICE_COUNT, MESH_SHAPE, MESH_AXIS_NAMES,
            os.environ.get("XLA_PERSISTENT_CACHE_PATH", ""),
        )
        try:
            import torch_xla
            logging.info("TPU runtime versions: torch=%s torch_xla=%s", torch.__version__, torch_xla.__version__)
        except ImportError:
            pass

    # ------------------------------------------------------------- interface

    def is_xla(self) -> bool:
        return True

    def mark_step(self) -> None:
        if self._xla_model is not None:
            self._xla_model.mark_step()

    def wait_device_ops(self) -> None:
        if self._xla_model is not None:
            self._xla_model.wait_device_ops()

    def apply_parameter_sharding(self, module, policy=None) -> Dict[str, int]:
        if policy is None:
            policy = tpu_sharding.policy_for_artifact(module.__class__.__name__)
        report = self._sharding_reports.setdefault(policy.artifact, tpu_sharding.ShardingReport(policy))
        sharded = 0
        replicated = 0
        for name, param in module.named_parameters():
            if param.device.type != "xla":
                continue
            spec = policy.spec_for(name, tuple(param.shape))
            self._mark(param, spec, report, name, param.dtype, policy)
            if policy.partition_dim(spec, tuple(param.shape)) is not None:
                sharded += 1
            else:
                replicated += 1
        return {"sharded": sharded, "replicated": replicated}

    def mark_activation_sharding(self, tensor, spec):
        if self._spmd is None or tensor.device.type != "xla":
            return tensor
        try:
            self._spmd.mark_sharding(tensor, self.mesh, _runtime_mesh_spec(spec))
        except Exception as e:
            logging.debug("activation sharding skipped for %s: %s", tuple(tensor.shape), e)
        return tensor

    def load_sharded_state_dict(self, module, state_dict, policy) -> Dict[str, int]:
        """Populate module parameters on XLA with their final sharding.

        Each parameter becomes one annotated XLA tensor holding only its local
        shard; the unsharded XLA copy is never created (spec section 9.1). The
        host-side full tensor is the mmap'd artifact value and is released as
        soon as the copy completes.
        """
        report = self._sharding_reports.setdefault(policy.artifact, tpu_sharding.ShardingReport(policy))
        sharded = 0
        replicated = 0
        for name, param in module.named_parameters():
            if param.device.type == "xla":
                continue
            key = name
            if key not in state_dict:
                # Krea2/Qwen models use the diffusion_model/transformer scope;
                # materialization is called on the submodule whose names match.
                continue
            value = state_dict[key]
            if value.dtype != param.dtype:
                value = value.to(param.dtype)
            spec = policy.spec_for(name, tuple(param.shape))
            self._mark(param, spec, report, name, param.dtype, policy)
            param.data.copy_(value)
            del value
            if policy.partition_dim(spec, tuple(param.shape)) is not None:
                sharded += 1
            else:
                replicated += 1
        return {"sharded": sharded, "replicated": replicated}

    def transfer_sharded(self, module, source, policy) -> Dict[str, int]:
        """Move host-resident module parameters to XLA with final sharding.

        ``source`` carries the current (CPU) parameter values, usually the same
        module the patcher loaded via the standard CPU weight staging. Buffers
        are handled the same way. Names are the full dotted names of ``module``;
        policy patterns match on substrings (diffusion_model./model. prefixes
        are harmless). Host tensors are dropped per parameter so staging stays
        bounded.
        """
        report = self._sharding_reports.setdefault(policy.artifact, tpu_sharding.ShardingReport(policy))
        sharded = 0
        replicated = 0
        # Use PyTorch/XLA's native module transfer. Replacing CPU Parameter
        # storage with an XLA tensor via ``.data`` or ``swap_tensors`` is not a
        # supported cross-device operation and can crash PJRT during the first
        # replicated execution.
        module.to(self.device)
        for name, param in module.named_parameters():
            spec = policy.spec_for(name, tuple(param.shape))
            if policy.partition_dim(spec, tuple(param.shape)) is not None:
                sharded += 1
            else:
                replicated += 1
            self._spmd.mark_sharding(param, self.mesh, _runtime_mesh_spec(policy.mesh_spec(spec, len(param.shape))))
            report.add(name, tuple(param.shape), str(param.dtype))
        for name, buf in module.named_buffers():
            spec = policy.spec_for(name, tuple(buf.shape))
            self._spmd.mark_sharding(buf, self.mesh, _runtime_mesh_spec(policy.mesh_spec(spec, len(buf.shape))))
            report.add(name, tuple(buf.shape), str(buf.dtype))
        return {"sharded": sharded, "replicated": replicated}

    def _mark(self, param, spec, report, name, dtype, policy=None):
        param.data = torch.empty(param.shape, dtype=dtype, device=self.device)
        self._annotate(param, name, spec, policy, report)

    def _annotate(self, tensor, name, spec, policy, report):
        mesh_spec = policy.mesh_spec(spec, len(tensor.shape)) if policy is not None else self.policy_mesh_spec(spec, len(tensor.shape))
        self._spmd.mark_sharding(tensor, self.mesh, _runtime_mesh_spec(mesh_spec))
        report.add(name, tuple(tensor.shape), str(tensor.dtype))

    def policy_mesh_spec(self, spec, ndim):
        if spec == "replicate":
            return tuple("replicated" for _ in range(ndim))
        if ndim == 1:
            return (MESH_AXIS_NAMES[0],)
        if spec == "rows":
            return (MESH_AXIS_NAMES[0], "replicated")
        return ("replicated", MESH_AXIS_NAMES[0])

    def write_sharding_report(self) -> str:
        path = os.path.join(self.tpu_cache_dir, "sharding_report.json")
        tpu_sharding.ShardingReport.write_all(self._sharding_reports, path)
        return path

    # ------------------------------------------------------------------ info

    def memory_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {"backend": "xla", "device_count": DEVICE_COUNT}
        try:
            mem = self._xla_model.get_memory_info(self.device)
            if mem is not None:
                info["xla_current_bytes"] = mem.get("current", 0)
                info["xla_peak_bytes"] = mem.get("peak", 0)
        except (AttributeError, NotImplementedError, RuntimeError):
            pass
        return info

    def metrics_report(self) -> str:
        try:
            from torch_xla.debug import metrics as xm_metrics
            lines = []
            for name in sorted(xm_metrics.counter_names()):
                try:
                    lines.append("{}={}".format(name, xm_metrics.counter_value(name)))
                except Exception:
                    continue
            return "\n".join(lines)
        except (ImportError, AttributeError):
            return ""

    def compile_counters(self) -> Dict[str, int]:
        counters: Dict[str, int] = {}
        try:
            from torch_xla.debug import metrics as xm_metrics
            for name in xm_metrics.counter_names():
                try:
                    counters[name] = int(xm_metrics.counter_value(name))
                except Exception:
                    pass
        except (ImportError, AttributeError):
            pass
        if counters:
            return counters
        report = self.metrics_report()
        for key in ("CompileTime", "CacheHit", "CacheMiss", "PersistentCache", "CompilationCache"):
            pattern = re.compile(r"^({}[^ ]*)\s*=\s*(\d+)".format(key), re.M)
            for m, value in pattern.findall(report):
                counters[m] = int(value)
        return counters

    def write_cache_profile(self) -> str:
        """Record the fingerprint inputs next to the executable cache."""
        base = os.path.join(self.tpu_cache_dir, "executables")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(self.tpu_cache_dir, "cache_profile.json")
        if args.tpu_profile in (getattr(tpu_profile, "PROFILE_PID", None), getattr(tpu_profile, "PROFILE_PID_ALIAS", None)):
            tokenizer_constants = {
                "pixeldit_fixed_len": tpu_profile.PIXELDIT_FIXED_LEN,
                "pixeldit_conditioning_seq": tpu_profile.PIXELDIT_CONDITIONING_SEQ,
                "pixeldit_conditioning_features": tpu_profile.PIXELDIT_CONDITIONING_FEATURES,
                "pid_input": list(tpu_profile.PID_INPUT_LATENT_SHAPE),
                "pid_output": list(tpu_profile.PID_OUTPUT_LATENT_SHAPE),
            }
        else:
            tokenizer_constants = {
                "fixed_input_len": tpu_profile.TOKENIZER_FIXED_INPUT_LEN,
                "prefix_tokens": tpu_profile.TOKENIZER_PREFIX_TOKENS,
                "closing_tokens": tpu_profile.TOKENIZER_CLOSING_TOKENS,
            }
        payload = {
            "profile": args.tpu_profile,
            "torch": torch.__version__,
            "mesh_shape": MESH_SHAPE,
            "axis_names": MESH_AXIS_NAMES,
            "policy_version": tpu_sharding.POLICY_VERSION,
            "dtype": "bf16",
            "tokenizer_constants": tokenizer_constants,
            "manifest": json.loads(json.dumps(tpu_profile.load_manifest().get("artifacts", {}))),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def shutdown(self) -> None:
        try:
            self._runtime.shutdown()
        except (AttributeError, RuntimeError):
            pass

"""TPU memory helper for fused pipelines.

In the fused Krea2->PiD prompt both diffusion models are staged in the same
queue item. The PiD 4096x2304 XLA program alone reserves ~9GB of HBM per chip;
when the 25GB Krea2 weights are still resident the reservation can OOM even on
a v5e-8 (see tpu_request RESOURCE_EXHAUSTED). The host-side VAE chain is on
CPU, but the diffusion weights are on xla:0 and are kept in current_loaded_models
until the prompt finishes. This node is a tiny execution barrier: it forwards
its image input unchanged while draining and unloading the TPU-resident diffusion
models that are no longer needed downstream.

It is otherwise a no-op and is allowed by the TPU validator.
"""
import torch

import comfy.model_management
import comfy.accelerator
from comfy_api.latest import ComfyExtension, io


class TPUFlush(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TPUFlush",
            display_name="TPU Flush (fused Krea->PiD)",
            category="utils",
            description="Forwards IMAGE unchanged and unloads TPU diffusion models to free HBM for the next stage. No-op on CPU.",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        import logging, gc
        logging.info("TPUFlush: draining XLA and unloading diffusion models before PiD stage")
        try:
            comfy.accelerator.wait_device_ops()
        except Exception as e:
            logging.info(f"TPUFlush wait before: {e}")
        # Log memory before
        try:
            info_before = comfy.accelerator.memory_info()
            logging.info(f"TPUFlush memory before: {info_before}")
            loaded_before = comfy.model_management.current_loaded_models
            logging.info(f"TPUFlush loaded models before: {len(loaded_before)}")
            for lm in loaded_before:
                try:
                    patcher = lm.model
                    inner = getattr(patcher, 'model', None)
                    inner_name = type(inner).__name__ if inner is not None else 'None'
                    artifact = getattr(patcher, 'tpu_artifact', 'unknown')
                    size = getattr(patcher, 'size', 0)
                    dev = getattr(lm, 'device', 'unknown')
                    logging.info(f"  - patcher {type(patcher).__name__} inner {inner_name} artifact {artifact} size {size} device {dev} loaded {lm.model_loaded_memory()}")
                except Exception as e:
                    logging.info(f"  - log patcher failed: {e}")
            # Also log XLA metrics
            try:
                import torch_xla.debug.metrics as met
                logging.info(f"TPUFlush metrics before: {met.metrics_report()[:2000]}")
            except Exception as e:
                logging.info(f"TPUFlush metrics log failed: {e}")
        except Exception as e:
            logging.info(f"TPUFlush log before failed: {e}")
        # Unload all TPU-resident diffusion models.
        try:
            # Try explicit free_memory for xla:0 with huge requirement
            dev = comfy.model_management.get_torch_device()
            logging.info(f"TPUFlush trying free_memory on {dev}")
            freed_list = comfy.model_management.free_memory(1e30, dev)
            logging.info(f"TPUFlush free_memory freed {len(freed_list)} models")
            # Also try unload_all
            comfy.model_management.unload_all_models()
            logging.info(f"TPUFlush unload_all_models done")
            # Force GC and XLA cleanup
            gc.collect()
            comfy.model_management.soft_empty_cache()
            try:
                import torch_xla.core.xla_model as xm
                xm.wait_device_ops()
                logging.info("TPUFlush xm.wait_device_ops done")
            except Exception as e:
                logging.info(f"TPUFlush xm wait failed: {e}")
            # Try to clear XLA compilation cache to free Krea program HBM
            try:
                import torch_xla._XLAC as _xla
                # Clear pending IRs and compilation cache
                if hasattr(_xla, '_xla_release_compilation_cache'):
                    _xla._xla_release_compilation_cache()
                    logging.info("TPUFlush _xla_release_compilation_cache done")
                if hasattr(_xla, '_clear_pending_irs'):
                    _xla._clear_pending_irs('TPUFlush')
                    logging.info("TPUFlush _clear_pending_irs done")
            except Exception as e:
                logging.info(f"TPUFlush XLA cache clear failed: {e}")
            try:
                import torch_xla.debug.metrics as met
                met.clear_all()
                logging.info("TPUFlush metrics clear_all done")
            except Exception as e:
                logging.info(f"TPUFlush metrics clear failed: {e}")
            # Also try XLA runtime cache
            try:
                import torch_xla.runtime as xr
                # No explicit clear, but try to trigger GC
                logging.info(f"TPUFlush xr device count {xr.global_runtime_device_count()}")
            except Exception as e:
                logging.info(f"TPUFlush xr log failed: {e}")
        except Exception as e:
            logging.warning(f"TPUFlush unload failed: {e}", exc_info=True)
        try:
            comfy.accelerator.wait_device_ops()
        except Exception as e:
            logging.info(f"TPUFlush wait after: {e}")
        try:
            info_after = comfy.accelerator.memory_info()
            logging.info(f"TPUFlush memory after: {info_after}")
            loaded_after = comfy.model_management.current_loaded_models
            logging.info(f"TPUFlush loaded models after: {len(loaded_after)}")
            for lm in loaded_after:
                try:
                    patcher = lm.model
                    inner = getattr(patcher, 'model', None)
                    inner_name = type(inner).__name__ if inner is not None else 'None'
                    artifact = getattr(patcher, 'tpu_artifact', 'unknown')
                    logging.info(f"  - after patcher {type(patcher).__name__} inner {inner_name} artifact {artifact}")
                except Exception as e:
                    logging.info(f"  - after log failed: {e}")
        except Exception as e:
            logging.info(f"TPUFlush log after failed: {e}")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logging.info("TPUFlush: done, forwarding image")
        return io.NodeOutput(image)


class TPUBridge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TPUBridge",
            display_name="TPU Bridge (delay model load)",
            category="utils",
            description="Forwards MODEL unchanged but creates an execution dependency on an IMAGE. Use to delay PiD model load until after Krea flush.",
            inputs=[
                io.Custom("MODEL").Input("model"),
                io.Image.Input("image"),
            ],
            outputs=[
                io.Custom("MODEL").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, image) -> io.NodeOutput:
        return io.NodeOutput(model)


class TPUFlushExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TPUFlush, TPUBridge]


async def comfy_entrypoint() -> TPUFlushExtension:
    return TPUFlushExtension()

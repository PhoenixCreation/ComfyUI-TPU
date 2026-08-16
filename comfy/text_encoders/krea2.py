"""Krea 2 (K2) text encoder: Qwen3-VL-4B, 12-layer tap.

K2 conditions on a stack of hidden states from 12 layers of Qwen3-VL-4B
(reference taps ``hidden_states[2,5,8,...,35]``), kept as a ``(B, 12, seq, 2560)`` tensor and
consumed by the DiT's internal ``txtfusion`` adapter. Comfy carries conditioning as a 3D tensor,
so the 12-layer stack is flattened to ``(B, seq, 12*2560)`` here and unpacked inside the model.
"""

import logging
import numbers

import torch

import comfy.accelerator
import comfy.model_management
import comfy.text_encoders.qwen3vl
from comfy import sd1_clip
from comfy import tpu_profile

# tap k == hidden_states[k] (no offset).
KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]

# Identical system template to Qwen-Image; Krea2 strips the system+user-opening prefix.
KREA2_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class Krea2Tokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_4b")
        self.llama_template = KREA2_TEMPLATE  # conditioning template; image text-gen uses qwen3vl's default image template.

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=True, **kwargs):
        # Krea2 conditions on the no-think template; thinking=True drops the empty <think> block qwen3vl adds.
        tokens = super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, images=images, prevent_empty_text=prevent_empty_text, thinking=thinking, **kwargs)

        if comfy.model_management.xla_enabled() and not images:
            # TPU fixed-input contract (spec section 10): the model receives
            # exactly TOKENIZER_FIXED_INPUT_LEN tokens. Preserve the constant
            # prefix (34) and closing (5) runs; truncate or pad the content run
            # so every request compiles the same (B, 512, 2560) trace.
            with comfy.accelerator.stage_timer("tokenization"):
                fixed = tpu_profile.TOKENIZER_FIXED_INPUT_LEN
                prefix = tpu_profile.TOKENIZER_PREFIX_TOKENS
                closing = tpu_profile.TOKENIZER_CLOSING_TOKENS
                pad_id = tpu_profile.TOKENIZER_PAD_TOKEN
                key = next(iter(tokens))
                out_pairs = []
                for batch in tokens[key]:
                    if len(batch) > fixed:
                        logging.debug(
                            "TPU tokenizer: prompt content truncated from %d to %d tokens (fixed input %d, prefix %d, closing %d)",
                            len(batch), fixed, fixed, prefix, closing)
                        batch = batch[:prefix] + batch[prefix:fixed - closing] + batch[len(batch) - closing:]
                    if len(batch) < fixed:
                        batch = list(batch) + [(pad_id, 1.0)] * (fixed - len(batch))
                    out_pairs.append(batch)
                tokens[key] = out_pairs
        return tokens


class Krea2Qwen3VLClipModel(comfy.text_encoders.qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=KREA2_TAP_LAYERS, layer_idx=None, dtype=dtype,
                         attention_mask=attention_mask, model_options=model_options, model_type="qwen3vl_4b")


class Krea2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=Krea2Qwen3VLClipModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs, template_end=-1):
        if comfy.model_management.xla_enabled():
            # TPU: the tokenizer has already fixed the input length; the prefix
            # strip is the profile constant, so conditioning is always
            # (B, CONDITIONING_SEQ, 12*2560) and the trace is shape-stable.
            with comfy.accelerator.stage_timer("text_encoder"):
                out, pooled, extra = super().encode_token_weights(token_weight_pairs)
            # Shallow copy so the TPU strip/mask handling never mutates the
            # caller's conditioning dict (the mask key is rebound below).
            extra = dict(extra)
            template_end = tpu_profile.TOKENIZER_PREFIX_TOKENS
        else:
            out, pooled, extra = super().encode_token_weights(token_weight_pairs)  # out: (B, 12, seq, 2560)
            tok_pairs = token_weight_pairs["qwen3vl_4b"][0]

            # Strip the system + user-opening prefix
            count_im_start = 0
            if template_end == -1:
                for i, v in enumerate(tok_pairs):
                    elem = v[0]
                    if not torch.is_tensor(elem) and isinstance(elem, numbers.Integral):
                        if elem == 151644 and count_im_start < 2:
                            template_end = i
                            count_im_start += 1
                if out.shape[2] > (template_end + 3):
                    if tok_pairs[template_end + 1][0] == 872:      # "user"
                        if tok_pairs[template_end + 2][0] == 198:   # "\n"
                            template_end += 3

        out = out[:, :, template_end:]

        b, n, seq, h = out.shape
        # Flatten the 12-layer axis into the feature dim: (B, seq, 12*2560). Unpacked in the model.
        out = out.permute(0, 2, 1, 3).reshape(b, seq, n * h)

        if "attention_mask" in extra:
            # Cloned to avoid mutating the caller's dict on the TPU path.
            extra["attention_mask"] = extra["attention_mask"][:, template_end:]
            if comfy.model_management.xla_enabled():
                # TPU fixed-mask contract (spec section 10): the mask is kept
                # for every prompt, and the all-ones probe (a device-tensor .sum()
                # + bool() sync) is never evaluated on XLA.
                pass
            elif extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
                extra.pop("attention_mask")

        if comfy.model_management.xla_enabled():
            # TPU boundary (spec section 12): drain the text-encoder section
            # before conditioning is consumed by the denoiser trace.
            comfy.accelerator.mark_step()

        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class Krea2TEModel_(Krea2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Krea2TEModel_

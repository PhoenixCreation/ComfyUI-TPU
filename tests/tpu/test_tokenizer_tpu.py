"""Fixed-length tokenizer contract (spec section 10), measured against the
real bundled Qwen tokenizer: every prompt yields exactly
TOKENIZER_FIXED_INPUT_LEN tokens; the 34-token prefix and 5-token closing
runs are invariant; conditioning is (B, CONDITIONING_SEQ, 12*2560)."""

import torch

from comfy import tpu_profile
from comfy.text_encoders.krea2 import Krea2Tokenizer, Krea2TEModel

CLOSING = [151645, 198, 151644, 77091, 198]  # <|im_end|> \n <|im_start|> assistant \n


def _pairs(text, **kw):
    return next(iter(Krea2Tokenizer().tokenize_with_weights(text, **kw).values()))[0]


def test_empty_prompt_yields_fixed_length():
    assert len(_pairs("")) == tpu_profile.TOKENIZER_FIXED_INPUT_LEN


def test_padding_uses_pad_token():
    pairs = _pairs("")
    trailing = pairs[-tpu_profile.TOKENIZER_CLOSING_TOKENS:]
    assert all(int(p[0]) == tpu_profile.TOKENIZER_PAD_TOKEN for p in trailing)


def test_prefix_and_closing_runs_match_documented_constants():
    empty = _pairs("")
    prefix = tpu_profile.TOKENIZER_PREFIX_TOKENS
    assert [int(p[0]) for p in empty[prefix:prefix + 5]] == CLOSING


def test_prefix_invariant_across_prompts():
    empty = _pairs("")
    sample = _pairs("A landscape painting by Albert Bierstadt.")
    prefix = tpu_profile.TOKENIZER_PREFIX_TOKENS
    assert [p[0] for p in empty[:prefix]] == [p[0] for p in sample[:prefix]]


def test_short_and_unicode_prompts_same_length():
    lens = {len(_pairs(x)) for x in ("hello world", "héllo wörld 你好", "a", "solo: A man with a beard")}
    assert lens == {tpu_profile.TOKENIZER_FIXED_INPUT_LEN}


def test_long_prompt_truncated_preserving_closing_run():
    long_text = " ".join(["word"] * 500)
    pairs = _pairs(long_text)
    assert len(pairs) == tpu_profile.TOKENIZER_FIXED_INPUT_LEN
    assert [int(p[0]) for p in pairs[-5:]] == CLOSING


def test_skip_template_still_fixed_length_on_tpu():
    # Shape stability wins: every conditioning on TPU is fixed-length even for
    # raw (template-free) text; template text is the only in-profile path.
    pairs = _pairs("plain text without template", skip_template=True)
    assert len(pairs) == tpu_profile.TOKENIZER_FIXED_INPUT_LEN
    head = [int(p[0]) for p in pairs[:4]]
    assert 151643 not in head  # raw tokens preserved at the front, pads appended
    assert 151643 in [int(p[0]) for p in pairs]


def test_images_bypass_fixed_input():
    image = torch.zeros(1, 3, 8, 8)
    tokens = Krea2Tokenizer().tokenize_with_weights("describe", images=[image])
    pairs = next(iter(tokens.values()))[0]
    assert any(isinstance(p[0], dict) and p[0].get("type") == "image" for p in pairs)


def test_conditioning_shape_after_strip_and_flatten(tpu_mode, monkeypatch):
    """The TPU encode path strips the 34-token prefix and flattens the 12 tap
    layers into the feature dim; the attention mask is retained."""
    seq, n_layers, feat = 512, 12, 2560
    out = torch.zeros(1, n_layers, seq, feat)
    extra = {"attention_mask": torch.ones(1, seq, dtype=torch.long)}

    model = Krea2TEModel.__new__(Krea2TEModel)
    monkeypatch.setattr("comfy.sd1_clip.SD1ClipModel.encode_token_weights",
                        lambda self, t: (out, torch.zeros(1, 2560), dict(extra)))

    result, pooled, extra_out = model.encode_token_weights({"qwen3vl_4b": [[]]})
    assert tuple(result.shape) == (1, tpu_profile.CONDITIONING_SEQ, tpu_profile.CONDITIONING_FEATURES)
    assert "attention_mask" in extra_out
    assert tuple(extra_out["attention_mask"].shape) == (1, tpu_profile.CONDITIONING_SEQ)
    assert pooled.shape == (1, 2560)
    assert tpu_mode.mark_step_calls == 1


def test_mask_clone_does_not_mutate_caller(tpu_mode, monkeypatch):
    seq, n_layers, feat = 512, 12, 2560
    out = torch.zeros(1, n_layers, seq, feat)
    extra = {"attention_mask": torch.ones(1, seq, dtype=torch.long)}

    model = Krea2TEModel.__new__(Krea2TEModel)
    monkeypatch.setattr("comfy.sd1_clip.SD1ClipModel.encode_token_weights",
                        lambda self, t: (out, torch.zeros(1, 2560), extra))

    model.encode_token_weights({"qwen3vl_4b": [[]]})
    assert extra["attention_mask"].shape == (1, seq)
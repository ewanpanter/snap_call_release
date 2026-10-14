import string

import pytest
import torch

from tests.fixtures.tiny_qwen3 import build_tiny_qwen3


def test_tiny_model_is_deterministic(tiny_qwen3):
    model, tok = tiny_qwen3
    other, _ = build_tiny_qwen3()
    ids = tok("<|im_start|>user\nIs this billing?<|im_end|>", return_tensors="pt").input_ids
    with torch.no_grad():
        assert torch.equal(model(ids).logits, other(ids).logits)


def test_tiny_model_architecture(tiny_qwen3):
    model, tok = tiny_qwen3
    assert type(model).__name__ == "Qwen3ForCausalLM"
    assert model.config.num_hidden_layers == 2
    assert model.config.vocab_size == len(tok)
    assert not model.config.use_sliding_window


@pytest.mark.parametrize("label", list(string.ascii_uppercase + string.digits))
def test_tiny_labels_are_single_tokens(tiny_qwen3, label):
    _, tok = tiny_qwen3
    assert len(tok.encode(label, add_special_tokens=False)) == 1


def test_tiny_special_tokens(tiny_qwen3):
    _, tok = tiny_qwen3
    for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        assert len(tok.encode(t, add_special_tokens=False)) == 1
    assert tok.decode(tok.encode("héllo wörld", add_special_tokens=False)) == "héllo wörld"


def test_tiny_model_accepts_4d_mask(tiny_qwen3):
    """The masking code needs custom 4D masks and position ids; check the plumbing exists."""
    model, tok = tiny_qwen3
    ids = tok("abcdef", return_tensors="pt").input_ids
    n = ids.shape[1]
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
    mask = torch.zeros(1, 1, n, n).masked_fill(~causal, torch.finfo(torch.float32).min)
    pos = torch.arange(n).unsqueeze(0)
    with torch.no_grad():
        a = model(ids).logits
        b = model(ids, attention_mask=mask, position_ids=pos).logits
    torch.testing.assert_close(a, b)


@pytest.mark.real_weights
def test_real_labels_are_single_tokens(real_qwen3):
    """Qwen3's tokenizer must encode each option label as one token."""
    _, tok = real_qwen3
    for label in list(string.ascii_uppercase) + [str(i) for i in range(10)] + ["yes", "no"]:
        assert len(tok.encode(label, add_special_tokens=False)) == 1, label


@pytest.mark.real_weights
def test_real_model_forward(real_qwen3):
    model, tok = real_qwen3
    assert not model.config.use_sliding_window
    ids = tok("Hello", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids).logits
    assert out.shape[-1] == model.config.vocab_size
    assert torch.isfinite(out).all()

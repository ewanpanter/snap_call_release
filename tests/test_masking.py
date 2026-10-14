"""Mask equivalence: the packed single pass and the KV-cache version must
give the same logits as running every branch on its own."""

from pathlib import Path

import pytest
import torch

from snapcall.masking import (
    cached_forward,
    compute_tokens,
    full_attention_only,
    pack,
    packed_forward,
    separate_forward,
)
from snapcall.prompt import build_prompt, encode_prompt
from snapcall.schema import DecideRequest
from tests.fixtures.tiny_qwen3 import build_tiny_qwen3

EXAMPLE = Path(__file__).parent.parent / "contract" / "examples" / "decide_request.example.json"


def example_ids(tok) -> tuple[list[int], list[list[int]]]:
    request = DecideRequest.model_validate_json(EXAMPLE.read_text())
    enc = encode_prompt(build_prompt(request), tok)
    return enc.prefix_ids, enc.branch_ids


def test_pack_worked_example():
    """The example from masking.py's docstring: prefix s0 s1, branches (a0 a1) and (b0)."""
    packed = pack([10, 11], [[20, 21], [30]])
    assert packed.input_ids.tolist() == [[10, 11, 20, 21, 30]]
    assert packed.position_ids.tolist() == [[0, 1, 2, 3, 2]]
    assert packed.last_index == [3, 4]
    expected = [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 0, 0, 1],
    ]
    assert packed.allowed.int().tolist() == expected


def test_compute_tokens():
    # prefix 10, branches of 3 and 5 tokens
    assert compute_tokens("packed", 10, [3, 5]) == 18
    assert compute_tokens("cached", 10, [3, 5]) == 20  # the 3-token branch is padded to 5
    assert compute_tokens("separate", 10, [3, 5]) == 28  # prefix read twice
    with pytest.raises(ValueError):
        compute_tokens("magic", 10, [3])


def test_pack_rejects_empty_branch():
    with pytest.raises(ValueError):
        pack([1, 2], [[3], []])


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("forward", [packed_forward, cached_forward])
def test_mask_equivalence_tiny(attn, forward):
    model, tok = build_tiny_qwen3(attn_implementation=attn)
    prefix, branches = example_ids(tok)
    with torch.inference_mode():
        expected = separate_forward(model, prefix, branches)
        got = forward(model, prefix, branches)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_without_the_mask_branches_leak(tiny_qwen3):
    """Control: the same packed sequence with a plain causal mask gives different logits
    for later branches, so the equivalence test above isn't passing by accident."""
    model, tok = tiny_qwen3
    prefix, branches = example_ids(tok)
    packed = pack(prefix, branches)
    with torch.inference_mode():
        expected = separate_forward(model, prefix, branches)
        leaky = model(input_ids=packed.input_ids).logits[0, packed.last_index]
    torch.testing.assert_close(leaky[0], expected[0], atol=1e-5, rtol=1e-5)  # first is fine
    assert not torch.allclose(leaky[1:], expected[1:], atol=1e-3)


def test_branches_cannot_see_each_other(tiny_qwen3):
    model, tok = tiny_qwen3
    prefix = tok.encode("shared state\n")
    b1, b2, other = tok.encode("first?"), tok.encode("second?"), tok.encode("something else")
    with torch.inference_mode():
        a = packed_forward(model, prefix, [b1, b2])
        b = packed_forward(model, prefix, [other, b2])
    torch.testing.assert_close(a[1], b[1], atol=1e-5, rtol=1e-5)


def test_packed_forward_keeps_gradients(tiny_qwen3):
    """LoRA training backpropagates through the packed forward."""
    model, tok = build_tiny_qwen3()
    prefix, branches = example_ids(tok)
    logits = packed_forward(model, prefix, branches)
    logits[:, 0].sum().backward()
    assert model.model.embed_tokens.weight.grad is not None


@pytest.mark.real_weights
@pytest.mark.parametrize("forward", [packed_forward, cached_forward])
def test_mask_equivalence_real(real_qwen3, forward):
    """Same check on Qwen3-0.6B with the real tokenizer, float32 on CPU."""
    model, tok = real_qwen3
    prefix, branches = example_ids(tok)
    with torch.inference_mode():
        expected = separate_forward(model, prefix, branches)
        got = forward(model, prefix, branches)
    # Different sequence lengths change the order of float additions, so allow a little
    # numerical noise on logits of magnitude ~10-30.
    torch.testing.assert_close(got, expected, atol=2e-3, rtol=1e-4)
    assert torch.equal(got.argmax(-1), expected.argmax(-1))


# --- hybrid models (Qwen3.5: Gated DeltaNet linear attention + full attention) ----------


@pytest.fixture(scope="module")
def tiny_qwen35():
    """A 4-layer random-weights Qwen3.5 text model (3 linear-attention layers, 1 full)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=512,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        max_position_embeddings=4096,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()


def hybrid_ids():
    g = torch.Generator().manual_seed(1)
    prefix = torch.randint(1, 512, (20,), generator=g).tolist()
    return prefix, [torch.randint(1, 512, (n,), generator=g).tolist() for n in (5, 8, 3)]


def test_full_attention_only(tiny_qwen3, tiny_qwen35):
    assert full_attention_only(tiny_qwen3[0])
    assert not full_attention_only(tiny_qwen35)


def test_cached_forward_on_hybrid_model(tiny_qwen35):
    """The cache copy must cover linear-attention state too (batch_repeat_interleave
    raised AttributeError here before reorder_cache)."""
    prefix, branches = hybrid_ids()
    with torch.inference_mode():
        expected = separate_forward(tiny_qwen35, prefix, branches)
        got = cached_forward(tiny_qwen35, prefix, branches)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_packed_forward_refuses_hybrid_model(tiny_qwen35):
    prefix, branches = hybrid_ids()
    with pytest.raises(ValueError, match="full attention"):
        packed_forward(tiny_qwen35, prefix, branches)


def cfg_model(**cfg):
    """Just enough of a model for ``full_attention_only``: it only reads the config."""
    from types import SimpleNamespace

    return SimpleNamespace(config=SimpleNamespace(**cfg))


@pytest.mark.parametrize(
    ("cfg", "seq_len", "full"),
    [
        # Qwen3: window switched off, whatever sliding_window says
        (
            {"sliding_window": 4096, "use_sliding_window": False, "max_position_embeddings": 40960},
            None,
            True,
        ),
        # Phi-4-mini: one window for all layers, longer than the context
        ({"sliding_window": 262144, "max_position_embeddings": 131072}, None, True),
        # Phi-4: no window
        ({"sliding_window": None, "max_position_embeddings": 16384}, None, True),
        # Mistral-style 4k window, no layer_types: only inputs that fit the window
        ({"sliding_window": 4096, "max_position_embeddings": 32768}, None, False),
        ({"sliding_window": 4096, "max_position_embeddings": 32768}, 1800, True),
        ({"sliding_window": 4096, "max_position_embeddings": 32768}, 5000, False),
        # Gemma 4: 512-token sliding layers mixed with full ones
        (
            {
                "sliding_window": 512,
                "max_position_embeddings": 131072,
                "layer_types": ["sliding_attention", "full_attention"],
            },
            1800,
            False,
        ),
        (
            {
                "sliding_window": 512,
                "max_position_embeddings": 131072,
                "layer_types": ["sliding_attention", "full_attention"],
            },
            300,
            True,
        ),
        # Qwen3.5: linear attention is never full, however short the input
        (
            {
                "max_position_embeddings": 262144,
                "layer_types": ["linear_attention", "full_attention"],
            },
            10,
            False,
        ),
    ],
)
def test_full_attention_only_from_config(cfg, seq_len, full):
    assert full_attention_only(cfg_model(**cfg), seq_len) is full

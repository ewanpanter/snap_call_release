"""A 2-layer random-weights Qwen3 and a byte-level tokenizer, built in code.

No downloads, loads in well under a second on a CPU. Same architecture class as the
real model (Qwen3ForCausalLM: full attention, GQA, RoPE), so masking and KV-cache code
exercised here behaves the same on Qwen3-0.6B / 4B. Outputs are meaningless: use it for
shape, equivalence and plumbing tests, never for accuracy.

The tokenizer is byte-level BPE with no merges plus Qwen's chat special tokens, so every
printable ASCII character (including the labels A-Z, a-z, 0-9) is a single token, and a
leading space is its own token ("Ġ"), as in Qwen's tokenizer.
"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
SEED = 0


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    for tok in SPECIAL_TOKENS:
        vocab[tok] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL_TOKENS[1:],
    )
    fast.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS[1:]})
    return fast


def tiny_qwen3_config(vocab_size: int, attn_implementation: str = "eager") -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
        use_sliding_window=False,
        pad_token_id=0,
    )
    config._attn_implementation = attn_implementation
    return config


def build_tiny_qwen3(
    attn_implementation: str = "eager", seed: int = SEED
) -> tuple[Qwen3ForCausalLM, PreTrainedTokenizerFast]:
    """Return (model in eval mode, float32, CPU; tokenizer). Deterministic for a given seed."""
    tokenizer = build_tiny_tokenizer()
    config = tiny_qwen3_config(len(tokenizer), attn_implementation)
    config.pad_token_id = tokenizer.pad_token_id
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(config).eval()
    return model, tokenizer

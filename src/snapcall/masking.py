"""Answer every question branch against one shared prefix. Three ways to do it:

1. ``packed_forward``: the single-pass trick. Put the prefix and all branches in **one**
   sequence and give the model a custom attention mask so that each branch sees the
   prefix and itself, but not the other branches. Position ids restart after the prefix
   for every branch, so each branch looks to the model exactly as if it were the only
   question. Needs full attention in every layer: ``packed_forward`` refuses
   models with sliding-window or linear-attention layers (``full_attention_only``).

2. ``cached_forward``: works on any architecture. Run the prefix once, keep its cache,
   copy the cache once per branch, and run all branches as a batch on top of it. On hybrid
   models such as Qwen3.5 the "cache" also holds linear-attention conv and recurrent state;
   the copy covers that too.

3. ``separate_forward``: the naive reference. Run ``prefix + branch`` from scratch for each
   branch. The tests check that 1 and 2 match it.

Each returns the next-token logits at the last token of every branch, shape
``[n_branches, vocab]``, in branch order. None of them calls ``torch.no_grad()``, so LoRA
training can backpropagate through them. Wrap them in ``torch.inference_mode()`` for inference.

Worked example: prefix ``s0 s1``, branch 1 ``a0 a1``, branch 2 ``b0``. Packed sequence
``s0 s1 a0 a1 b0``, position ids ``0 1 2 3 2``. Mask (1 = may attend; rows are queries)::

           s0 s1 a0 a1 b0
       s0   1  .  .  .  .
       s1   1  1  .  .  .
       a0   1  1  1  .  .
       a1   1  1  1  1  .
       b0   1  1  .  .  1     <- b0 skips a0, a1, and sits at position 2 like a0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedInputs:
    input_ids: torch.Tensor  # [1, T]
    position_ids: torch.Tensor  # [1, T]
    allowed: torch.Tensor  # [T, T] bool: allowed[i, j] = token i may attend to token j
    last_index: list[int]  # index in the sequence of each branch's last token


def pack(prefix_ids: list[int], branch_ids: list[list[int]]) -> PackedInputs:
    """Lay out ``prefix + branch_1 + branch_2 + ...`` with its mask and position ids."""
    p = len(prefix_ids)
    total = p + sum(len(b) for b in branch_ids)
    allowed = torch.zeros(total, total, dtype=torch.bool)
    # The prefix is ordinary causal text: each token sees itself and what came before.
    allowed[:p, :p] = torch.ones(p, p, dtype=torch.bool).tril()

    ids, positions, last_index = list(prefix_ids), list(range(p)), []
    start = p
    for branch in branch_ids:
        n = len(branch)
        if n == 0:
            raise ValueError("empty branch")
        end = start + n
        allowed[start:end, :p] = True  # every branch token sees the whole prefix
        allowed[start:end, start:end] = torch.ones(n, n, dtype=torch.bool).tril()  # causal
        ids += branch
        positions += range(p, p + n)  # restart numbering right after the prefix
        last_index.append(end - 1)
        start = end

    return PackedInputs(
        input_ids=torch.tensor([ids]),
        position_ids=torch.tensor([positions]),
        allowed=allowed,
        last_index=last_index,
    )


def additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bool ``[T, T]`` -> the ``[1, 1, T, T]`` float mask HF adds to attention scores:
    0 where allowed, a huge negative number where not (so softmax gives it ~0 weight)."""
    mask = torch.zeros(allowed.shape, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


def full_attention_only(model, seq_len: int | None = None) -> bool:
    """True if every layer attends to the whole sequence, so the packed mask reaches them all.

    Linear-attention layers (Qwen3.5's Gated DeltaNet) never do: a recurrent layer runs
    straight through the packed sequence, so every branch after the first sees the branches
    before it. Nothing errors, the answers are just wrong (measured: far from full agreement on
    Qwen3.5-4B), hence this check.

    A sliding-window layer is full attention when its window covers the whole input:
    ``seq_len`` tokens if given, else the model's maximum context (then it holds for any
    input). Phi-4-mini sets ``sliding_window=262144`` with a 131,072-token context, so it
    counts as full attention; Gemma 4's 512-token window doesn't. ``use_sliding_window=False``
    (Qwen3) switches the window off whatever ``sliding_window`` says.
    """
    cfg = getattr(model.config, "text_config", None) or model.config
    longest = seq_len or getattr(cfg, "max_position_embeddings", None)
    window = getattr(cfg, "sliding_window", None)
    if getattr(cfg, "use_sliding_window", None) is False:
        window = None
    covers = window is None or (longest is not None and window >= longest)
    types = getattr(cfg, "layer_types", None)
    if not types:  # one kind of layer for the whole model
        return covers
    return all(t == "full_attention" or (t == "sliding_attention" and covers) for t in types)


def _model_dtype_device(model) -> tuple[torch.dtype, torch.device]:
    p = next(model.parameters())
    return p.dtype, p.device


def packed_forward(model, prefix_ids: list[int], branch_ids: list[list[int]]) -> torch.Tensor:
    """All branches in one forward pass, using a custom 4D mask. Full-attention models only.

    Use ``attn_implementation="eager"`` or ``"sdpa"``: flash-attention ignores custom
    masks. Raises ``ValueError`` for models that aren't full attention in every
    layer; use ``cached_forward`` for those.
    """
    if not full_attention_only(model, len(prefix_ids) + sum(len(b) for b in branch_ids)):
        raise ValueError(
            "packed_forward needs full attention in every layer; this model has "
            "sliding-window or linear-attention layers. Use cached_forward."
        )
    dtype, device = _model_dtype_device(model)
    packed = pack(prefix_ids, branch_ids)
    logits = model(
        input_ids=packed.input_ids.to(device),
        position_ids=packed.position_ids.to(device),
        attention_mask=additive_mask(packed.allowed, dtype).to(device),
        use_cache=False,
    ).logits
    return logits[0, packed.last_index]


def cached_forward(model, prefix_ids: list[int], branch_ids: list[list[int]]) -> torch.Tensor:
    """Prefix once, then all branches as one batch on top of copies of its KV cache.

    Branches have different lengths, so they're right-padded to the longest. Padding
    comes after each branch's last real token, and attention is causal, so the padding
    can't affect the logits we read.
    """
    _, device = _model_dtype_device(model)
    p, b = len(prefix_ids), len(branch_ids)

    prefix = model(input_ids=torch.tensor([prefix_ids], device=device), use_cache=True)
    cache = prefix.past_key_values
    # [1, ...] -> [b, ...]: one full copy of the prefix's state per branch. Simple, but memory
    # grows with (branches x prefix length). Production servers avoid the copy: branches
    # point at one shared prefix in a paged KV cache (e.g. vLLM's prefix caching).
    # reorder_cache (beam search's "pick these rows") with b zeros = b copies of row 0. Unlike
    # batch_repeat_interleave, every cache layer type has it, including the linear-attention
    # layers of hybrid models (conv + recurrent state), where the latter raises.
    cache.reorder_cache(torch.zeros(b, dtype=torch.long, device=device))

    longest = max(len(x) for x in branch_ids)
    pad_id = 0  # any valid id: padded positions are masked out and their logits unused
    input_ids = torch.full((b, longest), pad_id, dtype=torch.long)
    real = torch.zeros(b, longest, dtype=torch.long)  # 1 = real token, 0 = padding
    for i, branch in enumerate(branch_ids):
        input_ids[i, : len(branch)] = torch.tensor(branch)
        real[i, : len(branch)] = 1

    # The 2D mask covers cache + new tokens; the prefix is always visible.
    attention_mask = torch.cat([torch.ones(b, p, dtype=torch.long), real], dim=1)
    position_ids = torch.arange(p, p + longest).expand(b, longest)

    logits = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        position_ids=position_ids.to(device),
        past_key_values=cache,
        use_cache=True,
    ).logits
    last = torch.tensor([len(x) - 1 for x in branch_ids], device=device)
    return logits[torch.arange(b, device=device), last]


def separate_forward(model, prefix_ids: list[int], branch_ids: list[list[int]]) -> torch.Tensor:
    """Reference: one ordinary forward pass per branch, re-reading the prefix every time."""
    _, device = _model_dtype_device(model)
    rows = []
    for branch in branch_ids:
        ids = torch.tensor([prefix_ids + branch], device=device)
        rows.append(model(input_ids=ids, use_cache=False).logits[0, -1])
    return torch.stack(rows)


def compute_tokens(forward: str, prefix_len: int, branch_lens: list[int]) -> int:
    """Token positions each forward actually runs through the model, including repeated
    and padded work. Compare with ``EncodedPrompt.n_tokens`` (prefix once + branches), which
    is the billing-style number and the same for all three.
    """
    if forward == "packed":
        return prefix_len + sum(branch_lens)
    if forward == "cached":
        return prefix_len + len(branch_lens) * max(branch_lens)  # right-padded batch
    if forward == "separate":
        return len(branch_lens) * prefix_len + sum(branch_lens)  # prefix re-read per branch
    raise ValueError(f"unknown forward {forward!r}")


FORWARDS = {"packed": packed_forward, "cached": cached_forward, "separate": separate_forward}

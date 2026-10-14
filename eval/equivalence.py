"""Is masking exact? Compare packed and cached against separate, in several dtypes.

The first GPU smoke test found, in bf16, label-probability gaps larger than a 0.01
tolerance (packed and cached), with the same answers. Two
explanations fit: the masking is subtly wrong, or bf16 rounds differently when the same maths
is laid out differently (a packed sequence, a batch on a copied cache, one sequence per branch
all use different kernel shapes and summation orders). This check tells them apart:

1. In **float32** the rounding noise is ~1e-7 per operation, so if masking is exact the gap
   should fall to ~1e-5 or below. A real masking bug would stay large in any dtype.
2. In **bfloat16** we also measure the *noise floor*: how far separate-in-bf16 (the reference
   itself) is from separate-in-fp32. If packed/cached are about as far from separate as
   separate is from fp32, the gap is bf16 rounding, and a bf16 tolerance should be set from
   that floor rather than from 0.01.

Everything is compared on each branch's restricted label distribution (what the readout uses)
and on the raw label logits.
"""

from __future__ import annotations

import math
import statistics

import torch

from snapcall.masking import FORWARDS
from snapcall.prompt import DEFAULT_LABELS, build_prompt, encode_prompt, profile_for
from snapcall.readout import restricted_softmax
from snapcall.schema import DecideRequest

METHODS = ("separate", "packed", "cached")


def label_logits(model, tokenizer, request: DecideRequest, methods=METHODS) -> dict:
    """For each method: one float32 CPU tensor of label logits per branch."""
    profile = profile_for(tokenizer)
    prompt = build_prompt(request, DEFAULT_LABELS, profile)
    enc = encode_prompt(prompt, tokenizer)
    out = {}
    with torch.inference_mode():
        for name in methods:
            try:
                logits = FORWARDS[name](model, enc.prefix_ids, enc.branch_ids)
            except Exception as e:  # a finding in itself (e.g. an architecture without it)
                out[name] = None
                out.setdefault("_errors", {})[name] = f"{type(e).__name__}: {e}"[:500]
                continue
            out[name] = [
                row[enc.label_ids[: len(b.keys)]].float().cpu()
                for row, b in zip(logits, prompt.branches, strict=True)
            ]
    out["_n_tokens"] = enc.n_tokens
    return out


def _probs(rows: list[torch.Tensor]) -> list[torch.Tensor]:
    return [restricted_softmax(r, list(range(len(r)))) for r in rows]


def compare(got: list[torch.Tensor] | None, ref: list[torch.Tensor] | None) -> dict | None:
    """Max |diff| in probabilities and logits over all branches, and argmax agreement."""
    if got is None or ref is None:
        return None
    pg, pr = _probs(got), _probs(ref)
    return {
        "max_abs_prob_diff": max((a - b).abs().max().item() for a, b in zip(pg, pr, strict=True)),
        "max_abs_logit_diff": max(
            (a - b).abs().max().item() for a, b in zip(got, ref, strict=True)
        ),
        "argmax_agree": sum(int(a.argmax() == b.argmax()) for a, b in zip(pg, pr, strict=True)),
        "n_branches": len(got),
    }


def summarise(rows: list[dict | None]) -> dict:
    """Per-request comparisons -> max / p95 / median of the gaps, total argmax agreement.
    ``None`` rows (a method that raised) are counted as errors."""
    n_failed = sum(r is None for r in rows)
    rows = [r for r in rows if r is not None]
    if not rows:
        return {"n_failed": n_failed}
    probs = sorted(r["max_abs_prob_diff"] for r in rows)
    logits = sorted(r["max_abs_logit_diff"] for r in rows)
    n_branches = sum(r["n_branches"] for r in rows)
    agree = sum(r["argmax_agree"] for r in rows)

    def p95(xs):
        return xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]

    return {
        "n_requests": len(rows),
        "n_branches": n_branches,
        "prob_diff_max": probs[-1],
        "prob_diff_p95": p95(probs),
        "prob_diff_median": statistics.median(probs),
        "logit_diff_max": logits[-1],
        "logit_diff_median": statistics.median(logits),
        "argmax_agree": agree,
        "argmax_agree_rate": agree / n_branches,
        "n_failed": n_failed,
    }


def run(model, tokenizer, requests: list[DecideRequest], methods=METHODS) -> list[dict]:
    """Label logits for every request and method (kept for cross-dtype comparisons)."""
    return [label_logits(model, tokenizer, r, methods) for r in requests]


def report(
    by_dtype: dict[str, list[dict]], reference_dtype: str = "float32", methods=METHODS
) -> dict:
    """``by_dtype``: dtype -> ``run()`` output for the same requests.

    For every dtype: each method vs separate in the same dtype. For every dtype other than
    the reference: separate vs separate-in-the-reference-dtype (the noise floor), and each
    method vs that reference (how far each is from the "true" answer).
    """
    out: dict = {}
    ref = by_dtype.get(reference_dtype)
    for dtype, runs in by_dtype.items():
        section: dict = {}
        for m in methods:
            if m == "separate":
                continue
            section[f"{m}_vs_separate"] = summarise([compare(r[m], r["separate"]) for r in runs])
        if ref is not None and dtype != reference_dtype:
            for m in methods:
                section[f"{m}_vs_{reference_dtype}_separate"] = summarise(
                    [compare(r[m], x["separate"]) for r, x in zip(runs, ref, strict=True)]
                )
        section["n_tokens_max"] = max(r["_n_tokens"] for r in runs)
        errors = {k: v for r in runs for k, v in r.get("_errors", {}).items()}
        if errors:
            section["errors"] = errors  # one example message per method
        out[dtype] = section
    return out


def suggested_tolerance(section: dict, reference_dtype: str = "float32") -> float | None:
    """A bf16 tolerance justified by the measurement: 2x the largest gap between
    separate-in-bf16 and separate-in-fp32 (the reference's own rounding error), rounded up
    to one significant figure. Packed/cached passing this are within bf16 noise."""
    floor = section.get(f"separate_vs_{reference_dtype}_separate", {}).get("prob_diff_max")
    if floor is None:
        return None
    x = 2 * floor
    if x == 0:
        return 0.0
    e = 10 ** math.floor(math.log10(x))
    return math.ceil(round(x / e, 9)) * e

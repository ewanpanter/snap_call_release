"""Run a whole ``DecideRequest``: prompt -> forward -> readout. Plus two small tools.

``decide()`` is what serving and eval call. It picks how to run the forward:

- ``"packed"`` (default): one pass with the branch mask. Needs an attention backend that
  honours custom masks (eager or sdpa), and full attention in every layer. With any other
  backend, e.g. flash-attention, or a hybrid / sliding-window model, we warn and use
  ``"cached"`` instead. Long prompts also switch to ``"cached"``, because the
  packed mask is (prompt tokens)^2 in size.
- ``"cached"``: prefix once, branches as a batch on copies of its KV cache.
- ``"separate"``: one full forward per branch; the slow reference.

Tools:

- ``label_prior()``: the label-prior probe. How much the model likes each
  label when there's nothing to decide. Useful for contextual calibration.
- ``equivalence_report()`` and ``python -m snapcall.decide smoke --model ... [--dtype ...]``:
  the GPU smoke test. Checks the labels are single tokens and that packed and cached
  match separate (exactly in float32, the default; loosely in bfloat16), and writes
  ``outputs/equivalence.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from snapcall.masking import FORWARDS, compute_tokens, full_attention_only
from snapcall.prompt import (
    DEFAULT_LABELS,
    AnyQuestion,
    PromptProfile,
    build_prompt,
    encode_prompt,
    label_token_ids,
    profile_for,
)
from snapcall.readout import QuestionResult, read_branches, restricted_softmax, to_answer
from snapcall.schema import (
    ChoiceQuestion,
    DecideRequest,
    DecideResponse,
    QuestionType,
    Usage,
)

Forward = Literal["packed", "cached", "separate"]

# Above this many prompt tokens (prefix + all branches), "packed" switches to "cached".
PACKED_MAX_TOKENS = 2048
# Longest state we accept, counted as prefix tokens (the state plus a short system prompt).
MAX_STATE_TOKENS = 32_768
# Attention backends that apply a custom 4D mask as given.
MASK_SAFE_ATTENTION = ("eager", "sdpa")


class StateTooLong(ValueError):
    pass


def choose_forward(
    requested: Forward,
    attn_implementation: str,
    n_tokens: int,
    packed_max_tokens: int,
    full_attention: bool = True,
) -> Forward:
    """The forward that will actually run."""
    if requested != "packed":
        return requested
    if not full_attention:
        warnings.warn(
            "this model has sliding-window or linear-attention layers, which ignore the "
            "packed mask; using the KV-cache forward instead",
            stacklevel=3,
        )
        return "cached"
    if attn_implementation not in MASK_SAFE_ATTENTION:
        warnings.warn(
            f"attention backend {attn_implementation!r} may ignore custom masks; "
            "using the KV-cache forward instead of packed",
            stacklevel=3,
        )
        return "cached"
    if n_tokens > packed_max_tokens:
        return "cached"
    return "packed"


@dataclass(frozen=True)
class Decision:
    """A ``DecideResponse`` plus what eval needs to explain it."""

    response: DecideResponse
    results: list[QuestionResult]  # includes each permutation's distribution
    forward: Forward  # the forward that actually ran
    compute_tokens: int  # token positions run through the model, incl. repeats and padding


def decide_detailed(
    model,
    tokenizer,
    request: DecideRequest,
    *,
    forward: Forward = "packed",
    temperature: dict[QuestionType, float] | None = None,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    profile: PromptProfile | None = None,
    packed_max_tokens: int = PACKED_MAX_TOKENS,
    model_name: str = "snapcall",
) -> Decision:
    """Answer every question in ``request``. ``profile=None`` reads it from the tokenizer's
    chat template; pass one to force the empty thinking block on or off."""
    start = time.perf_counter()
    profile = profile_for(tokenizer) if profile is None else profile
    prompt = build_prompt(request, labels, profile)
    encoded = encode_prompt(prompt, tokenizer)
    if len(encoded.prefix_ids) > MAX_STATE_TOKENS:
        raise StateTooLong(
            f"state is {len(encoded.prefix_ids)} tokens with the system prompt; "
            f"the limit is {MAX_STATE_TOKENS}"
        )

    attn = getattr(model.config, "_attn_implementation", None) or "eager"
    used = choose_forward(
        forward,
        attn,
        encoded.n_tokens,
        packed_max_tokens,
        full_attention_only(model, encoded.n_tokens),
    )
    with torch.inference_mode():
        logits = FORWARDS[used](model, encoded.prefix_ids, encoded.branch_ids)

    results = read_branches(request, prompt.branches, logits, encoded.label_ids, temperature)
    answers = {r.question_id: to_answer(request.questions[r.question_id], r) for r in results}
    response = DecideResponse(
        model=model_name,
        answers=answers,
        usage=Usage(input_tokens=encoded.n_tokens, output_tokens=0),
        latency_ms=(time.perf_counter() - start) * 1000,
    )
    branch_lens = [len(b) for b in encoded.branch_ids]
    return Decision(
        response, results, used, compute_tokens(used, len(encoded.prefix_ids), branch_lens)
    )


def decide(model, tokenizer, request: DecideRequest, **kwargs) -> DecideResponse:
    """Answer every question in ``request``. Same options as ``decide_detailed``."""
    return decide_detailed(model, tokenizer, request, **kwargs).response


# --------------------------------------------------------------------------------------
# Label-prior probe
# --------------------------------------------------------------------------------------

CONTENT_FREE = "N/A"

# Candidate label sets for the probe chart. The flattest prior makes the best fixed labels.
CANDIDATE_LABELS: dict[str, tuple[str, ...]] = {
    "A-D": ("A", "B", "C", "D"),
    "1-4": ("1", "2", "3", "4"),
    "random letters": ("K", "Q", "M", "X"),
}


def label_prior(
    model,
    tokenizer,
    labels: tuple[str, ...],
    *,
    question: AnyQuestion | None = None,
    profile: PromptProfile | None = None,
) -> dict[str, float]:
    """P(label) when there's nothing to decide: the state is "N/A", and so is every
    option unless you pass a real ``question`` (to get the prior for that question's
    wording). One branch, in the original order, so label i is option i.
    """
    if question is None:
        options = {f"o{i}": CONTENT_FREE for i in range(len(labels))}
        question = ChoiceQuestion(question=CONTENT_FREE, options=options)
    request = DecideRequest(state=CONTENT_FREE, questions={"probe": question}, debias="off")
    (result,) = decide_detailed(model, tokenizer, request, labels=labels, profile=profile).results
    return dict(zip(labels, result.per_permutation[0].values(), strict=False))


def flatness(prior: dict[str, float]) -> float:
    """Entropy of ``prior`` divided by its maximum: 1 = no label preferred, 0 = one label."""
    entropy = -sum(p * math.log(p) for p in prior.values() if p > 0)
    return entropy / math.log(len(prior))


# --------------------------------------------------------------------------------------
# Equivalence check (CPU tests and the GPU smoke test)
# --------------------------------------------------------------------------------------

# Measured with eval/equivalence.py on Qwen3-0.6B and Qwen3-4B-Instruct-2507:
# in float32, packed and cached match separate to <= 8.4e-5 with every answer the same, so
# masking is exact. In bfloat16 the reference itself moves by up to 0.33 against its own
# float32 result: near-ties swing on bf16 rounding. A max-gap tolerance can't work there.
# So the check depends on the dtype:
# - float32 (exactness): every branch's label probabilities within FP32_PROB_TOLERANCE (10x
#   the worst gap seen) and every top label the same.
# - half precision (bf16/fp16, sanity only): at least HALF_MIN_SAME_ANSWER of branches keep
#   their top label, and the 95th-percentile gap is at most HALF_P95_TOLERANCE (worst seen:
#   98.8% and 0.185 over 337 branches).
FP32_PROB_TOLERANCE = 1e-3
HALF_MIN_SAME_ANSWER = 0.98
HALF_P95_TOLERANCE = 0.2


def _p95(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]


def compare_branches(got: list[torch.Tensor], reference: list[torch.Tensor], dtype) -> dict:
    """One method's branch probabilities vs the reference's, judged for ``dtype``."""
    gaps = [(g - r).abs().max().item() for g, r in zip(got, reference, strict=True)]
    same = [bool(g.argmax() == r.argmax()) for g, r in zip(got, reference, strict=True)]
    out = {
        "max_abs_prob_diff": max(gaps),
        "p95_abs_prob_diff": _p95(gaps),
        "same_argmax": all(same),
        "same_argmax_rate": sum(same) / len(same),
    }
    if dtype == torch.float32:
        out["check"] = f"float32: max gap < {FP32_PROB_TOLERANCE}, every answer the same"
        out["ok"] = out["max_abs_prob_diff"] < FP32_PROB_TOLERANCE and out["same_argmax"]
    else:
        out["check"] = (
            f"{str(dtype).removeprefix('torch.')}: >= {HALF_MIN_SAME_ANSWER:.0%} same answer, "
            f"p95 gap <= {HALF_P95_TOLERANCE}"
        )
        out["ok"] = (
            out["same_argmax_rate"] >= HALF_MIN_SAME_ANSWER
            and out["p95_abs_prob_diff"] <= HALF_P95_TOLERANCE
        )
    return out


def equivalence_report(
    model,
    tokenizer,
    request: DecideRequest,
    *,
    profile: PromptProfile | None = None,
) -> dict:
    """Compare packed and cached against separate on every branch's label probabilities.
    How strict the check is depends on the model's dtype (see ``compare_branches``)."""
    profile = profile_for(tokenizer) if profile is None else profile
    prompt = build_prompt(request, DEFAULT_LABELS, profile)
    enc = encode_prompt(prompt, tokenizer)
    dtype = next(model.parameters()).dtype

    def label_probs(logits: torch.Tensor) -> list[torch.Tensor]:
        return [
            restricted_softmax(row, enc.label_ids[: len(branch.keys)])
            for row, branch in zip(logits, prompt.branches, strict=True)
        ]

    with torch.inference_mode():
        reference = label_probs(FORWARDS["separate"](model, enc.prefix_ids, enc.branch_ids))
        report: dict = {"empty_think": profile.empty_think, "n_branches": len(prompt.branches)}
        for name in ("packed", "cached"):
            got = label_probs(FORWARDS[name](model, enc.prefix_ids, enc.branch_ids))
            report[name] = compare_branches(got, reference, dtype)
    report["ok"] = report["packed"]["ok"] and report["cached"]["ok"]
    return report


# A support ticket with one question of each type, for the smoke test.
SMOKE_REQUEST = DecideRequest.model_validate(
    {
        "state": "Subject: Charged twice\n\nMy card was billed twice for my mobile plan "
        "this month. Please refund the duplicate charge.",
        "questions": {
            "queue": {
                "type": "choice",
                "question": "Which queue should handle this ticket?",
                "options": {"billing": "Billing", "technical": "Technical", "sales": "Sales"},
            },
            "priority": {
                "type": "score",
                "question": "How urgent is this ticket?",
                "levels": ["low", "medium", "high", "critical"],
            },
            "is_billing": {"type": "yesno", "question": "Is this about billing or a refund?"},
        },
    }
)


def _load(model_path: str, dtype_name: str = "float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation="sdpa"
    )
    return model.to(device).eval(), tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m snapcall.decide")
    parser.add_argument("command", choices=["smoke", "probe"])
    parser.add_argument("--model", required=True, help="HF id or local path (e.g. /gcs/...)")
    parser.add_argument("--out", default="outputs")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="float32 (default) is the exactness check; bfloat16 is a looser sanity check, "
        "for models too big for float32 on one GPU (e.g. Qwen3-8B on an L4)",
    )
    args = parser.parse_args(argv)

    model, tokenizer = _load(args.model, args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {"model": args.model, "device": str(model.device), "dtype": str(model.dtype)}

    if args.command == "smoke":
        label_token_ids(tokenizer, DEFAULT_LABELS)  # raises if any label is multi-token
        report = {**info, **equivalence_report(model, tokenizer, SMOKE_REQUEST)}
        (out / "equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    priors = {name: label_prior(model, tokenizer, ls) for name, ls in CANDIDATE_LABELS.items()}
    report = {
        **info,
        "priors": priors,
        "flatness": {name: flatness(p) for name, p in priors.items()},
    }
    (out / "label_prior.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

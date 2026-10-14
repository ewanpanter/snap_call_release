"""The GPU job for one model (Decision 7): equivalence, zero-shot eval, latency, label prior.

Run on the burst runner, one model per job (only one GPU exists project-wide)::

    gh workflow run run-gpu.yml -f script=eval.run_eval \\
      -f args="--model {models}/Qwen/Qwen3-4B-Instruct-2507 \\
               --data gs://<project>-scratch/data/test.jsonl"

Stages, in order (``--stages`` picks a subset; each one's failure is recorded and the rest
still run, so one bug doesn't waste the GPU slot):

1. ``equivalence``: packed and cached vs separate on the smoke request plus the first
   ``--equivalence-states`` test states, for every model in ``--equivalence-models`` (default:
   ``--model``) in every dtype in ``--equivalence-dtypes``. -> ``equivalence.json``.
   ``--gate-cached`` stops the job here unless cached matches separate in float32 (the
   cache-copy smoke test for non-Qwen3 architectures).
2. ``eval``: every test state through ``decide`` (debias auto: all permutations in one pass).
   -> ``predictions.jsonl`` (with per-permutation distributions in ``meta``), ``metrics*.json``,
   ``eval_timings.json`` (for cost per 1k decisions).
3. ``latency``: 1/2/4/8 questions per state (``--latency-n-questions``) x debias off/auto x
   each forward.
   -> ``latency.jsonl``.
4. ``prior``: the label-prior probe. -> ``label_prior.json``.

Every stage writes ``stages.json`` as it finishes. Exit status 1 if any stage failed.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from eval import equivalence
from eval.common import device_info, free, load_model, local_path, model_label, sync, write_json
from eval.metrics import compute_all
from snapcall.decide import CANDIDATE_LABELS, SMOKE_REQUEST, decide_detailed, flatness, label_prior
from snapcall.schema import (
    DataRecord,
    Prediction,
    predictions_from_response,
    read_jsonl,
    write_jsonl,
)

STAGES = ("equivalence", "eval", "latency", "prior")
# The cache-copy smoke test passes if cached matches separate this closely in float32.
GATE_PROB_DIFF = 1e-3


def round_robin(records: list[DataRecord], n: int) -> list[DataRecord]:
    """``n`` records alternating between datasets, so small samples cover both."""
    by_ds: dict[str, list[DataRecord]] = {}
    for r in records:
        by_ds.setdefault(r.dataset, []).append(r)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by_ds.values()):
        for v in by_ds.values():
            if i < len(v) and len(out) < n:
                out.append(v[i])
        i += 1
    return out


# --------------------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------------------


def stage_equivalence(args, records: list[DataRecord], out: Path) -> dict:
    methods = equivalence.METHODS
    pool = records
    if args.equivalence_datasets:  # e.g. skip telco's 26-option prompts for big fp32 models
        pool = [r for r in records if r.dataset in args.equivalence_datasets]
    requests = [SMOKE_REQUEST] + [
        r.to_request(debias="auto") for r in round_robin(pool, args.equivalence_states)
    ]
    body: dict = {"device": device_info(), "n_requests": len(requests), "models": {}}
    for model_path in args.equivalence_models or [args.model]:
        by_dtype = {}
        for dtype in args.equivalence_dtypes:
            t0 = time.perf_counter()
            model, tok = load_model(model_path, dtype, quant=args.quant)
            # An 8-bit model's section is labelled "8bit", whatever its compute dtype.
            by_dtype[args.quant or dtype] = equivalence.run(model, tok, requests, methods)
            del model
            free()
            print(f"equivalence {model_label(model_path)} {dtype}: {time.perf_counter() - t0:.0f}s")
        rep = equivalence.report(by_dtype, methods=methods)
        for dtype, section in rep.items():
            if dtype != "float32" and "float32" in rep:
                section["suggested_tolerance"] = equivalence.suggested_tolerance(section)
        body["models"][model_label(model_path)] = rep
        write_json(out / "equivalence.json", body)  # after each model, in case a later one dies
    return body


# bf16 can't be held to an exactness tolerance (eval/README.md): same thresholds as
# snapcall.decide's half-precision check.
HALF_MIN_SAME_ANSWER = 0.98
HALF_P95_GAP = 0.2


def cached_gate(
    body: dict, model_path: str, method: str = "cached", dtype: str = "float32"
) -> tuple[bool, str]:
    sec = body["models"][model_label(model_path)].get(dtype, {}).get(f"{method}_vs_separate", {})
    if sec.get("n_failed"):
        return False, f"{method} forward raised (see equivalence.json errors)"
    if not sec:
        return False, f"no {method} vs separate result in {dtype}"
    if dtype == "float32":
        ok = sec["prob_diff_max"] <= GATE_PROB_DIFF and sec["argmax_agree_rate"] == 1
        rule = f"max gap <= {GATE_PROB_DIFF}, every answer the same"
    else:
        ok = (
            sec["argmax_agree_rate"] >= HALF_MIN_SAME_ANSWER
            and sec["prob_diff_p95"] <= HALF_P95_GAP
        )
        rule = f">= {HALF_MIN_SAME_ANSWER:.0%} same answer, p95 gap <= {HALF_P95_GAP}"
    verdict = "matches" if ok else "differs from"
    return ok, (
        f"{method} {verdict} separate in {dtype} ({rule}): max {sec['prob_diff_max']:.2g}, "
        f"p95 {sec['prob_diff_p95']:.2g}, same answer {sec['argmax_agree_rate']:.1%}"
    )


# --------------------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------------------


def stage_eval(args, model, tok, records: list[DataRecord], out: Path) -> dict:
    system, run_id = args.system, args.run_id
    for r in records[:3]:  # warm-up: CUDA kernels, allocator
        decide_detailed(model, tok, r.to_request(debias="auto"), forward=args.forward)
    sync()
    preds: list[Prediction] = []
    forwards: Counter = Counter()
    in_tokens = compute_tokens = n_errors = 0
    t0 = time.perf_counter()
    for i, r in enumerate(records):
        req = r.to_request(debias="auto")
        meta = {"dataset": r.dataset, "n_questions": len(r.questions), "debias": "auto"}
        try:
            d = decide_detailed(model, tok, req, forward=args.forward, model_name=system)
        except Exception as e:
            n_errors += 1
            msg = f"{type(e).__name__}: {e}"[:300]
            preds += [
                Prediction(
                    system=system,
                    run_id=run_id,
                    state_id=r.state_id,
                    question_id=qid,
                    type=q.type,
                    answer=None,
                    probabilities=None,
                    error=msg,
                    meta=meta,
                )
                for qid, q in r.questions.items()
            ]
            continue
        forwards[d.forward] += 1
        in_tokens += d.response.usage.input_tokens
        compute_tokens += d.compute_tokens
        extra = {**meta, "forward": d.forward, "compute_tokens": d.compute_tokens}
        per_perm = {res.question_id: res.per_permutation for res in d.results}
        for p in predictions_from_response(
            d.response, system=system, state_id=r.state_id, run_id=run_id, meta=extra
        ):
            p.meta["per_permutation"] = per_perm[p.question_id]
            preds.append(p)
        if (i + 1) % 100 == 0:
            print(f"eval {i + 1}/{len(records)} {time.perf_counter() - t0:.0f}s", flush=True)
    sync()
    seconds = time.perf_counter() - t0

    write_jsonl(out / "predictions.jsonl", preds)
    timings = {
        "system": system,
        "eval_seconds": seconds,
        "n_states": len(records),
        "n_decisions": sum(len(r.questions) for r in records),
        "n_state_errors": n_errors,
        "input_tokens": in_tokens,
        "compute_tokens": compute_tokens,
        "forwards": dict(forwards),
        "dtype": args.dtype,
        "quant": getattr(args, "quant", None),  # train_lora passes its own args
        "device": device_info(),
    }
    write_json(out / "eval_timings.json", timings)
    for name, m in compute_all(preds, records, system=system, run_id=run_id).items():
        suffix = "" if name == "all" else f"_{name}"
        (out / f"metrics{suffix}.json").write_text(m.model_dump_json(indent=2) + "\n")
    return timings


# --------------------------------------------------------------------------------------
# Stage 3
# --------------------------------------------------------------------------------------


def with_n_questions(record: DataRecord, n: int) -> dict:
    """The record's questions, cycled (with suffixed ids) or cut to exactly ``n``."""
    items = list(record.questions.items())
    qs = {}
    for i in range(n):
        qid, q = items[i % len(items)]
        qs[qid if i < len(items) else f"{qid}_{i // len(items) + 1}"] = q
    return qs


def stage_latency(args, model, tok, records: list[DataRecord], out: Path) -> dict:
    from snapcall.schema import DecideRequest

    sample = [r for r in records if r.dataset == "customer-support-tickets"][: args.latency_states]
    rows = []
    for n_q in args.latency_n_questions:
        for debias in ("off", "auto"):
            for fwd in args.latency_forwards:
                reqs = [
                    DecideRequest(state=r.state, questions=with_n_questions(r, n_q), debias=debias)
                    for r in sample
                ]
                # Big enough that "packed" really is packed at 8 questions with debiasing.
                kw = {"forward": fwd, "packed_max_tokens": 16_384}
                decide_detailed(model, tok, reqs[0], **kw)  # warm-up for this shape
                for r, req in zip(sample, reqs, strict=True):
                    sync()
                    t0 = time.perf_counter()
                    d = decide_detailed(model, tok, req, **kw)
                    sync()
                    rows.append(
                        {
                            "state_id": r.state_id,
                            "n_questions": n_q,
                            "debias": debias,
                            "forward": fwd,
                            "forward_used": d.forward,
                            "latency_ms": (time.perf_counter() - t0) * 1000,
                            "input_tokens": d.response.usage.input_tokens,
                            "compute_tokens": d.compute_tokens,
                            "n_branches": sum(len(x.per_permutation) for x in d.results),
                        }
                    )
                print(f"latency q={n_q} debias={debias} {fwd}: done", flush=True)
    import json

    (out / "latency.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"n_rows": len(rows), "n_states": len(sample)}


# --------------------------------------------------------------------------------------
# Stage 4
# --------------------------------------------------------------------------------------


def stage_prior(args, model, tok, out: Path) -> dict:
    priors = {name: label_prior(model, tok, ls) for name, ls in CANDIDATE_LABELS.items()}
    body = {
        "model": model_label(args.model),
        "priors": priors,
        "flatness": {name: flatness(p) for name, p in priors.items()},
    }
    write_json(out / "label_prior.json", body)
    return body["flatness"]


# --------------------------------------------------------------------------------------


def default_system(model_path: str, quant: str | None = None) -> str:
    """``snapcall-<model>[-8bit]-zeroshot``: quantisation is part of the name everywhere."""
    return f"snapcall-{model_label(model_path)}{'-' + quant if quant else ''}-zeroshot"


def extra_data(text: str) -> tuple[str, str]:
    name, sep, uri = text.partition("=")
    if not sep or not name.isidentifier() or not uri:
        raise argparse.ArgumentTypeError(f"expected NAME=URI, got {text!r}")
    return name, uri


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m eval.run_eval")
    p.add_argument("--model", required=True, help="local path (runner: {models}/<org>/<name>)")
    p.add_argument("--data", required=True, help="test.jsonl, local or gs://")
    p.add_argument("--out", default="outputs")
    p.add_argument("--system", default=None, help="default: snapcall-<model>-zeroshot")
    p.add_argument("--run-id", default=None, help="default: $SNAPCALL_RUN_ID")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--forward", default="packed", choices=["packed", "cached", "separate"])
    p.add_argument("--limit", type=int, default=None, help="first N test states only")
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    p.add_argument("--equivalence-models", nargs="*", default=None)
    p.add_argument("--equivalence-dtypes", nargs="+", default=["bfloat16"])
    p.add_argument("--equivalence-states", type=int, default=30)
    p.add_argument("--equivalence-datasets", nargs="*", default=None)
    p.add_argument(
        "--quant",
        default=None,
        choices=["8bit"],
        help="bitsandbytes 8-bit weights; the system name gets -8bit",
    )
    p.add_argument(
        "--extra-data",
        type=extra_data,
        action="append",
        default=[],
        metavar="NAME=URI",
        help="also score this gold file in the eval stage, into OUT/NAME (e.g. ood=gs://...)",
    )
    p.add_argument("--gate-cached", action="store_true")
    p.add_argument(
        "--gate-method",
        default="cached",
        choices=["cached", "packed"],
        help="which fast path the gate checks against separate (packed: e.g. Phi-4-mini)",
    )
    p.add_argument(
        "--gate-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="dtype the gate judges (bfloat16 for models too big for float32 on one L4)",
    )
    p.add_argument("--latency-states", type=int, default=30)
    p.add_argument("--latency-n-questions", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument(
        "--latency-forwards",
        nargs="+",
        default=["packed", "cached", "separate"],
        choices=["packed", "cached", "separate"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    # Several models load one after another; don't let fragmentation cause an OOM.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args(argv)
    args.run_id = args.run_id or os.environ.get("SNAPCALL_RUN_ID")
    args.system = args.system or default_system(args.model, args.quant)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(local_path(args.data), DataRecord))
    if args.limit:
        records = round_robin(records, args.limit)

    stages: dict = {"args": vars(args), "stages": {}}

    def record(name, fn):
        t0 = time.perf_counter()
        try:
            result = fn()
            stages["stages"][name] = {"ok": True, "seconds": time.perf_counter() - t0}
            return result
        except Exception as e:
            traceback.print_exc()
            stages["stages"][name] = {
                "ok": False,
                "seconds": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {e}"[:1000],
            }
            return None
        finally:
            write_json(out / "stages.json", stages)

    if "equivalence" in args.stages:
        body = record("equivalence", lambda: stage_equivalence(args, records, out))
        if args.gate_cached:
            ok, why = (False, "equivalence failed")
            if body:
                ok, why = cached_gate(body, args.model, args.gate_method, args.gate_dtype)
            stages["cache_gate"] = {"ok": ok, "why": why}
            write_json(out / "stages.json", stages)
            print(f"cache-copy gate: {why}")
            if not ok:
                return 3

    rest = [s for s in args.stages if s != "equivalence"]
    if rest:
        t0 = time.perf_counter()
        model, tok = load_model(args.model, args.dtype, quant=args.quant)
        stages["load_seconds"] = time.perf_counter() - t0
        if "eval" in rest:
            record("eval", lambda: stage_eval(args, model, tok, records, out))
            for name, uri in args.extra_data:  # e.g. ood=gs://.../ood_test.jsonl -> out/ood/
                extra = list(read_jsonl(local_path(uri), DataRecord))
                (out / name).mkdir(parents=True, exist_ok=True)
                record(
                    f"eval:{name}",
                    lambda extra=extra, name=name: stage_eval(args, model, tok, extra, out / name),
                )
        if "latency" in rest:
            record("latency", lambda: stage_latency(args, model, tok, records, out))
        if "prior" in rest:
            record("prior", lambda: stage_prior(args, model, tok, out))

    failed = [k for k, v in stages["stages"].items() if not v["ok"]]
    print(f"stages: {stages['stages']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

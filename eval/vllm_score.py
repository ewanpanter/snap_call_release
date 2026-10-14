"""SnapCall decisions on vLLM: zero-shot scoring and latency for models our transformers path
can't run properly on one L4 (a 26B MoE in 4-bit).

SnapCall's three steps, with only the middle one swapped:

1. prompt (``snapcall.prompt``): the state as a shared prefix plus one branch per question
   and option order. Unchanged; the same token ids as ``decide``.
2. forward: vLLM instead of our transformers pass. Our ``cached`` method (the prefix once,
   every branch on top of its KV cache) is what vLLM's prefix caching does natively, with
   real kernels (Marlin int4 MoE for compressed-tensors w4a16 checkpoints on the L4). Each
   branch is a one-token request restricted to its answer labels (``allowed_token_ids``),
   and the label log-probabilities come back as ``processed_logprobs``, i.e. already
   normalised over the labels: exactly our restricted softmax.
3. read-out (``snapcall.readout.read_branches``): unchanged. The label log-probs are put
   in a logits row (everything else -inf), so the restricted softmax gives the same
   distribution, and permutation averaging and confidence work as before.

Latency per state is wall time for its branches. ``--prefix-first`` sends the prefix alone
first, so the branches hit a warm prefix cache (vLLM computes a prefix once per batch only
if it is already cached when the batch is scheduled); both ways are timed on a sample.

``--gate-ms``: stop after the probe sample if the one-order p50 is above it (e.g. only carry on
if it's about 250 ms or less).

vLLM isn't in the project's lockfile (it pins its own torch), so on the runner VM this
module installs it into a separate venv with uv and re-runs itself there.

    gh workflow run run-gpu.yml --ref <branch> -f script=eval.vllm_score -f args="\\
      --model {models}/cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 \\
      --data gs://<scratch>/data/cfpb_test.jsonl --limit 50 --gate-ms 250"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VLLM_VERSION = "0.30.0"


def ensure_vllm(argv: list[str]) -> None:
    """Import vllm, or install it into its own venv and re-exec this module there."""
    try:
        import vllm  # noqa: F401

        return
    except ImportError:
        pass
    if os.environ.get("SNAPCALL_VLLM_REEXEC"):
        raise SystemExit("vllm still isn't importable after installing it")
    venv = Path(os.environ.get("SNAPCALL_VLLM_VENV", Path(sys.prefix).parent / "vllm-venv"))
    t0 = time.perf_counter()
    if not (venv / "bin" / "python").exists():
        subprocess.run(["uv", "venv", "--python", "3.12", str(venv)], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), f"vllm=={VLLM_VERSION}"],
        check=True,
    )
    print(f"vllm {VLLM_VERSION} installed in {time.perf_counter() - t0:.0f}s", flush=True)
    env = {**os.environ, "SNAPCALL_VLLM_REEXEC": "1"}
    py = str(venv / "bin" / "python")
    os.execve(py, [py, "-m", "eval.vllm_score", *argv], env)


def pct(xs: list[float], p: float) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="python -m eval.vllm_score")
    p.add_argument(
        "--model", required=True, action="append", help="repeat for several (one process each)"
    )
    p.add_argument("--data", required=True, help="DataRecord jsonl, local or gs://")
    p.add_argument("--out", default="outputs")
    p.add_argument("--name", default=None, help="output folder (default: the model's name)")
    p.add_argument("--system", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--debias",
        nargs="+",
        default=["auto", "off"],
        choices=["auto", "off"],
        help="scored mode first (predictions.jsonl); every mode is timed",
    )
    p.add_argument("--timing-states", type=int, default=50, help="states timed per mode/setup")
    p.add_argument("--gate-ms", type=float, default=None, help="stop if one-order p50 is above")
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    a = p.parse_args(argv)
    ensure_vllm(argv)  # once, before any per-model process
    if len(a.model) > 1:  # one process per model: vLLM doesn't reliably free the GPU in-process
        rc = 0
        for m in a.model:
            rest = [
                x for i, x in enumerate(argv) if not (x == "--model" or argv[i - 1] == "--model")
            ]
            rc |= subprocess.run(
                [sys.executable, "-m", "eval.vllm_score", "--model", m, *rest]
            ).returncode
        return rc
    a.model = a.model[0]

    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    from eval.common import local_path, model_label, write_json
    from eval.metrics import compute_all
    from snapcall.prompt import DEFAULT_LABELS, build_prompt, encode_prompt, profile_for
    from snapcall.readout import read_branches, to_answer
    from snapcall.schema import (
        DataRecord,
        DecideResponse,
        Usage,
        predictions_from_response,
        read_jsonl,
        write_jsonl,
    )

    label = model_label(a.model)
    out = Path(a.out) / (a.name or label)
    out.mkdir(parents=True, exist_ok=True)
    system = a.system or f"snapcall-{label}-vllm-zeroshot"
    records = list(read_jsonl(local_path(a.data), DataRecord))[: a.limit]

    t0 = time.perf_counter()
    kwargs = dict(
        model=a.model,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},  # text only: skip encoders
        max_logprobs=32,
    )
    logprobs_mode = "processed_logprobs"
    try:
        llm = LLM(**kwargs, logprobs_mode=logprobs_mode)
    except (TypeError, ValueError) as e:  # older/newer API: fall back to raw top-k
        print(f"logprobs_mode unsupported ({e}); using raw logprobs", flush=True)
        logprobs_mode = "raw_logprobs"
        llm = LLM(**kwargs)
    load_s = time.perf_counter() - t0
    tok = llm.get_tokenizer()
    profile = profile_for(tok)
    print(f"{a.model}: loaded in {load_s:.0f}s, vllm {vllm.__version__}", flush=True)

    missing_labels = 0

    def run_state(record, debias: str, prefix_first: bool):
        """(results, request, encoded, ms) for one state."""
        nonlocal missing_labels
        req = record.to_request(debias=debias)
        prompt = build_prompt(req, DEFAULT_LABELS, profile)
        enc = encode_prompt(prompt, tok)
        prompts = [TokensPrompt(prompt_token_ids=enc.prefix_ids + b) for b in enc.branch_ids]
        params = [
            SamplingParams(
                max_tokens=1,
                temperature=1.0,
                logprobs=len(br.keys),
                allowed_token_ids=enc.label_ids[: len(br.keys)],
            )
            for br in prompt.branches
        ]
        start = time.perf_counter()
        if prefix_first:
            llm.generate(
                [TokensPrompt(prompt_token_ids=enc.prefix_ids)],
                SamplingParams(max_tokens=1),
                use_tqdm=False,
            )
        outs = llm.generate(prompts, params, use_tqdm=False)
        ms = (time.perf_counter() - start) * 1000
        width = max(enc.label_ids) + 1
        logits = torch.full((len(outs), width), float("-inf"))
        for i, (o, br) in enumerate(zip(outs, prompt.branches, strict=True)):
            got = o.outputs[0].logprobs[0]
            ids = enc.label_ids[: len(br.keys)]
            floor = min(lp.logprob for lp in got.values()) - 5.0
            for t in ids:
                if t in got:
                    logits[i, t] = got[t].logprob
                else:  # raw top-k missed a label: far below the lowest returned
                    logits[i, t] = floor
                    missing_labels += 1
        results = read_branches(req, prompt.branches, logits, enc.label_ids)
        return results, req, enc, ms

    # warm-up: CUDA graphs, allocator, the first prefix-cache blocks
    for r in records[:3]:
        run_state(r, a.debias[0], False)

    # 1. latency: each mode, with and without prefix-first, on the same sample
    sample = records[3 : 3 + a.timing_states] or records
    timing: dict = {}
    for debias in a.debias:
        for prefix_first in (False, True):
            ms, tokens = [], []
            for r in sample:
                _, _, enc, t = run_state(r, debias, prefix_first)
                ms.append(t)
                tokens.append(enc.n_tokens)
            key = f"{debias}{'-prefix-first' if prefix_first else ''}"
            timing[key] = {
                "n": len(ms),
                "p50_ms": pct(ms, 0.5),
                "p95_ms": pct(ms, 0.95),
                "mean_ms": sum(ms) / len(ms),
                "tokens_p50": pct(tokens, 0.5),
            }
            print(f"latency {key}: {timing[key]}", flush=True)
    best_off = min((v["p50_ms"] for k, v in timing.items() if k.startswith("off")), default=None)
    summary = {
        "model": a.model,
        "system": system,
        "vllm": vllm.__version__,
        "logprobs_mode": logprobs_mode,
        "load_seconds": load_s,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timing": timing,
        "one_order_p50_ms": best_off,
        "gate_ms": a.gate_ms,
    }
    write_json(out / "vllm.json", summary)
    if a.gate_ms is not None and best_off is not None and best_off > a.gate_ms:
        summary["gate"] = "FAIL"
        write_json(out / "vllm.json", summary)
        print(f"GATE FAIL: one-order p50 {best_off:.0f} ms > {a.gate_ms:.0f} ms", flush=True)
        return 0
    summary["gate"] = "pass" if a.gate_ms is not None else None

    # 2. score every state in the first mode (the sweep's "auto" by default)
    preds, n_errors = [], 0
    t1 = time.perf_counter()
    for i, r in enumerate(records):
        try:
            results, req, enc, ms = run_state(r, a.debias[0], False)
            response = DecideResponse(
                model=system,
                answers={
                    x.question_id: to_answer(req.questions[x.question_id], x) for x in results
                },
                usage=Usage(input_tokens=enc.n_tokens, output_tokens=0),
                latency_ms=ms,
            )
            per_perm = {x.question_id: x.per_permutation for x in results}
            meta = {"dataset": r.dataset, "debias": a.debias[0], "engine": "vllm"}
            for pr in predictions_from_response(
                response, system=system, state_id=r.state_id, run_id=os.environ.get(
                    "SNAPCALL_RUN_ID"), meta=meta
            ):  # fmt: skip
                pr.meta["per_permutation"] = per_perm[pr.question_id]
                preds.append(pr)
        except Exception as e:  # noqa: BLE001 - one bad state must not lose the run
            n_errors += 1
            print(f"{r.state_id}: {type(e).__name__}: {e}"[:300], flush=True)
        if (i + 1) % 100 == 0:
            print(f"scored {i + 1}/{len(records)} {time.perf_counter() - t1:.0f}s", flush=True)
    write_jsonl(out / "predictions.jsonl", preds)
    summary.update(
        n_states=len(records),
        n_state_errors=n_errors,
        score_seconds=time.perf_counter() - t1,
        missing_label_logprobs=missing_labels,
    )
    write_json(out / "vllm.json", summary)
    for name, m in compute_all(preds, records, system=system).items():
        suffix = "" if name == "all" else f"_{name}"
        (out / f"metrics{suffix}.json").write_text(m.model_dump_json(indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("one_order_p50_ms", "gate", "n_state_errors")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

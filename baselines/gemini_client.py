"""Gemini on Vertex baseline: the "just ask a general LLM" approach.

One structured-output call per test state answers all its questions at once. The JSON
schema forces, per question, an ``answer`` from the allowed keys and a stated
``probabilities`` object over all of them (the model's own, verbalised numbers, which is
what you get from a general LLM). If the model exposes logprobs, the chosen answer token's
log-probability and its top alternatives are also kept in ``meta``, so the report can set
the stated numbers against the token-level ones. ``meta.prob_source`` says which applies.

Auth is Vertex IAM via ADC (Experiment tier, no API key).

    uv run --group baselines python -m baselines.gemini_client \\
        --project P --region R --data test.jsonl --run-id ID --out outputs/ [--model M]

If the model rejects logprobs or ``thinking_budget=0``, the client drops logprobs or falls
back to ``thinking_level=minimal`` once for the rest of the run and records that in
``summary.json`` (``config_fallbacks``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
from typing import Any

from baselines.common import (
    DEFAULT_USD_TO_GBP,
    RetryableError,
    StateResult,
    error_predictions,
    load_records,
    make_cost,
    run_all,
    summarise,
    truncate,
    with_permutations,
    with_retries,
    write_outputs,
)
from snapcall.schema import (
    ChoiceQuestion,
    DataRecord,
    Prediction,
    ScoreQuestion,
    YesNoQuestion,
    state_text,
)

DEFAULT_MODEL = "gemini-2.5-flash"
# USD per 1M tokens (input, output). Thinking tokens bill at the output rate. Only models
# whose price has been checked go here; for anything else pass --price-in/--price-out, or
# cost.json reports tokens with cost 0 and says so.
PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
}
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
SUM_TOL = 0.02
TOP_LOGPROBS = 5

SYSTEM_INSTRUCTION = (
    "You are a decision service. Read the STATE, then answer every question about it "
    'independently. For each question, set "answer" to exactly one of the allowed keys, and '
    'set "probabilities" to your probability for every allowed key: numbers between 0 and 1 '
    "that sum to 1. Be calibrated: when you are unsure, spread the probability."
)


# --------------------------------------------------------------------------------------
# Prompt and schema
# --------------------------------------------------------------------------------------


def answer_space(q: Any) -> list[str]:
    """The keys the model answers with. Score levels are answered by their own text when
    that is unambiguous (reads better to a general LLM), else by index."""
    if isinstance(q, ChoiceQuestion):
        return list(q.options)
    if isinstance(q, ScoreQuestion):
        levels = [lvl.strip() for lvl in q.levels]
        simple = len(set(levels)) == len(levels) and all(0 < len(x) <= 64 for x in levels)
        return levels if simple else [str(i) for i in range(len(levels))]
    return ["yes", "no"]


def build_prompt(record: DataRecord) -> str:
    lines = ["STATE:", "<<<", state_text(record.state), ">>>", "", "QUESTIONS:"]
    for qid, q in record.questions.items():
        keys = answer_space(q)
        if isinstance(q, ChoiceQuestion):
            lines.append(f"[{qid}] (pick one) {q.question}")
            lines += [f"  - {k}: {desc}" for k, desc in q.options.items()]
        elif isinstance(q, ScoreQuestion):
            lines.append(f"[{qid}] (ordered scale, lowest first) {q.question}")
            lines += [
                f"  - {k}" + ("" if k == lvl else f": {lvl}")
                for k, lvl in zip(keys, q.levels, strict=True)
            ]
        else:
            lines.append(f"[{qid}] (is this statement true?) {q.question}")
            lines.append(f"  - yes{': ' + q.yes if q.yes else ''}")
            lines.append(f"  - no{': ' + q.no if q.no else ''}")
    return "\n".join(lines)


def build_schema(record: DataRecord) -> dict[str, Any]:
    """JSON Schema for the response: {qid: {answer: enum, probabilities: {key: number}}}."""
    props: dict[str, Any] = {}
    for qid, q in record.questions.items():
        keys = answer_space(q)
        props[qid] = {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": keys},
                "probabilities": {
                    "type": "object",
                    "properties": {k: {"type": "number"} for k in keys},
                    "required": keys,
                },
            },
            "required": ["answer", "probabilities"],
        }
    return {"type": "object", "properties": props, "required": list(record.questions)}


# --------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------


def _answer_token_meta(text: str, logprobs: Any, qid: str) -> dict[str, Any]:
    """Find the token where question ``qid``'s answer value starts, and return its
    probability and top alternatives. Best effort: {} if it can't be located."""
    chosen = getattr(logprobs, "chosen_candidates", None) or []
    if not chosen:
        return {}
    m = re.search(r'"' + re.escape(qid) + r'"\s*:\s*\{\s*"answer"\s*:\s*"', text)
    if not m:
        return {}
    target = m.end()  # first character of the answer value
    pos = 0
    top = getattr(logprobs, "top_candidates", None) or []
    for i, c in enumerate(chosen):
        tok = getattr(c, "token", "") or ""
        if pos + len(tok) > target:
            alts = {}
            if i < len(top):
                for alt in getattr(top[i], "candidates", None) or []:
                    alts[getattr(alt, "token", "")] = round(math.exp(alt.log_probability), 6)
            return {
                "answer_token": tok,
                "answer_token_prob": round(math.exp(c.log_probability), 6),
                "answer_token_top": alts,
            }
        pos += len(tok)
    return {}


def _clean_probs(raw: Any, keys: list[str], answer: str) -> tuple[dict[str, float], list[str]]:
    """Stated probabilities -> a proper distribution over ``keys``, plus what was wrong."""
    kinds: list[str] = []
    raw = raw if isinstance(raw, dict) else {}
    vals = {}
    for k in keys:
        v = raw.get(k)
        vals[k] = min(max(float(v), 0.0), 1.0) if isinstance(v, int | float) else 0.0
    if set(raw) - set(keys):
        kinds.append(f"prob_keys_extra:{sorted(set(raw) - set(keys))[:5]}")
    total = sum(vals.values())
    if total <= 0:
        kinds.append("no_probabilities")
        return {k: float(k == answer) for k in keys}, kinds
    if abs(total - 1) > SUM_TOL:
        kinds.append(f"prob_sum:{total:.3f}")
    probs = {k: v / total for k, v in vals.items()}
    if probs[answer] < max(probs.values()) - 1e-9:
        kinds.append(f"answer_not_argmax:{answer}={probs[answer]:.3f}")
    return probs, kinds


def parse_response(
    record: DataRecord,
    text: str,
    *,
    system: str,
    run_id: str | None,
    latency_ms: float | None,
    logprobs: Any = None,
    base_meta: dict[str, Any] | None = None,
) -> tuple[list[Prediction], dict[str, list[str]]]:
    """Model JSON -> one Prediction per question. A bad answer fails only its question."""
    data = json.loads(text)  # a non-JSON body fails the whole state
    preds: list[Prediction] = []
    anomalies: dict[str, list[str]] = {}
    for qid, q in record.questions.items():
        keys = answer_space(q)
        meta = {
            "dataset": record.dataset,
            "n_questions": len(record.questions),
            "prob_source": "verbalised" + ("+logprobs" if logprobs else ""),
            **(base_meta or {}),
        }
        common = dict(
            system=system,
            run_id=run_id,
            state_id=record.state_id,
            question_id=qid,
            type=q.type,
            latency_ms=latency_ms,
        )
        a = data.get(qid) if isinstance(data, dict) else None
        answer = a.get("answer") if isinstance(a, dict) else None
        if answer not in keys:
            anomalies[qid] = [f"bad_answer:{answer!r}"]
            preds.append(
                Prediction(
                    answer=None,
                    probabilities=None,
                    error=truncate(f"answer {answer!r} not in allowed keys"),
                    meta={**meta, "anomalies": anomalies[qid]},
                    **common,
                )
            )
            continue
        probs, kinds = _clean_probs(a.get("probabilities"), keys, answer)
        if kinds:
            anomalies[qid] = kinds
        meta["anomalies"] = kinds
        if logprobs:
            meta.update(_answer_token_meta(text, logprobs, qid))
        if isinstance(q, ChoiceQuestion):
            preds.append(Prediction(answer=answer, probabilities=probs, meta=meta, **common))
        elif isinstance(q, ScoreQuestion):
            by_index = {str(i): probs[k] for i, k in enumerate(keys)}
            value = sum(i * p for i, p in enumerate(by_index.values()))
            preds.append(
                Prediction(
                    answer=keys.index(answer),
                    probabilities=by_index,
                    value=min(value, len(keys) - 1),
                    meta=meta,
                    **common,
                )
            )
        else:
            assert isinstance(q, YesNoQuestion)
            preds.append(
                Prediction(answer=answer == "yes", probabilities=probs, meta=meta, **common)
            )
    return preds, anomalies


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------


def _error_code(e: Exception) -> int | None:
    code = getattr(e, "code", None)
    return code if isinstance(code, int) else None


class GeminiClient:
    """Calls ``client.models.generate_content``; the SDK client is injected (tests use a fake).

    Config is a plain dict (the SDK accepts one), so this module never imports the SDK
    outside ``make_genai_client``.
    """

    def __init__(
        self,
        genai_client: Any,
        *,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        run_id: str | None = None,
        logprobs: bool = True,
        thinking_budget: int | None = 0,
        thinking_level: str | None = None,
        max_output_tokens: int = 4096,
        max_attempts: int = 5,
        sleep=time.sleep,
    ):
        self._client = genai_client
        self.model = model
        self.system = system or model
        self.run_id = run_id
        self.logprobs = logprobs
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._lock = threading.Lock()
        self.fallbacks: list[str] = []

    def config(self, record: DataRecord) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": 0.0,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
            "response_json_schema": build_schema(record),
        }
        with self._lock:
            if self.logprobs:
                cfg["response_logprobs"] = True
                cfg["logprobs"] = TOP_LOGPROBS
            if self.thinking_level:
                cfg["thinking_config"] = {"thinking_level": self.thinking_level}
            elif self.thinking_budget is not None:
                cfg["thinking_config"] = {"thinking_budget": self.thinking_budget}
        return cfg

    def _fallback(self, e: Exception) -> bool:
        """Adapt once to a model that rejects a feature. True if worth trying again."""
        msg = str(e).lower()
        with self._lock:
            if "logprob" in msg and self.logprobs:
                self.logprobs = False
                self.fallbacks.append(f"logprobs disabled: {truncate(str(e), 160)}")
                return True
            if "thinking" in msg and not self.thinking_level:
                self.thinking_level = "MINIMAL"
                self.fallbacks.append(f"thinking_level=MINIMAL: {truncate(str(e), 160)}")
                return True
            # Another thread may already have switched the feature off.
            return ("logprob" in msg and not self.logprobs) or (
                "thinking" in msg and bool(self.thinking_level)
            )

    def _call(self, record: DataRecord) -> tuple[Any, float]:
        prompt = build_prompt(record)
        for _ in range(3):  # at most two feature fallbacks
            cfg = self.config(record)
            t0 = time.perf_counter()
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                return resp, (time.perf_counter() - t0) * 1000
            except Exception as e:
                code = _error_code(e)
                if code in RETRYABLE_CODES or (code is None and _looks_transient(e)):
                    msg = f"{code or type(e).__name__}: {truncate(str(e), 200)}"
                    raise RetryableError(msg) from None
                if code == 400 and self._fallback(e):
                    continue
                raise
        raise RuntimeError("model kept rejecting the request config")

    def decide(self, record: DataRecord) -> StateResult:
        (resp, latency), attempts = with_retries(
            lambda: self._call(record), max_attempts=self.max_attempts, sleep=self._sleep
        )
        usage = getattr(resp, "usage_metadata", None)
        tokens = dict(
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            thinking_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
        )
        version = getattr(resp, "model_version", None)
        cand = (getattr(resp, "candidates", None) or [None])[0]
        finish = getattr(getattr(cand, "finish_reason", None), "value", None) or str(
            getattr(cand, "finish_reason", None)
        )
        text = getattr(resp, "text", None) or ""
        raw = {"model_version": version, "finish_reason": finish, "text": text, **tokens}
        base = dict(
            state_id=record.state_id,
            latency_ms=latency,
            model_version=version,
            attempts=attempts,
            raw=raw,
            **tokens,
        )
        if finish not in ("STOP", "FinishReason.STOP") or not text:
            err = f"finish_reason={finish}, {len(text)} chars"
            preds = error_predictions(record, system=self.system, run_id=self.run_id, error=err)
            return StateResult(predictions=preds, error=err, **base)
        try:
            preds, anomalies = parse_response(
                record,
                text,
                system=self.system,
                run_id=self.run_id,
                latency_ms=latency,
                logprobs=getattr(cand, "logprobs_result", None),
                base_meta={"model": version},
            )
        except (json.JSONDecodeError, AttributeError) as e:
            err = f"unparseable response: {type(e).__name__}: {e}"
            preds = error_predictions(record, system=self.system, run_id=self.run_id, error=err)
            return StateResult(predictions=preds, error=truncate(err), **base)
        return StateResult(predictions=preds, anomalies=anomalies, **base)

    def failed(self, record: DataRecord, error: str) -> StateResult:
        preds = error_predictions(record, system=self.system, run_id=self.run_id, error=error)
        return StateResult(state_id=record.state_id, predictions=preds, error=truncate(error))


def _looks_transient(e: Exception) -> bool:
    name = type(e).__name__.lower()
    return any(s in name for s in ("timeout", "connect", "remoteprotocol", "readerror"))


def make_genai_client(project: str, location: str, timeout_s: float) -> Any:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("google-genai is missing: run with `uv run --group baselines ...`")
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )


def total_usd(results: list[StateResult], price: tuple[float, float] | None) -> float:
    if price is None:
        return 0.0
    p_in, p_out = price
    tin = sum(r.input_tokens for r in results)
    tout = sum(r.output_tokens + r.thinking_tokens for r in results)
    return tin / 1e6 * p_in + tout / 1e6 * p_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--project", required=True)
    p.add_argument("--region", required=True, help="Vertex location (e.g. us-central1)")
    p.add_argument("--location", default=None, help="overrides --region (e.g. global)")
    p.add_argument("--data", required=True, help="DataRecord JSONL (e.g. test.jsonl)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--system", default=None, help="Prediction.system label (default: model)")
    p.add_argument("--limit", type=int, default=None, help="first N states, across datasets")
    p.add_argument(
        "--permutations",
        type=int,
        default=1,
        metavar="K",
        help="ask each choice question in K option orders (SnapCall's cyclic shifts) and "
        "average; the system name gets -permK",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=90.0, help="seconds per request")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--no-logprobs", action="store_true")
    p.add_argument("--thinking-budget", type=int, default=0, help="-1 = model default")
    p.add_argument("--thinking-level", default=None, help="MINIMAL/LOW/...; overrides budget")
    p.add_argument("--max-output-tokens", type=int, default=4096)
    p.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    p.add_argument("--usd-to-gbp", type=float, default=DEFAULT_USD_TO_GBP)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    location = args.location or args.region
    if args.permutations > 1:
        args.system = f"{args.system or args.model}-perm{args.permutations}"
    client = GeminiClient(
        make_genai_client(args.project, location, args.timeout),
        model=args.model,
        system=args.system,
        run_id=args.run_id,
        logprobs=not args.no_logprobs,
        thinking_budget=None if args.thinking_budget < 0 else args.thinking_budget,
        thinking_level=args.thinking_level,
        max_output_tokens=args.max_output_tokens,
        max_attempts=args.max_attempts,
    )
    records = load_records(args.data, args.limit)
    print(f"{args.model} @ {location}: {len(records)} states, concurrency {args.concurrency}")
    t0 = time.perf_counter()
    decide = with_permutations(client.decide, args.permutations)
    results = run_all(records, decide, client.failed, concurrency=args.concurrency)
    wall_s = time.perf_counter() - t0

    price = PRICES_USD_PER_M.get(args.model)
    if args.price_in is not None and args.price_out is not None:
        price = (args.price_in, args.price_out)
    usd = total_usd(results, price)
    note = (
        f"Estimated from token counts at ${price[0]}/M in, ${price[1]}/M out (thinking as out)."
        if price
        else f"No checked price for {args.model}: cost is 0, tokens are real. Pass --price-*."
    )
    cost = make_cost(
        run_id=args.run_id,
        job="baseline-gemini.yml:baselines.gemini_client",
        system=client.system,
        provider="vertex-gemini",
        results=results,
        usd=usd,
        usd_to_gbp=args.usd_to_gbp,
        notes=note,
    )
    summary = summarise(
        results,
        {
            "system": client.system,
            "model_requested": args.model,
            "location": location,
            "concurrency": args.concurrency,
            "permutations": args.permutations,
            "wall_seconds": round(wall_s, 1),
            "logprobs_final": client.logprobs,
            "thinking_final": client.thinking_level or client.thinking_budget,
            "config_fallbacks": client.fallbacks,
            "price_usd_per_m": price,
            "usd": usd,
        },
    )
    write_outputs(args.out, results, cost, summary)
    print(json.dumps({k: summary[k] for k in ("n_states", "n_states_failed", "config_fallbacks")}))
    return 1 if summary["n_states_failed"] == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())

"""TypeSafe Jev baseline: send each test state with all its questions to the System One API.

Release tier only (the workflow reads the key from Secret Manager). The key comes from the
``TYPESAFE_API_KEY`` env var and is only ever put in the Authorization header: never
printed, logged, or written to outputs.

    uv run python -m baselines.typesafe_client --data test.jsonl --run-id ID --out outputs/

Endpoint: TypeSafe direct, ``POST https://api.typesafe.ai/v1/systemone`` with a Bearer key
(same body as OpenRouter's Decisions API; ``--endpoint``/``--model`` switch to OpenRouter).

Besides the contract outputs, every response is validated (probabilities that sum to 1, a
choice that is the most probable option, a score at an expected level, ...). Anything that
fails is flagged in each prediction's ``meta.anomalies`` and counted in ``summary.json``, so
malformed responses are visible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from baselines.common import (
    DEFAULT_USD_TO_GBP,
    FatalError,
    RetryableError,
    StateResult,
    error_predictions,
    load_records,
    make_cost,
    merge_repeats,
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
    DecideRequest,
    Prediction,
    ScoreQuestion,
    answer_keys,
    from_typesafe_response,
    predictions_from_response,
    to_typesafe_request,
)

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"  # pinned, not jev-latest, so reruns are comparable
DEFAULT_SYSTEM = "typesafe-jev"
# List price per 1M input tokens (output tokens are free). Only used when a response has
# no usage.cost of its own.
PRICE_USD_PER_M_INPUT = 0.042

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401, 402, 403}
TS_TYPE = {"choice": "choice", "score": "score", "yesno": "noul"}

# Jev rounds probabilities to two decimals, so small sum errors are normal.
SUM_TOL = 0.02
SCORE_TOL = 0.05


def _retry_after(headers: Any) -> float | None:
    try:
        return float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def post_json(endpoint: str, body: dict, api_key: str, timeout: float = 60.0) -> dict:
    """POST ``body``; map failures to Retryable/Fatal/ValueError. Never echoes the key."""
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "snapcall-baseline/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = truncate(e.read().decode("utf-8", "replace"), 200)
        msg = f"HTTP {e.code}: {detail}"
        if e.code in FATAL_STATUS:
            raise FatalError(msg) from None
        if e.code in RETRYABLE_STATUS:
            raise RetryableError(msg, _retry_after(e.headers)) from None
        raise ValueError(msg) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RetryableError(f"network: {getattr(e, 'reason', e)}") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        text = truncate(raw.decode("utf-8", "replace"))
        raise RetryableError(f"non-JSON response: {text}") from None
    if not isinstance(data, dict) or "answers" not in data:
        raise ValueError(f"unexpected response shape: {truncate(json.dumps(data), 200)}")
    return data


def check_answer(q: Any, a: Any) -> list[str]:
    """Anything odd about one raw Jev answer, as short ``kind[:detail]`` strings."""
    if not isinstance(a, dict):
        return ["missing_answer"]
    out = []
    if a.get("type") not in (None, TS_TYPE[q.type]):
        out.append(f"type_mismatch:{a.get('type')}")
    if q.type == "yesno":
        p = a.get("noul")
        if not isinstance(p, int | float) or not 0 <= p <= 1:
            out.append(f"noul_out_of_range:{p!r}")
        return out

    probs = a.get("probabilities") or {}
    expected = answer_keys(q)
    extra = sorted(set(probs) - set(expected))
    missing = sorted(set(expected) - set(probs))
    if extra:
        out.append(f"prob_keys_extra:{extra[:5]}")
    if missing:
        out.append(f"prob_keys_missing:{missing[:5]}")
    vals = {k: float(v) for k, v in probs.items() if isinstance(v, int | float)}
    total = sum(vals.values())
    if abs(total - 1) > SUM_TOL:
        out.append(f"prob_sum:{total:.3f}")
    if a.get("confidence") is None:
        out.append("no_confidence")
    elif not 0 <= a["confidence"] <= 1:
        out.append(f"confidence_out_of_range:{a['confidence']}")
    top = max(vals.values(), default=0.0)

    if isinstance(q, ChoiceQuestion):
        choice = a.get("choice")
        if choice not in q.options:
            out.append(f"choice_not_an_option:{choice!r}")
        elif vals.get(choice, 0.0) < top - 1e-9:
            out.append(f"choice_not_argmax:{choice}={vals.get(choice, 0.0)} max={top}")
    elif isinstance(q, ScoreQuestion):
        score = a.get("score")
        n = len(q.levels)
        if not isinstance(score, int | float):
            out.append(f"score_missing:{score!r}")
        else:
            if not 0 <= score <= n - 1:
                out.append(f"score_out_of_range:{score}")
            if total > 0:
                ev = sum(int(k) * v for k, v in vals.items() if k.isdigit()) / total
                if abs(score - ev) > SCORE_TOL:
                    out.append(f"score_not_expectation:{score:.3f} vs {ev:.3f}")
        legend = a.get("legend")
        if legend is not None and legend != {str(i): lvl for i, lvl in enumerate(q.levels)}:
            out.append("legend_mismatch")
    return out


def _question_meta(q: Any, a: Any) -> dict[str, Any]:
    """Raw numbers the report may want (before the adapter renormalises)."""
    if not isinstance(a, dict):
        return {}
    if q.type == "yesno":
        return {"raw_noul": a.get("noul")}
    probs = a.get("probabilities") or {}
    vals = [float(v) for v in probs.values() if isinstance(v, int | float)]
    meta: dict[str, Any] = {
        "raw_prob_sum": round(sum(vals), 6),
        "raw_max_prob": max(vals, default=None),
        "raw_confidence": a.get("confidence"),
    }
    if q.type == "score":
        meta["raw_score"] = a.get("score")
    return meta


def convert(
    record: DataRecord, body: dict, *, system: str, run_id: str | None, latency_ms: float | None
) -> tuple[list[Prediction], dict[str, list[str]]]:
    """Jev response -> one Prediction per question. A bad answer fails only its question."""
    preds: list[Prediction] = []
    anomalies: dict[str, list[str]] = {}
    answers = body.get("answers") or {}
    base_meta = {
        "dataset": record.dataset,
        "n_questions": len(record.questions),
        "model": body.get("model"),
        "response_id": body.get("id"),
    }
    for qid, q in record.questions.items():
        a = answers.get(qid)
        kinds = check_answer(q, a)
        if kinds:
            anomalies[qid] = kinds
        meta = {**base_meta, **_question_meta(q, a), "anomalies": kinds}
        sub_request = DecideRequest(state=record.state, questions={qid: q}, debias="off")
        sub_body = {"model": body.get("model"), "answers": {qid: a}}
        try:
            response = from_typesafe_response(sub_body, sub_request, latency_ms=latency_ms)
            preds += predictions_from_response(
                response, system=system, state_id=record.state_id, run_id=run_id, meta=meta
            )
        except (KeyError, TypeError, ValueError) as e:
            preds.append(
                Prediction(
                    system=system,
                    run_id=run_id,
                    state_id=record.state_id,
                    question_id=qid,
                    type=q.type,
                    answer=None,
                    probabilities=None,
                    latency_ms=latency_ms,
                    error=truncate(f"bad answer: {type(e).__name__}: {e}"),
                    meta=meta,
                )
            )
    return preds, anomalies


class TypeSafeClient:
    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        system: str = DEFAULT_SYSTEM,
        run_id: str | None = None,
        timeout: float = 60.0,
        max_attempts: int = 5,
        post=post_json,
        sleep=time.sleep,
    ):
        self._api_key = api_key
        self.endpoint, self.model, self.system, self.run_id = endpoint, model, system, run_id
        self.timeout, self.max_attempts = timeout, max_attempts
        self._post, self._sleep = post, sleep

    def __repr__(self) -> str:  # never show the key
        return f"TypeSafeClient(endpoint={self.endpoint!r}, model={self.model!r})"

    def decide(self, record: DataRecord) -> StateResult:
        body = to_typesafe_request(record.to_request(debias="off"), model=self.model)
        timing: dict[str, float] = {}

        def attempt() -> dict:
            t0 = time.perf_counter()
            data = self._post(self.endpoint, body, self._api_key, self.timeout)
            timing["ms"] = (time.perf_counter() - t0) * 1000
            return data

        data, attempts = with_retries(attempt, max_attempts=self.max_attempts, sleep=self._sleep)
        latency = timing["ms"]
        preds, anomalies = convert(
            record, data, system=self.system, run_id=self.run_id, latency_ms=latency
        )
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        return StateResult(
            state_id=record.state_id,
            predictions=preds,
            latency_ms=latency,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            usd_cost=float(cost) if isinstance(cost, int | float) else None,
            model_version=data.get("model"),
            attempts=attempts,
            anomalies=anomalies,
            raw=data,
        )

    def failed(self, record: DataRecord, error: str) -> StateResult:
        preds = error_predictions(record, system=self.system, run_id=self.run_id, error=error)
        return StateResult(state_id=record.state_id, predictions=preds, error=truncate(error))


def total_usd(results: list[StateResult]) -> tuple[float, str]:
    """Provider-reported cost where given, list-price estimate for the rest."""
    reported = [r.usd_cost for r in results if r.usd_cost is not None]
    estimated = [
        r.input_tokens / 1e6 * PRICE_USD_PER_M_INPUT
        for r in results
        if r.usd_cost is None and r.error is None
    ]
    note = (
        f"{len(reported)} responses with usage.cost; {len(estimated)} estimated at "
        f"${PRICE_USD_PER_M_INPUT}/M input tokens."
    )
    return sum(reported) + sum(estimated), note


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="DataRecord JSONL (e.g. test.jsonl)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="Prediction.system label")
    p.add_argument("--limit", type=int, default=None, help="first N states, across datasets")
    p.add_argument(
        "--permutations",
        type=int,
        default=1,
        metavar="K",
        help="ask each choice question in K option orders (SnapCall's cyclic shifts) and "
        "average; the system name gets -permK",
    )
    p.add_argument(
        "--repeat-same-order",
        type=int,
        default=1,
        metavar="N",
        help="determinism check: send every state N times, in N full passes over the file, "
        "always in the original option order (pass 2 starts after pass 1 ends). Pass 1 is "
        "scored; later passes go in meta.repeats and summary.json; the system name gets -repN",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--usd-to-gbp", type=float, default=DEFAULT_USD_TO_GBP)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not api_key.strip():
        print("TYPESAFE_API_KEY is not set (the workflow injects it)", file=sys.stderr)
        return 2
    if args.permutations > 1 and args.repeat_same_order > 1:
        print("--permutations and --repeat-same-order don't mix", file=sys.stderr)
        return 2
    if args.permutations > 1:
        args.system = f"{args.system}-perm{args.permutations}"
    if args.repeat_same_order > 1:
        args.system = f"{args.system}-rep{args.repeat_same_order}"
    client = TypeSafeClient(
        api_key.strip(),
        endpoint=args.endpoint,
        model=args.model,
        system=args.system,
        run_id=args.run_id,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
    )
    records = load_records(args.data, args.limit)
    print(f"{client!r}: {len(records)} states, concurrency {args.concurrency}", flush=True)
    t0 = time.perf_counter()
    decide = with_permutations(client.decide, args.permutations)
    passes = []
    for i in range(max(1, args.repeat_same_order)):
        if args.repeat_same_order > 1:
            print(f"pass {i + 1} of {args.repeat_same_order}", flush=True)
        passes.append(run_all(records, decide, client.failed, concurrency=args.concurrency))
    repeats = None
    results = passes[0]
    if len(passes) > 1:
        results, repeats = merge_repeats(passes)
    wall_s = time.perf_counter() - t0

    usd, cost_note = total_usd(results)
    cost = make_cost(
        run_id=args.run_id,
        job="baseline-typesafe.yml:baselines.typesafe_client",
        system=args.system,
        provider="typesafe",
        results=results,
        usd=usd,
        usd_to_gbp=args.usd_to_gbp,
        notes=cost_note,
    )
    summary = summarise(
        results,
        {
            "system": args.system,
            "endpoint": args.endpoint,
            "model_requested": args.model,
            "concurrency": args.concurrency,
            "permutations": args.permutations,
            "repeat_same_order": repeats,
            "wall_seconds": round(wall_s, 1),
            "usd": usd,
        },
    )
    write_outputs(args.out, results, cost, summary)
    print(json.dumps({k: summary[k] for k in ("n_states", "n_states_failed", "anomalies")}))
    if repeats:
        print(json.dumps({"repeat_same_order": repeats["per_question"]}))
    # Fail the workflow if nothing worked, so a dead key or endpoint is obvious.
    return 1 if summary["n_states_failed"] == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())

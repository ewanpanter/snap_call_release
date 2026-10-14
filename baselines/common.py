"""Shared plumbing for the baseline clients: loading, retries, a small thread pool, outputs.

A client supplies one function, ``decide(record) -> StateResult``, and ``run_baseline`` does
the rest. Nothing here ever sees a credential: clients read their own from the environment.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from snapcall.schema import (
    ApiCost,
    Cost,
    DataRecord,
    Prediction,
    read_jsonl,
    write_jsonl,
)

# USD -> GBP for cost.json (contract currency is GBP). Approximate; the raw USD figure is
# kept alongside in ApiCost.raw_cost, so this can be redone with the real rate later.
DEFAULT_USD_TO_GBP = 0.75

# Error text is truncated so a chatty upstream error body can't flood the outputs.
MAX_ERROR_CHARS = 300


class RetryableError(Exception):
    """A failure worth retrying (429, 5xx, timeouts). ``retry_after`` in seconds, if given."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class FatalError(Exception):
    """A failure that will repeat for every request (bad key, no credits): stop the run."""


@dataclass
class StateResult:
    """What one request (one state, all its questions) produced."""

    state_id: str
    predictions: list[Prediction]
    latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    usd_cost: float | None = None  # as reported by the provider, if it reports one
    model_version: str | None = None
    error: str | None = None
    attempts: int = 1
    requests: int = 1  # distinct requests (option-order permutations make several per state)
    anomalies: dict[str, list[str]] = field(default_factory=dict)  # question id -> kinds
    raw: Any = None  # the provider's response body, for raw_responses.jsonl


def truncate(text: str, n: int = MAX_ERROR_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def load_records(path: str | Path, limit: int | None = None) -> list[DataRecord]:
    """Read a data file. With ``limit``, take states round-robin across datasets, so a small
    probe run still covers every dataset and question type."""
    records = list(read_jsonl(path, DataRecord))
    if limit is None or limit >= len(records):
        return records
    by_ds: dict[str, list[DataRecord]] = {}
    for r in records:
        by_ds.setdefault(r.dataset, []).append(r)
    queues = [iter(v) for v in by_ds.values()]
    out: list[DataRecord] = []
    while len(out) < limit and queues:
        for q in list(queues):
            r = next(q, None)
            if r is None:
                queues.remove(q)
            elif len(out) < limit:
                out.append(r)
    return out


def with_retries[T](
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Call ``fn`` until it succeeds, retrying ``RetryableError`` with jittered exponential
    backoff (or the server's Retry-After). Returns (result, attempts used)."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), attempt
        except RetryableError as e:
            if attempt == max_attempts:
                raise RetryableError(f"{e} (after {attempt} attempts)", e.retry_after) from e
            delay = e.retry_after
            if delay is None:
                delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            sleep(min(delay, max_delay))
    raise AssertionError("unreachable")


def error_predictions(
    record: DataRecord, *, system: str, run_id: str | None, error: str, meta: dict | None = None
) -> list[Prediction]:
    """One error line per question, as the contract asks when a system fails."""
    return [
        Prediction(
            system=system,
            run_id=run_id,
            state_id=record.state_id,
            question_id=qid,
            type=q.type,
            answer=None,
            probabilities=None,
            error=truncate(error),
            meta={"dataset": record.dataset, "n_questions": len(record.questions), **(meta or {})},
        )
        for qid, q in record.questions.items()
    ]


def run_all(
    records: list[DataRecord],
    decide: Callable[[DataRecord], StateResult],
    on_failure: Callable[[DataRecord, str], StateResult],
    concurrency: int = 4,
    progress_every: int = 50,
) -> list[StateResult]:
    """Run ``decide`` over every record with a small thread pool, in input order.

    ``decide`` may raise: ``FatalError`` stops the run (every remaining state gets an error
    line instead of a pointless request), anything else becomes an error for that state.
    """
    stop = threading.Event()
    stop_reason: list[str] = []
    done = Counter()
    lock = threading.Lock()

    def one(record: DataRecord) -> StateResult:
        if stop.is_set():
            return on_failure(record, f"not attempted: run stopped ({stop_reason[0]})")
        try:
            result = decide(record)
        except FatalError as e:
            with lock:
                if not stop.is_set():
                    stop_reason.append(truncate(str(e), 120))
                    stop.set()
            result = on_failure(record, f"fatal: {e}")
        except Exception as e:  # noqa: BLE001 - one bad state must not kill the run
            result = on_failure(record, f"{type(e).__name__}: {e}")
        with lock:
            done["n"] += 1
            done["err"] += result.error is not None
            if done["n"] % progress_every == 0 or done["n"] == len(records):
                print(f"  {done['n']}/{len(records)} states, {done['err']} failed", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, records))


def latency_stats(values: Iterable[float | None]) -> dict[str, float] | None:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None

    def pct(p: float) -> float:
        return xs[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))]

    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "max_ms": xs[-1],
    }


def summarise(results: list[StateResult], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run-level facts for the report: errors, anomalies, models seen, tokens, latency."""
    preds = [p for r in results for p in r.predictions]
    anomaly_counts: Counter[str] = Counter()
    anomaly_examples: dict[str, list[str]] = {}
    for r in results:
        for qid, kinds in r.anomalies.items():
            for kind in kinds:
                key = kind.split(":", 1)[0]
                anomaly_counts[key] += 1
                ex = anomaly_examples.setdefault(key, [])
                if len(ex) < 5:
                    ex.append(f"{r.state_id}/{qid}: {kind}")
    return {
        "n_states": len(results),
        "n_states_failed": sum(r.error is not None for r in results),
        "n_predictions": len(preds),
        "n_prediction_errors": sum(p.error is not None for p in preds),
        "errors": dict(Counter(truncate(r.error, 80) for r in results if r.error).most_common(10)),
        "retried_states": sum(r.attempts > r.requests for r in results),
        "models_seen": dict(Counter(r.model_version for r in results if r.model_version)),
        "tokens": {
            "input": sum(r.input_tokens for r in results),
            "output": sum(r.output_tokens for r in results),
            "thinking": sum(r.thinking_tokens for r in results),
        },
        "latency": latency_stats(r.latency_ms for r in results if r.error is None),
        "anomalies": dict(anomaly_counts.most_common()),
        "anomaly_examples": anomaly_examples,
        **(extra or {}),
    }


def make_cost(
    *,
    run_id: str,
    job: str,
    system: str,
    provider: str,
    results: list[StateResult],
    usd: float,
    usd_to_gbp: float,
    notes: str,
) -> Cost:
    gbp = usd * usd_to_gbp
    api = ApiCost(
        provider=provider,
        requests=sum(
            r.attempts for r in results if not (r.error or "").startswith("not attempted")
        ),
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens + r.thinking_tokens for r in results),
        cost=gbp,
        raw_cost=usd,
        raw_currency="USD",
    )
    return Cost(
        run_id=run_id,
        job=job,
        system=system,
        created_at=datetime.now(UTC),
        api=[api],
        total=gbp,
        n_decisions=sum(p.error is None for r in results for p in r.predictions),
        notes=f"{notes} USD->GBP at {usd_to_gbp}. GitHub runner time not included.",
    )


def write_outputs(
    out_dir: str | Path, results: list[StateResult], cost: Cost, summary: dict[str, Any]
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out / "predictions.jsonl", (p for r in results for p in r.predictions))
    (out / "cost.json").write_text(cost.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with open(out / "raw_responses.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            row = {"state_id": r.state_id, "latency_ms": r.latency_ms, "error": r.error}
            f.write(json.dumps({**row, "response": r.raw}, ensure_ascii=False) + "\n")
    print(f"wrote {n} predictions to {out / 'predictions.jsonl'}")
    return out


# --------------------------------------------------------------------------------------
# Option-order permutations (the same debiasing SnapCall does, applied to an API)
# --------------------------------------------------------------------------------------


def option_shifts(n_options: int, k: int) -> list[int]:
    """SnapCall's cyclic shifts (``snapcall.prompt.permutations`` with seed 0): ``k`` shifts
    spread evenly over ``n_options``; shift 0 is the original order."""
    k = max(1, min(k, n_options))
    return [(i * n_options) // k for i in range(k)]


def rotated(record: DataRecord, perm: int, k: int) -> DataRecord | None:
    """``record`` with only its choice questions, options rotated to the ``perm``-th shift.

    Options are a key -> description map and the API answers with the key, so reordering
    the map changes what the model sees without changing what an answer means. Score
    levels can't be reordered (the order is the scale) and yes/no has no options, so
    those questions stay with the original request. ``None`` if nothing is left to ask."""
    questions, gold = {}, {}
    for qid, q in record.questions.items():
        if q.type != "choice":
            continue
        shifts = option_shifts(len(q.options), k)
        if perm >= len(shifts):
            continue
        keys = list(q.options)
        s = shifts[perm]
        order = keys[s:] + keys[:s]
        questions[qid] = q.model_copy(update={"options": {key: q.options[key] for key in order}})
        gold[qid] = record.gold[qid]
    if not questions:
        return None
    return record.model_copy(update={"questions": questions, "gold": gold})


def merge_permutations(record: DataRecord, results: list[StateResult], k: int) -> StateResult:
    """One ``StateResult`` from the original request (``results[0]``) plus the rotated ones.

    Choice questions: probabilities averaged over every order that answered, answer = the
    most probable key, every order's distribution kept in ``meta.per_permutation`` (first =
    original order, as SnapCall's predictions do), so the report computes the flip rate and
    the original-order accuracy. Other questions: the original request's prediction as is.
    Tokens, cost, attempts and latency are summed over the requests."""
    first = results[0]
    by_q: dict[str, list[Prediction]] = {}
    for r in results:
        for p in r.predictions:
            by_q.setdefault(p.question_id, []).append(p)
    merged = []
    for p in first.predictions:
        ok = [x for x in by_q.get(p.question_id, []) if x.error is None and x.probabilities]
        if p.type != "choice" or p.error is not None or len(ok) < 2:
            merged.append(p)
            continue
        keys = list(record.questions[p.question_id].options)
        dists = [{key: x.probabilities.get(key, 0.0) for key in keys} for x in ok]
        dists = [{key: v / (sum(d.values()) or 1) for key, v in d.items()} for d in dists]
        mean = {key: sum(d[key] for d in dists) / len(dists) for key in keys}
        meta = {
            **p.meta,
            "per_permutation": dists,
            "n_permutations": len(dists),
            "first_perm_answer": p.answer,
            "first_perm_confidence": p.confidence,
        }
        merged.append(
            p.model_copy(
                update={"answer": max(mean, key=mean.get), "probabilities": mean, "meta": meta}
            )
        )
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    usd = [r.usd_cost for r in results if r.usd_cost is not None]
    anomalies: dict[str, list[str]] = {}
    for r in results:
        for qid, kinds in r.anomalies.items():
            anomalies.setdefault(qid, []).extend(kinds)
    return StateResult(
        state_id=first.state_id,
        predictions=merged,
        latency_ms=sum(latencies) if latencies else None,
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
        thinking_tokens=sum(r.thinking_tokens for r in results),
        usd_cost=sum(usd) if usd else None,
        model_version=first.model_version,
        error=first.error,
        attempts=sum(r.attempts for r in results),
        requests=len(results),
        anomalies=anomalies,
        raw=[r.raw for r in results],
    )


def with_permutations(
    decide: Callable[[DataRecord], StateResult], k: int
) -> Callable[[DataRecord], StateResult]:
    """Wrap a client's ``decide`` to ask each choice question in ``k`` option orders.
    ``k <= 1`` returns ``decide`` unchanged. The requests for one state run one after
    another, so the state's latency is the sum, as a client doing this itself would see."""
    if k <= 1:
        return decide

    def decide_k(record: DataRecord) -> StateResult:
        results = [decide(record)]
        if results[0].error is None:
            for perm in range(1, k):
                variant = rotated(record, perm, k)
                if variant is None:
                    continue
                try:
                    results.append(decide(variant))
                except FatalError:
                    raise  # bad key, no credit: stop the run as usual
                except Exception as e:  # noqa: BLE001 - lose one order, keep the answer
                    print(f"  {record.state_id}: order {perm} failed: {truncate(str(e), 120)}")
        return merge_permutations(record, results, k)

    return decide_k


# --------------------------------------------------------------------------------------
# Same-order repeats (is the provider deterministic?)
# --------------------------------------------------------------------------------------


def _same(a: Any, b: Any) -> bool:
    return a == b if not (isinstance(a, float) and isinstance(b, float)) else abs(a - b) < 1e-9


def merge_repeats(passes: list[list[StateResult]]) -> tuple[list[StateResult], dict[str, Any]]:
    """Pass 1's results, with every later pass of the SAME request kept alongside.

    ``passes[i]`` is one full run over the data, all in the original option order and in
    the same state order. Pass 1 stays the scored prediction (so the run scores like any
    other); each prediction's ``meta.repeats`` lists the later passes' answer and
    probabilities. Tokens, cost and attempts are summed over passes; latency stays pass 1's,
    so the latency stats are still one call's. Also returns a summary: per question, how
    often a later pass gave a different answer (any change = the provider's own
    randomness, since the request was identical), the largest probability change, and
    latency per pass (a much faster later pass would hint at a server-side cache)."""
    first = passes[0]
    later = [{r.state_id: r for r in p} for p in passes[1:]]
    stats: dict[str, Counter[str]] = {}
    max_dp: dict[str, float] = {}
    merged: list[StateResult] = []
    for r in first:
        others = [p.get(r.state_id) for p in later]
        by_q = [{x.question_id: x for x in (o.predictions if o else [])} for o in others]
        preds = []
        for p in r.predictions:
            reps = [b.get(p.question_id) for b in by_q]
            rows = [
                None
                if x is None or x.error is not None
                else {"answer": x.answer, "probabilities": x.probabilities}
                for x in reps
            ]
            c = stats.setdefault(p.question_id, Counter())
            if p.error is None and all(row is not None for row in rows):
                c["n"] += 1
                c["answer_changed"] += any(not _same(row["answer"], p.answer) for row in rows)
                c["probs_changed"] += any(row["probabilities"] != p.probabilities for row in rows)
                for row in rows:
                    for k, v in (p.probabilities or {}).items():
                        d = abs(v - (row["probabilities"] or {}).get(k, 0.0))
                        max_dp[p.question_id] = max(max_dp.get(p.question_id, 0.0), d)
            else:
                c["skipped"] += 1
            preds.append(p.model_copy(update={"meta": {**p.meta, "repeats": rows}}))
        done = [r] + [o for o in others if o is not None]
        usd = [x.usd_cost for x in done if x.usd_cost is not None]
        merged.append(
            StateResult(
                state_id=r.state_id,
                predictions=preds,
                latency_ms=r.latency_ms,
                input_tokens=sum(x.input_tokens for x in done),
                output_tokens=sum(x.output_tokens for x in done),
                thinking_tokens=sum(x.thinking_tokens for x in done),
                usd_cost=sum(usd) if usd else None,
                model_version=r.model_version,
                error=r.error,
                attempts=sum(x.attempts for x in done),
                requests=len(done),
                anomalies=r.anomalies,
                raw=[x.raw for x in done],
            )
        )
    per_q = {
        qid: {
            "n": c["n"],
            "skipped": c["skipped"],
            "answer_changed": c["answer_changed"],
            "answer_change_rate": c["answer_changed"] / c["n"] if c["n"] else None,
            "probs_changed": c["probs_changed"],
            "max_abs_prob_change": max_dp.get(qid, 0.0),
        }
        for qid, c in stats.items()
    }
    summary = {
        "passes": len(passes),
        "per_question": per_q,
        "latency_per_pass": [
            latency_stats(x.latency_ms for x in p if x.error is None) for p in passes
        ],
        "models_per_pass": [
            dict(Counter(x.model_version for x in p if x.model_version)) for p in passes
        ],
    }
    return merged, summary

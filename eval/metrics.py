"""Metrics over ``predictions.jsonl`` + gold, for any system (SnapCall, Jev, Gemini).

Definitions (the contract's ``TypeMetrics`` docstring, plus what goes in ``extra``):

- **accuracy**: the system's ``answer`` equals gold (yes/no: ``answer`` is p_yes >= 0.5).
- **ece**: expected calibration error of the probability the system gave its own answer,
  10 equal-width bins, weighted by bin size. ``reliability`` holds the bins.
- **brier**: multi-class Brier, the sum over answer keys of (p - 1[key is gold])^2. For yes/no
  that is twice the familiar binary Brier; ``extra`` also has the binary one.
- **log_loss**: -log p(gold), with p clipped at ``LOG_LOSS_FLOOR``. Jev rounds to two decimals
  and often says 0.00, which would make log loss infinite, so the floor matters.
- **mae** (score only): |expected level - gold level|.
- ``extra``: ``macro_f1__<qid>`` (F1 averaged over the gold classes, so rare queues count as
  much as common ones), ``auroc__<qid>`` and ``ap__<qid>`` (average precision) for yes/no, which
  with ~10% positives say more than accuracy (always-"no" already scores ~90%), the positive
  rate, and order-sensitivity numbers when the predictions carry per-permutation results.

Probabilities are renormalised to sum to 1 before scoring (Gemini's stated ones sometimes sum
to 1.05). Error predictions are counted in ``n_errors`` and scored as wrong for accuracy
(so a system can't gain by failing on hard cases); they're left out of the probability metrics.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations

import numpy as np

from snapcall.schema import (
    DataRecord,
    LatencyStats,
    Metrics,
    Prediction,
    ReliabilityBin,
    TypeMetrics,
)

N_BINS = 10
LOG_LOSS_FLOOR = 1e-3


def gold_key(gold) -> str:
    """Gold answer -> the probability key it corresponds to."""
    if isinstance(gold, bool):
        return "yes" if gold else "no"
    return str(gold)


def answer_key(answer) -> str:
    return gold_key(answer)


def normalise(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    if total <= 0:
        return {k: 1 / len(probs) for k in probs}
    return {k: v / total for k, v in probs.items()}


@dataclass(frozen=True)
class Scored:
    """One prediction joined with its gold answer."""

    state_id: str
    dataset: str
    question_id: str
    type: str
    gold: str
    answer: str | None  # None = error
    probs: dict[str, float] | None
    value: float | None
    latency_ms: float | None
    per_permutation: list[dict[str, float]] | None

    @property
    def correct(self) -> bool:
        return self.answer is not None and self.answer == self.gold


def join(predictions: Iterable[Prediction], records: Iterable[DataRecord]) -> list[Scored]:
    """Join predictions to gold on (state_id, question_id). Every gold question gets a row:
    a question the system never answered counts as an error."""
    preds = {(p.state_id, p.question_id): p for p in predictions}
    out = []
    for r in records:
        for qid, q in r.questions.items():
            p = preds.get((r.state_id, qid))
            ok = p is not None and p.error is None
            probs = normalise(p.probabilities) if ok else None
            value = None
            if ok and q.type == "score":
                value = (
                    p.value if p.value is not None else sum(int(k) * v for k, v in probs.items())
                )
            out.append(
                Scored(
                    state_id=r.state_id,
                    dataset=r.dataset,
                    question_id=qid,
                    type=q.type,
                    gold=gold_key(r.gold[qid]),
                    answer=answer_key(p.answer) if ok else None,
                    probs=probs,
                    value=value,
                    latency_ms=p.latency_ms if p is not None else None,
                    per_permutation=(p.meta.get("per_permutation") if ok else None),
                )
            )
    return out


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    """Equal-width bins over [0, 1] -> (ece, bins)."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        mc = float(conf[m].mean()) if n else None
        acc = float(correct[m].mean()) if n else None
        if n:
            ece += n / len(conf) * abs(mc - acc)
        bins.append(
            ReliabilityBin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                count=n,
                mean_confidence=mc,
                accuracy=acc,
            )
        )
    return ece, bins


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Probability a random positive outscores a random negative (ties count half)."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]  # every (positive, negative) pair; test sets are small
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the precision-recall curve (step-wise; ties broken pessimistically)."""
    if labels.sum() == 0:
        return None
    order = np.lexsort((labels, -scores))  # high score first; negatives first among ties
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def macro_f1(gold: list[str], pred: list[str | None]) -> float | None:
    """F1 per gold class, averaged. An error (None) is a miss for its gold class."""
    classes = sorted(set(gold))
    if not classes:
        return None
    f1s = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, pred, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, pred, strict=True))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def order_sensitivity(rows: list[Scored]) -> dict[str, float] | None:
    """From per-permutation distributions (SnapCall only): mean pairwise total-variation
    distance, the share of questions whose top answer changes with option order, and the
    accuracy of the first permutation alone (= what debias=off would have answered)."""
    usable = [r for r in rows if r.per_permutation and len(r.per_permutation) > 1]
    if not usable:
        return None
    tvs, flips, first_correct = [], [], []
    for r in usable:
        dists = r.per_permutation
        pairs = list(combinations(dists, 2))
        tvs.append(sum(0.5 * sum(abs(p[k] - q[k]) for k in p) for p, q in pairs) / len(pairs))
        tops = {max(d, key=d.get) for d in dists}
        flips.append(len(tops) > 1)
        first_correct.append(max(dists[0], key=dists[0].get) == r.gold)
    return {
        "mean_tv": float(np.mean(tvs)),
        "flip_rate": float(np.mean(flips)),
        "first_perm_accuracy": float(np.mean(first_correct)),
        "n": len(usable),
    }


# --------------------------------------------------------------------------------------
# TypeMetrics over a group of rows
# --------------------------------------------------------------------------------------


def type_metrics(rows: list[Scored], with_reliability: bool = False) -> TypeMetrics:
    ok = [r for r in rows if r.answer is not None]
    n_err = len(rows) - len(ok)
    if not rows:
        return TypeMetrics(n=0)
    acc = sum(r.correct for r in rows) / len(rows)
    if not ok:
        return TypeMetrics(n=len(rows), n_errors=n_err, accuracy=acc)
    conf = np.array([r.probs[r.answer] for r in ok])
    correct = np.array([r.correct for r in ok], dtype=float)
    ece, bins = reliability(conf, correct)
    brier = np.mean([sum((p - (k == r.gold)) ** 2 for k, p in r.probs.items()) for r in ok])
    ll = np.mean([-math.log(max(r.probs.get(r.gold, 0.0), LOG_LOSS_FLOOR)) for r in ok])
    scores = [r for r in ok if r.type == "score"]
    mae = float(np.mean([abs(r.value - int(r.gold)) for r in scores])) if scores else None
    return TypeMetrics(
        n=len(rows),
        n_errors=n_err,
        accuracy=float(acc),
        ece=float(ece),
        brier=float(brier),
        log_loss=float(ll),
        mae=mae,
        reliability=bins if with_reliability else None,
    )


def question_extra(qid: str, rows: list[Scored]) -> dict[str, float]:
    extra: dict[str, float] = {}
    f1 = macro_f1([r.gold for r in rows], [r.answer for r in rows])
    if f1 is not None:
        extra[f"macro_f1__{qid}"] = f1
    if rows and rows[0].type == "yesno":
        ok = [r for r in rows if r.probs is not None]
        s = np.array([r.probs["yes"] for r in ok])
        y = np.array([r.gold == "yes" for r in ok])
        extra[f"positive_rate__{qid}"] = float(y.mean()) if len(y) else 0.0
        for name, fn in (("auroc", auroc), ("ap", average_precision)):
            v = fn(s, y) if len(ok) else None
            if v is not None:
                extra[f"{name}__{qid}"] = v
        if len(ok):
            extra[f"binary_brier__{qid}"] = float(np.mean((s - y) ** 2))
            tp = int(((s >= 0.5) & y).sum())
            extra[f"recall__{qid}"] = tp / max(1, int(y.sum()))
            extra[f"precision__{qid}"] = tp / max(1, int((s >= 0.5).sum()))
    return extra


COVERAGE_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def coverage_curve(rows: list[Scored], target: float = 0.9) -> dict | None:
    """Selective prediction: answer only the most confident decisions, send the rest to a
    person. Sorts decisions by the probability the system gave its own answer and reports
    the accuracy of the top ``x`` share for each coverage level, plus the largest coverage
    whose accuracy is still at least ``target``. Errors count as wrong and least confident.
    Decisions with tied confidence count at their group's average accuracy.
    """
    if not rows:
        return None
    conf = np.array([r.probs[r.answer] if r.answer is not None else -1.0 for r in rows])
    correct = np.array([r.correct for r in rows], dtype=float)
    order = np.argsort(-conf, kind="stable")
    conf, correct = conf[order], correct[order]
    # Tied confidences (Jev rounds to two decimals) can't be ranked against each other:
    # give every decision in a tie its group's mean, so the curve doesn't depend on order.
    _, start, counts = np.unique(-conf, return_index=True, return_counts=True)
    for s0, n in zip(start, counts, strict=True):
        if n > 1:
            correct[s0 : s0 + n] = correct[s0 : s0 + n].mean()
    cum = np.cumsum(correct) / np.arange(1, len(rows) + 1)
    points = []
    for c in COVERAGE_LEVELS:
        k = max(1, round(c * len(rows)))
        points.append({"coverage": c, "accuracy": float(cum[k - 1])})
    ok = np.nonzero(cum >= target)[0]
    at_target = float((ok.max() + 1) / len(rows)) if len(ok) else 0.0
    return {"points": points, "target": target, "coverage_at_target": at_target}


def brier_row(r: Scored) -> float | None:
    """One decision's multi-class Brier (lower is better); None for an error."""
    if r.probs is None:
        return None
    return sum((p - (k == r.gold)) ** 2 for k, p in r.probs.items()) + (
        0.0 if r.gold in r.probs else 1.0
    )


def paired_bootstrap(
    a: list[Scored],
    b: list[Scored],
    n: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    score=lambda r: float(r.correct),
) -> dict[str, dict[str, float]]:
    """Mean ``score`` of ``b`` minus that of ``a`` on the same decisions (default: accuracy),
    with a paired bootstrap interval, per question and ``"all"``. Resamples whole states (a
    state's questions share one ticket, so they aren't independent). Only decisions both
    systems were asked, and both scored (``score`` not None), count; for accuracy, errors
    count as wrong. ``{qid: {"diff", "lo", "hi", "n"}}``."""
    ca = {(r.state_id, r.question_id): score(r) for r in a}
    cb = {(r.state_id, r.question_id): score(r) for r in b}
    ca = {k: v for k, v in ca.items() if v is not None}
    cb = {k: v for k, v in cb.items() if v is not None}
    keys = sorted(ca.keys() & cb.keys())
    if not keys:
        return {}
    states = sorted({s for s, _ in keys})
    qids = sorted({q for _, q in keys})
    si, qi = {s: i for i, s in enumerate(states)}, {q: i for i, q in enumerate(qids)}
    # per state x question: b - a in {-1, 0, 1}, and whether the pair exists
    diff = np.zeros((len(states), len(qids)))
    have = np.zeros((len(states), len(qids)))
    for s, q in keys:
        diff[si[s], qi[q]] = float(cb[(s, q)]) - float(ca[(s, q)])
        have[si[s], qi[q]] = 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(states), size=(n, len(states)))
    # counts[k, s] = how often state s is drawn in resample k
    counts = np.zeros((n, len(states)))
    np.add.at(counts, (np.arange(n)[:, None], idx), 1.0)
    tail = (1 - level) / 2 * 100
    out = {}
    columns = [(q, have[:, [qi[q]]], diff[:, [qi[q]]]) for q in qids]
    columns.append(("all", have, diff))
    for name, h, d in columns:
        num, den = counts @ (d * h).sum(axis=1), counts @ h.sum(axis=1)
        boot = num[den > 0] / den[den > 0]
        out[name] = {
            "diff": float((d * h).sum() / h.sum()),
            "lo": float(np.percentile(boot, tail)),
            "hi": float(np.percentile(boot, 100 - tail)),
            "n": int(h.sum()),
        }
    return out


def balanced_accuracy(rows: list[Scored]) -> float | None:
    """Yes/no: the mean of yes-recall and no-recall (always "no" scores 0.5). Errors are wrong."""
    recalls = []
    for label in ("yes", "no"):
        of = [r for r in rows if r.gold == label]
        if not of:
            return None
        recalls.append(sum(r.correct for r in of) / len(of))
    return sum(recalls) / 2


def _auc_arrays(rows: list[Scored]):
    keep = [r for r in rows if r.probs is not None]
    return (
        [r.state_id for r in keep],
        np.array([r.probs.get("yes", 0.0) for r in keep]),
        np.array([r.gold == "yes" for r in keep]),
    )


def paired_bootstrap_auc(
    a: list[Scored], b: list[Scored], n: int = 2000, seed: int = 0, level: float = 0.95
) -> dict[str, dict[str, float]]:
    """AUROC of p_yes for one yes/no question: ``a``, ``b`` and ``b - a``, each with a
    bootstrap interval from resampling the same states for both (pass the rows of one
    question; ``a is b`` gives one system's interval). Decisions either system failed on
    are left out of both."""
    ka = {(r.state_id): r for r in a if r.probs is not None}
    kb = {(r.state_id): r for r in b if r.probs is not None}
    states = sorted(ka.keys() & kb.keys())
    if not states:
        return {}
    _, sa, la = _auc_arrays([ka[s] for s in states])
    _, sb, _ = _auc_arrays([kb[s] for s in states])
    rng = np.random.default_rng(seed)
    boots = {"a": [], "b": [], "diff": []}
    for _ in range(n):
        i = rng.integers(0, len(states), size=len(states))
        xa, xb = auroc(sa[i], la[i]), auroc(sb[i], la[i])
        if xa is None or xb is None:
            continue
        boots["a"].append(xa)
        boots["b"].append(xb)
        boots["diff"].append(xb - xa)
    point = {"a": auroc(sa, la), "b": auroc(sb, la)}
    point["diff"] = point["b"] - point["a"]
    tail = (1 - level) / 2 * 100
    return {
        k: {
            "value": point[k],
            "lo": float(np.percentile(v, tail)),
            "hi": float(np.percentile(v, 100 - tail)),
            "n": len(states),
        }
        for k, v in boots.items()
    }


def latency_stats(rows: list[Scored]) -> LatencyStats | None:
    per_state = {}
    for r in rows:
        if r.latency_ms is not None:
            per_state[r.state_id] = r.latency_ms
    if not per_state:
        return None
    v = np.array(list(per_state.values()))
    return LatencyStats(
        n=len(v),
        mean_ms=float(v.mean()),
        p50_ms=float(np.percentile(v, 50)),
        p95_ms=float(np.percentile(v, 95)),
    )


def compute(
    predictions: Iterable[Prediction],
    records: list[DataRecord],
    *,
    system: str,
    run_id: str | None = None,
    dataset: str | None = None,
    split: str = "test",
) -> Metrics:
    """One ``Metrics`` over ``records`` (optionally one dataset only)."""
    if dataset is not None:
        records = [r for r in records if r.dataset == dataset]
    rows = join(predictions, records)
    by_type: dict[str, list[Scored]] = defaultdict(list)
    by_q: dict[str, list[Scored]] = defaultdict(list)
    for r in rows:
        by_type[r.type].append(r)
        by_q[r.question_id].append(r)

    extra: dict[str, float] = {}
    for qid, qrows in by_q.items():
        extra.update(question_extra(qid, qrows))
        o = order_sensitivity(qrows)
        if o:
            extra[f"order_tv__{qid}"] = o["mean_tv"]
            extra[f"order_flip_rate__{qid}"] = o["flip_rate"]
            extra[f"first_perm_accuracy__{qid}"] = o["first_perm_accuracy"]
    macro = [v for k, v in extra.items() if k.startswith("macro_f1__")]
    if macro:
        extra["macro_f1_mean"] = float(np.mean(macro))
    o_all = order_sensitivity(rows)
    if o_all:
        extra["order_flip_rate"] = o_all["flip_rate"]
        extra["first_perm_accuracy"] = o_all["first_perm_accuracy"]

    return Metrics(
        system=system,
        run_id=run_id,
        dataset=dataset,
        split=split,
        created_at=datetime.now(UTC),
        n_states=len(records),
        n_questions=len(rows),
        overall=type_metrics(rows),
        by_type={t: type_metrics(v, with_reliability=True) for t, v in by_type.items()},
        by_question={q: type_metrics(v, with_reliability=True) for q, v in by_q.items()},
        latency=latency_stats(rows),
        order_sensitivity=o_all["mean_tv"] if o_all else None,
        extra=extra,
    )


def compute_all(predictions: list[Prediction], records: list[DataRecord], **kw) -> dict:
    """``{"all": Metrics, <dataset>: Metrics, ...}``."""
    out = {"all": compute(predictions, records, **kw)}
    for ds in sorted({r.dataset for r in records}):
        out[ds] = compute(predictions, records, dataset=ds, **kw)
    return out

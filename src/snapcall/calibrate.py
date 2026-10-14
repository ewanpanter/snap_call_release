"""Temperature scaling, one temperature per question type, fitted on the validation split.

A model is calibrated if its "80% sure" answers are right about 80% of the time. Temperature
scaling divides the label logits by one number ``T`` before the softmax: ``T > 1`` softens
over-confident probabilities, ``T < 1`` sharpens timid ones. It never changes which label a
single branch prefers, so accuracy barely moves (it can move a little after averaging over
permutations); only the probabilities do.

No second GPU pass is needed. SnapCall predictions keep every permutation's label
distribution at ``T = 1`` (``meta.per_permutation``, written by the eval harness), and

    softmax(logits / T)  ∝  softmax(logits) ** (1 / T)

so any temperature can be applied exactly from those distributions: rescale each
permutation, average them as ``readout.py`` does, and rebuild the answer.

**Yes/no also gets a bias** ``b``, added to the "yes" logit after the temperature:
``p(yes) = sigmoid((logit_yes - logit_no) / T + b)``. A temperature can't move a yes/no
decision threshold; a bias can. Zero-shot Qwen3-4B answers "yes" to ``wants_to_leave`` far too
often (precision 0.24 at 0.5, though AUROC 0.91: it ranks well, its threshold is off).

The bias is fitted **per dataset**, or per question when that question has enough val data
(``MIN_QUESTION_ITEMS``, with ``MIN_CLASS`` of each answer), on that dataset's val split only
(Decision 14). A dataset with no val data (every OOD set) gets **no bias**, only the
temperature: a threshold learned on support tickets is the wrong one for AG News (one shared
bias cost Qwen3-8B 7 points on AG News ``is_business``). The yes/no temperature and the biases
are fitted together on validation NLL, alternating between them (Platt scaling on the logit
difference, with one intercept per group). Bias keys are ``"<dataset>:<question>"`` or
``"<dataset>:*"``; an old ``{"yesno": b}`` (one shared bias) is still read.

    python -m snapcall.calibrate fit   --predictions val/predictions.jsonl --data val.jsonl \\
        --out temperatures.json
    python -m snapcall.calibrate apply --predictions predictions.jsonl --data test.jsonl \\
        --temperatures temperatures.json --out calibrated.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snapcall.readout import QuestionResult, agreement, to_answer
from snapcall.schema import (
    DataRecord,
    DecideResponse,
    Prediction,
    QuestionType,
    answer_keys,
    predictions_from_response,
    read_jsonl,
    write_jsonl,
)

# Probabilities below this are treated as this (softmax in float32 underflows to 0).
PROB_FLOOR = 1e-30
# Search range for T, and the grid the search starts from. Bounded: if validation has no
# errors for a type (0.6B smoke run: 100/100 yes/no right), NLL keeps falling as T -> 0 and an
# unbounded fit would make every test answer near-certain. A fit on a bound is flagged.
T_MIN, T_MAX, GRID = 0.2, 10.0, 241
B_MAX = 8.0  # yes/no bias range: [-B_MAX, B_MAX] (e^8 ~ 3000x odds)
# A yes/no question gets its own bias with at least this many val items and MIN_CLASS of each
# answer; otherwise it shares its dataset's bias, if the dataset has MIN_DATASET_ITEMS.
MIN_QUESTION_ITEMS, MIN_CLASS, MIN_DATASET_ITEMS = 100, 10, 30
FIT_ROUNDS = 4  # alternations between the yes/no temperature and the biases
N_BINS = 15  # for ECE; the eval harness uses its own


@dataclass(frozen=True)
class Item:
    """One answered question: log-probabilities per permutation, and the gold key's index."""

    type: QuestionType
    log_probs: np.ndarray  # [n_permutations, n_keys], canonical key order
    gold: int
    dataset: str = ""
    question_id: str = ""


def gold_key(question, gold) -> str:
    if question.type == "yesno":
        return "yes" if gold else "no"
    return str(gold)


def items_from(predictions: Iterable[Prediction], records: Iterable[DataRecord]) -> list[Item]:
    """Join predictions to gold. Skips errored predictions and ones without permutations."""
    by_id = {r.state_id: r for r in records}
    items = []
    for p in predictions:
        perms = p.meta.get("per_permutation")
        record = by_id.get(p.state_id)
        if p.error or not perms or record is None:
            continue
        q = record.questions[p.question_id]
        keys = answer_keys(q)
        probs = np.array([[d[k] for k in keys] for d in perms], dtype=np.float64)
        gold = keys.index(gold_key(q, record.gold[p.question_id]))
        log_probs = np.log(np.maximum(probs, PROB_FLOOR))
        items.append(Item(q.type, log_probs, gold, record.dataset, p.question_id))
    return items


def tempered(log_probs: np.ndarray, t: float, bias: float = 0.0) -> np.ndarray:
    """Per-permutation distributions at temperature ``t`` (plus ``bias`` on key 0, "yes"),
    averaged: ``[n_keys]``."""
    z = log_probs / t
    z[..., 0] += bias
    z = np.exp(z - z.max(axis=-1, keepdims=True))
    return (z / z.sum(axis=-1, keepdims=True)).mean(axis=0)


@dataclass(frozen=True)
class _Stack:
    """Items padded into arrays, so a temperature is applied to all of them at once."""

    log_probs: np.ndarray  # [n, max_perms, max_keys]; padded keys are -inf
    weight: np.ndarray  # [n, max_perms]: 1 / n_perms for real permutations, 0 for padding
    gold: np.ndarray  # [n]

    @classmethod
    def of(cls, items: list[Item]) -> _Stack:
        n = len(items)
        perms = max(i.log_probs.shape[0] for i in items)
        keys = max(i.log_probs.shape[1] for i in items)
        lp = np.full((n, perms, keys), -np.inf)
        lp[:, :, 0] = 0.0  # padded permutations: any valid distribution, weight 0
        weight = np.zeros((n, perms))
        for j, i in enumerate(items):
            p, k = i.log_probs.shape
            lp[j, :p, :] = -np.inf
            lp[j, :p, :k] = i.log_probs
            weight[j, :p] = 1.0 / p
        return cls(lp, weight, np.array([i.gold for i in items]))

    def dists(self, t: float, bias: float | np.ndarray = 0.0) -> np.ndarray:
        """``[n, max_keys]``: each item's averaged distribution at temperature ``t``, with
        ``bias`` (one number, or one per item) added to key 0 (yes/no only: "yes")."""
        z = self.log_probs / t
        b = np.asarray(bias, dtype=np.float64)
        z[..., 0] += b[:, None] if b.ndim else b
        z = np.exp(z - z.max(axis=-1, keepdims=True))
        z /= z.sum(axis=-1, keepdims=True)
        return (z * self.weight[..., None]).sum(axis=1)

    def nll(self, t: float, bias: float | np.ndarray = 0.0) -> float:
        p = self.dists(t, bias)[np.arange(len(self.gold)), self.gold]
        return -float(np.log(np.maximum(p, 1e-12)).mean())


def nll(items: list[Item], t: float, bias: float = 0.0) -> float:
    """Mean negative log-likelihood of the gold answers at temperature ``t``."""
    return _Stack.of(items).nll(t, bias) if items else float("nan")


def ece(items: list[Item], t: float, bias: float = 0.0, n_bins: int = N_BINS) -> float:
    """Expected calibration error of the top answer, equal-width bins."""
    if not items:
        return float("nan")
    stack = _Stack.of(items)
    dists = stack.dists(t, bias)
    conf = dists.max(axis=1)
    correct = dists.argmax(axis=1) == stack.gold
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(
        sum(
            abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
            for b in range(n_bins)
            if (bins == b).any()
        )
    )


def accuracy(items: list[Item], t: float = 1.0, bias: float = 0.0) -> float:
    if not items:
        return float("nan")
    stack = _Stack.of(items)
    return float((stack.dists(t, bias).argmax(axis=1) == stack.gold).mean())


def _minimise(loss, lo: float, hi: float, n: int) -> float:
    """Minimise a 1-D ``loss`` on [lo, hi]: an n-point grid, then golden-section search
    between the grid neighbours of the best point."""
    grid = np.linspace(lo, hi, n)
    best = int(np.argmin([loss(x) for x in grid]))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, n - 1)]
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(40):
        if loss(a) < loss(b):
            hi, b = b, a
            a = hi - g * (hi - lo)
        else:
            lo, a = a, b
            b = lo + g * (hi - lo)
    return float((lo + hi) / 2)


def fit_temperature(items: list[Item], bias: float | np.ndarray = 0.0) -> float:
    """The ``T`` that minimises validation NLL (with ``bias`` held fixed), searched on a log
    scale. 1.0 if there's nothing to fit."""
    if not items:
        return 1.0
    stack = _Stack.of(items)
    log_t = _minimise(
        lambda lt: stack.nll(math.exp(lt), bias), math.log(T_MIN), math.log(T_MAX), GRID
    )
    return math.exp(log_t)


def fit_bias(items: list[Item], t: float) -> float:
    """The yes/no bias that minimises validation NLL at temperature ``t``."""
    stack = _Stack.of(items)
    return _minimise(lambda b: stack.nll(t, b), -B_MAX, B_MAX, 161)


def bias_groups(items: list[Item]) -> dict[str, list[int]]:
    """Yes/no items (indices into ``items``) grouped by which bias they get: their own
    question's (``"<dataset>:<question>"``) if it has enough val data, else their dataset's
    (``"<dataset>:*"``). Datasets too small for either get no group: no bias."""
    by_question: dict[tuple[str, str], list[int]] = {}
    for j, i in enumerate(items):
        by_question.setdefault((i.dataset, i.question_id), []).append(j)
    groups: dict[str, list[int]] = {}
    pooled: dict[str, list[int]] = {}
    for (dataset, qid), idx in by_question.items():
        n_yes = sum(items[j].gold == 0 for j in idx)  # key 0 is "yes"
        if len(idx) >= MIN_QUESTION_ITEMS and min(n_yes, len(idx) - n_yes) >= MIN_CLASS:
            groups[f"{dataset}:{qid}"] = idx
        else:
            pooled.setdefault(dataset, []).extend(idx)
    for dataset, idx in pooled.items():
        if len(idx) >= MIN_DATASET_ITEMS:
            groups[f"{dataset}:*"] = idx
    return groups


def fit_temperature_and_biases(items: list[Item]) -> tuple[float, dict[str, float]]:
    """Yes/no: one temperature and one bias per group (``bias_groups``) minimising validation
    NLL, alternating between them. ``(1.0, {})`` if there's nothing to fit."""
    if not items:
        return 1.0, {}
    groups = bias_groups(items)
    biases = {g: 0.0 for g in groups}
    t = 1.0
    for _ in range(FIT_ROUNDS):
        t = fit_temperature(items, item_biases(items, groups, biases))
        biases = {g: fit_bias([items[j] for j in idx], t) for g, idx in groups.items()}
    return t, biases


def item_biases(items: list[Item], groups: dict[str, list[int]], biases: dict[str, float]):
    """One bias per item: its group's, or 0 if it has none."""
    out = np.zeros(len(items))
    for g, idx in groups.items():
        out[idx] = biases[g]
    return out


def bias_for(bias: dict[str, float] | None, dataset: str, question_id: str, qtype: str) -> float:
    """The bias a question gets: its own, else its dataset's, else an old-style shared one
    (``{"yesno": b}``), else none."""
    bias = bias or {}
    for key in (f"{dataset}:{question_id}", f"{dataset}:*", qtype):
        if key in bias:
            return bias[key]
    return 0.0


def fit(
    predictions: Iterable[Prediction], records: Iterable[DataRecord]
) -> tuple[dict[QuestionType, float], dict[str, float], dict]:
    """One temperature per question type, yes/no biases per dataset or question, and a
    report of NLL/ECE/accuracy before and after: ``(temperatures, bias, report)``."""
    items = items_from(predictions, records)
    temperatures: dict[QuestionType, float] = {}
    biases: dict[QuestionType, float] = {}
    report: dict = {}
    for qtype in ("choice", "score", "yesno"):
        of_type = [i for i in items if i.type == qtype]
        if not of_type:
            continue
        if qtype == "yesno":
            t, group_bias = fit_temperature_and_biases(of_type)
            biases.update(group_bias)
            b = item_biases(of_type, bias_groups(of_type), group_bias)
            report["yesno_bias"] = {
                g: {
                    "n": len(idx),
                    "n_yes": int(sum(of_type[j].gold == 0 for j in idx)),
                    "bias": group_bias[g],
                    "at_bound": bool(abs(group_bias[g]) > B_MAX * 0.99),
                }
                for g, idx in bias_groups(of_type).items()
            }
        else:
            t, b = fit_temperature(of_type), 0.0
        temperatures[qtype] = t
        report[qtype] = {
            "n": len(of_type),
            "temperature": t,
            "at_bound": bool(not T_MIN * 1.01 < t < T_MAX / 1.01),
            "nll_before": nll(of_type, 1.0),
            "nll_after": nll(of_type, t, b),
            "ece_before": ece(of_type, 1.0),
            "ece_after": ece(of_type, t, b),
            "accuracy_before": accuracy(of_type, 1.0),
            "accuracy_after": accuracy(of_type, t, b),
        }
    return temperatures, biases, report


def apply_temperature(
    prediction: Prediction, question, t: float, system: str, bias: float = 0.0
) -> Prediction:
    """``prediction`` re-read at temperature ``t`` (and ``bias`` on the first key, "yes"), as
    ``system``. Unchanged (apart from the system name) if it has no per-permutation
    distributions."""
    perms = prediction.meta.get("per_permutation")
    if prediction.error or not perms:
        return prediction.model_copy(update={"system": system})
    keys = answer_keys(question)
    log_probs = np.log(np.maximum([[d[k] for k in keys] for d in perms], PROB_FLOOR))
    dists = []
    for row in log_probs:
        z = row / t
        z[0] += bias
        z = np.exp(z - z.max())
        dists.append(dict(zip(keys, (z / z.sum()).tolist(), strict=True)))
    mean = {k: sum(d[k] for d in dists) / len(dists) for k in keys}
    result = QuestionResult(prediction.question_id, question.type, mean, dists, agreement(dists))
    response = DecideResponse(
        model=system,
        answers={prediction.question_id: to_answer(question, result)},
        latency_ms=prediction.latency_ms,
    )
    (out,) = predictions_from_response(
        response,
        system=system,
        state_id=prediction.state_id,
        run_id=prediction.run_id,
        meta={**prediction.meta, "per_permutation": dists, "temperature": t, "bias": bias},
    )
    return out


def apply(
    predictions: Iterable[Prediction],
    records: Iterable[DataRecord],
    temperatures: dict[QuestionType, float],
    system: str | None = None,
    bias: dict[str, float] | None = None,
) -> list[Prediction]:
    """Re-read ``predictions`` with the fitted temperatures and biases (``bias_for``): a
    dataset that had no val data, such as an OOD set, gets temperature only."""
    by_id = {r.state_id: r for r in records}
    out = []
    for p in predictions:
        record = by_id[p.state_id]
        q = record.questions[p.question_id]
        t = temperatures.get(q.type, 1.0)
        b = bias_for(bias, record.dataset, p.question_id, q.type) if q.type == "yesno" else 0.0
        out.append(apply_temperature(p, q, t, system or p.system, b))
    return out


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m snapcall.calibrate")
    parser.add_argument("command", choices=["fit", "apply"])
    parser.add_argument("--predictions", required=True, help="predictions.jsonl (T = 1)")
    parser.add_argument("--data", required=True, help="the DataRecord jsonl they answer")
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperatures", help="apply: temperatures.json from fit")
    parser.add_argument("--system", help="apply: system name for the output (default: same)")
    args = parser.parse_args(argv)

    preds = list(read_jsonl(args.predictions, Prediction))
    records = list(read_jsonl(args.data, DataRecord))
    if args.command == "fit":
        temperatures, bias, report = fit(preds, records)
        body = {"temperatures": temperatures, "bias": bias, "report": report}
        Path(args.out).write_text(json.dumps(body, indent=2) + "\n")
        print(json.dumps(body, indent=2))
        return 0
    if not args.temperatures:
        parser.error("apply needs --temperatures")
    body = json.loads(Path(args.temperatures).read_text())
    temperatures, bias = body["temperatures"], body.get("bias", {})
    n = write_jsonl(args.out, apply(preds, records, temperatures, args.system, bias))
    print(f"{n} predictions at T={temperatures}, bias={bias} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

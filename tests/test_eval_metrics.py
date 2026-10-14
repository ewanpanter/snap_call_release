import math

import numpy as np
import pytest

from eval import metrics
from snapcall.schema import DataRecord, Metrics, Prediction, read_jsonl

EXAMPLES = "contract/examples"


def rec(state_id, gold, dataset="d"):
    return DataRecord.model_validate(
        {
            "state_id": state_id,
            "dataset": dataset,
            "split": "test",
            "state": "s",
            "questions": {
                "q": {"type": "choice", "question": "?", "options": {"a": "A", "b": "B", "c": "C"}},
                "y": {"type": "yesno", "question": "?"},
                "s": {"type": "score", "question": "?", "levels": ["lo", "mid", "hi"]},
            },
            "gold": gold,
        }
    )


def pred(state_id, qid, probs, answer=None, **kw):
    typ = {"q": "choice", "y": "yesno", "s": "score"}[qid]
    if answer is None:
        top = max(probs, key=probs.get)
        answer = top if typ == "choice" else (top == "yes" if typ == "yesno" else int(top))
    if typ == "score" and "value" not in kw:
        kw["value"] = sum(int(k) * v for k, v in probs.items())
    return Prediction(
        system="t",
        state_id=state_id,
        question_id=qid,
        type=typ,
        answer=answer,
        probabilities=probs,
        **kw,
    )


def test_auroc_and_ap():
    s = np.array([0.9, 0.8, 0.3, 0.2])
    y = np.array([True, False, True, False])
    assert metrics.auroc(s, y) == pytest.approx(0.75)
    assert metrics.auroc(np.array([0.5, 0.5]), np.array([True, False])) == 0.5
    assert metrics.auroc(s, np.zeros(4, bool)) is None
    # ranking: +(0.9) -(0.8) +(0.3): precision at hits 1/1, 2/3 -> AP = (1 + 2/3) / 2
    assert metrics.average_precision(s, y) == pytest.approx((1 + 2 / 3) / 2)


def test_macro_f1_counts_rare_classes_equally():
    gold = ["a"] * 9 + ["b"]
    assert metrics.macro_f1(gold, ["a"] * 10) == pytest.approx((2 * 9 / (2 * 9 + 1) + 0) / 2)
    assert metrics.macro_f1(gold, gold) == 1.0
    assert metrics.macro_f1(["a", "b"], [None, "b"]) == pytest.approx(0.5)


def test_reliability_perfectly_calibrated():
    conf = np.array([0.95] * 20 + [0.55] * 20)
    correct = np.array([1] * 19 + [0] + [1] * 11 + [0] * 9, dtype=float)
    ece, bins = metrics.reliability(conf, correct)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert sum(b.count for b in bins) == 40 and len(bins) == 10
    assert bins[9].count == 20 and bins[5].count == 20


def test_compute_basic_numbers():
    records = [rec("1", {"q": "a", "y": True, "s": 2}), rec("2", {"q": "b", "y": False, "s": 0})]
    preds = [
        pred("1", "q", {"a": 0.7, "b": 0.2, "c": 0.1}),
        pred("1", "y", {"yes": 0.8, "no": 0.2}),
        pred("1", "s", {"0": 0.1, "1": 0.2, "2": 0.7}),
        pred("2", "q", {"a": 0.6, "b": 0.3, "c": 0.1}),  # wrong
        pred("2", "y", {"yes": 0.1, "no": 0.9}),
        # "s" for state 2 is missing -> counted as an error
    ]
    m = metrics.compute(preds, records, system="t")
    Metrics.model_validate(m.model_dump())
    assert m.n_questions == 6 and m.overall.n_errors == 1
    assert m.overall.accuracy == pytest.approx(4 / 6)
    q = m.by_question["q"]
    assert q.accuracy == 0.5
    assert q.brier == pytest.approx(((0.3**2 + 0.2**2 + 0.1**2) + (0.6**2 + 0.7**2 + 0.1**2)) / 2)
    assert q.log_loss == pytest.approx((-math.log(0.7) - math.log(0.3)) / 2)
    assert m.by_question["s"].mae == pytest.approx(abs(0.2 + 1.4 - 2))
    assert m.extra["auroc__y"] == 1.0
    assert m.extra["positive_rate__y"] == 0.5
    assert m.extra["binary_brier__y"] == pytest.approx((0.2**2 + 0.1**2) / 2)


def test_probabilities_are_renormalised_and_log_loss_floored():
    records = [rec("1", {"q": "c", "y": True, "s": 0})]
    preds = [pred("1", "q", {"a": 1.0, "b": 0.01, "c": 0.0}, answer="a")]  # sums to 1.01
    m = metrics.compute(preds, records, system="t")
    assert m.by_question["q"].log_loss == pytest.approx(-math.log(metrics.LOG_LOSS_FLOOR))
    a, b = 1.0 / 1.01, 0.01 / 1.01
    assert m.by_question["q"].brier == pytest.approx(a**2 + b**2 + 1.0)


def test_order_sensitivity_from_permutations():
    records = [rec("1", {"q": "a", "y": True, "s": 0})]
    perms = [{"a": 0.6, "b": 0.3, "c": 0.1}, {"a": 0.2, "b": 0.7, "c": 0.1}]
    preds = [pred("1", "q", {"a": 0.4, "b": 0.5, "c": 0.1}, meta={"per_permutation": perms})]
    m = metrics.compute(preds, records, system="t")
    assert m.extra["order_tv__q"] == pytest.approx(0.4)
    assert m.extra["order_flip_rate__q"] == 1.0
    assert m.extra["first_perm_accuracy__q"] == 1.0  # the first order alone was right
    assert m.by_question["q"].accuracy == 0.0  # the average wasn't
    assert m.order_sensitivity == pytest.approx(0.4)


def test_contract_examples_score():
    records = list(read_jsonl(f"{EXAMPLES}/data.example.jsonl", DataRecord))
    preds = list(read_jsonl(f"{EXAMPLES}/predictions.example.jsonl", Prediction))
    out = metrics.compute_all(preds, records, system="x")
    assert "all" in out
    for m in out.values():
        Metrics.model_validate(m.model_dump())


def test_coverage_curve():
    records = [rec(str(i), {"q": "a", "y": True, "s": 0}) for i in range(4)]
    preds = [
        pred("0", "q", {"a": 0.9, "b": 0.05, "c": 0.05}),  # right, most confident
        pred("1", "q", {"a": 0.8, "b": 0.1, "c": 0.1}),  # right
        pred("2", "q", {"a": 0.1, "b": 0.7, "c": 0.2}),  # wrong
        pred("3", "q", {"a": 0.2, "b": 0.5, "c": 0.3}),  # wrong, least confident
    ]
    rows = [r for r in metrics.join(preds, records) if r.question_id == "q"]
    c = metrics.coverage_curve(rows, target=0.9)
    acc = {p["coverage"]: p["accuracy"] for p in c["points"]}
    assert acc[1.0] == 0.5 and acc[0.5] == 1.0
    assert c["coverage_at_target"] == 0.5


def test_coverage_curve_ignores_input_order_within_ties():
    records = [rec(str(i), {"q": "a", "y": True, "s": 0}) for i in range(4)]
    right = {"a": 0.9, "b": 0.05, "c": 0.05}
    wrong = {"a": 0.05, "b": 0.9, "c": 0.05}  # same confidence (0.9), wrong answer
    for order in (["r", "w", "r", "w"], ["w", "w", "r", "r"]):
        preds = [pred(str(i), "q", right if o == "r" else wrong) for i, o in enumerate(order)]
        rows = [r for r in metrics.join(preds, records) if r.question_id == "q"]
        c = metrics.coverage_curve(rows)
        assert {p["coverage"]: p["accuracy"] for p in c["points"]}[0.5] == 0.5


def _scored(state, qid, correct):
    return metrics.Scored(
        state_id=state, dataset="d", question_id=qid, type="choice", gold="a",
        answer="a" if correct else "b", probs=None, value=None, latency_ms=None,
        per_permutation=None,
    )  # fmt: skip


def test_paired_bootstrap_diff_and_interval():
    # 40 states x 2 questions; b fixes q1 on half the states, both identical on q2
    a = [_scored(str(s), q, q == "q2" or s % 2 == 0) for s in range(40) for q in ("q1", "q2")]
    b = [_scored(str(s), q, True) for s in range(40) for q in ("q1", "q2")]
    out = metrics.paired_bootstrap(a, b, n=500)
    assert out["q1"]["diff"] == pytest.approx(0.5) and out["q1"]["n"] == 40
    assert 0.3 < out["q1"]["lo"] < 0.5 < out["q1"]["hi"] < 0.7
    assert out["q2"] == {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 40}
    assert out["all"]["diff"] == pytest.approx(0.25) and out["all"]["n"] == 80
    # only shared decisions count; same seed, same interval
    assert metrics.paired_bootstrap(a, b[:10], n=500)["all"]["n"] == 10
    assert metrics.paired_bootstrap(a, b, n=500) == out
    assert metrics.paired_bootstrap(a, [], n=10) == {}


def test_paired_bootstrap_with_brier():
    def row(state, p_yes, gold):
        return metrics.Scored(
            state_id=state, dataset="d", question_id="y", type="yesno", gold=gold,
            answer="yes" if p_yes >= 0.5 else "no", probs={"yes": p_yes, "no": 1 - p_yes},
            value=None, latency_ms=None, per_permutation=None,
        )  # fmt: skip

    a = [row(str(s), 0.5, "yes") for s in range(20)]  # Brier 0.5 each
    b = [row(str(s), 1.0, "yes") for s in range(20)]  # Brier 0 each
    assert metrics.brier_row(a[0]) == pytest.approx(0.5)
    out = metrics.paired_bootstrap(a, b, n=200, score=metrics.brier_row)
    assert out["y"]["diff"] == pytest.approx(-0.5) and out["y"]["hi"] == pytest.approx(-0.5)


def _yn(state, p_yes, gold):
    return metrics.Scored(
        state_id=state, dataset="d", question_id="y", type="yesno", gold=gold,
        answer="yes" if p_yes >= 0.5 else "no", probs={"yes": p_yes, "no": 1 - p_yes},
        value=None, latency_ms=None, per_permutation=None,
    )  # fmt: skip


def test_balanced_accuracy_always_no_is_half():
    rows = [_yn(str(i), 0.1, "yes" if i < 10 else "no") for i in range(100)]
    assert metrics.balanced_accuracy(rows) == 0.5
    assert sum(r.correct for r in rows) / len(rows) == 0.9


def test_paired_bootstrap_auc():
    gold = ["yes" if i % 5 == 0 else "no" for i in range(100)]
    perfect = [_yn(str(i), 0.9 if g == "yes" else 0.1, g) for i, g in enumerate(gold)]
    flat = [_yn(str(i), 0.3, g) for i, g in enumerate(gold)]  # all tied: AUC 0.5
    out = metrics.paired_bootstrap_auc(flat, perfect, n=200)
    assert out["a"]["value"] == 0.5 and out["b"]["value"] == 1.0
    assert out["diff"]["value"] == 0.5 and out["diff"]["lo"] == pytest.approx(0.5)
    one = metrics.paired_bootstrap_auc(perfect, perfect, n=50)
    assert one["diff"]["value"] == 0.0 and one["b"]["n"] == 100

import json
import math
import random

import numpy as np
import pytest

from snapcall import calibrate
from snapcall.schema import DataRecord, Prediction, answer_keys, read_jsonl
from tests.test_schema import EXAMPLES


def softmax(z):
    z = np.exp(z - z.max())
    return z / z.sum()


def make_items(t_true: float, n: int = 3000, k: int = 4, seed: int = 0):
    """Items whose gold is drawn from softmax(logits / t_true): the right T is t_true."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        logits = rng.normal(0, 3, size=k)
        gold = int(rng.choice(k, p=softmax(logits / t_true)))
        items.append(calibrate.Item("choice", np.log(softmax(logits))[None], gold))
    return items


@pytest.mark.parametrize("t_true", [0.5, 1.0, 2.5])
def test_fit_recovers_the_true_temperature(t_true):
    t = calibrate.fit_temperature(make_items(t_true))
    assert t == pytest.approx(t_true, rel=0.15)


def test_fitted_temperature_never_worse_than_one():
    items = make_items(2.0)
    t = calibrate.fit_temperature(items)
    assert calibrate.nll(items, t) <= calibrate.nll(items, 1.0)
    assert calibrate.ece(items, t) < calibrate.ece(items, 1.0)


def test_tempering_matches_scaling_the_logits():
    logits = np.array([2.0, -1.0, 0.5])
    got = calibrate.tempered(np.log(softmax(logits))[None], 3.0)
    assert got == pytest.approx(softmax(logits / 3.0))


def test_perfect_validation_stops_at_the_bound():
    items = [calibrate.Item("yesno", np.log(np.array([[0.9, 0.1]])), 0)] * 50
    assert calibrate.fit_temperature(items) == pytest.approx(calibrate.T_MIN, rel=1e-3)


def test_yesno_bias_fixes_a_shifted_threshold():
    """A model that ranks well but says yes too often: a temperature alone can't fix it,
    a bias can (the wants_to_leave case)."""
    rng = np.random.default_rng(0)
    items = []
    for _ in range(2000):
        gold_yes = rng.random() < 0.15
        diff = rng.normal(2.0 if gold_yes else -1.0, 1.0) + 2.5  # +2.5: the yes-bias
        lp = np.log(softmax(np.array([diff, 0.0])))
        items.append(calibrate.Item("yesno", lp[None], 0 if gold_yes else 1, "ds", "q"))
    t, biases = calibrate.fit_temperature_and_biases(items)
    assert set(biases) == {"ds:q"}  # 2,000 items, ~300 yes: its own bias
    b = biases["ds:q"]
    assert b < -1.5
    assert calibrate.accuracy(items, t, b) > calibrate.accuracy(items, 1.0) + 0.1
    assert calibrate.nll(items, t, b) < calibrate.nll(items, calibrate.fit_temperature(items))


def shifted_yesno(dataset: str, question: str, n: int, shift: float, rng) -> list:
    """Yes/no items from a model that ranks well, with its threshold off by ``shift``."""
    items = []
    for _ in range(n):
        gold_yes = rng.random() < 0.2
        diff = rng.normal(2.0 if gold_yes else -1.0, 1.0) + shift
        lp = np.log(softmax(np.array([diff, 0.0])))
        items.append(calibrate.Item("yesno", lp[None], 0 if gold_yes else 1, dataset, question))
    return items


def test_each_dataset_gets_its_own_bias():
    """Two datasets with opposite threshold errors: one shared bias can't fix both; per
    dataset (Decision 14), each gets its own sign."""
    rng = np.random.default_rng(1)
    items = shifted_yesno("a", "q1", 600, +2.5, rng) + shifted_yesno("b", "q2", 600, -2.5, rng)
    t, biases = calibrate.fit_temperature_and_biases(items)
    assert biases["a:q1"] < -1.0 < 1.0 < biases["b:q2"]
    stack = calibrate._Stack.of(items)
    per_group = calibrate.item_biases(items, calibrate.bias_groups(items), biases)
    shared = calibrate.fit_bias(items, t)
    assert stack.nll(t, per_group) < stack.nll(t, shared)


def test_small_questions_share_their_dataset_bias_and_tiny_datasets_get_none():
    rng = np.random.default_rng(2)
    items = (
        shifted_yesno("big", "q", 300, 0.0, rng)  # its own bias
        + shifted_yesno("mid", "x", 40, 0.0, rng)  # too small alone ...
        + shifted_yesno("mid", "y", 40, 0.0, rng)  # ... pooled: 80 >= MIN_DATASET_ITEMS
        + shifted_yesno("tiny", "z", 10, 0.0, rng)  # no bias at all
    )
    assert set(calibrate.bias_groups(items)) == {"big:q", "mid:*"}


def test_bias_lookup_order():
    bias = {"d:q": 1.0, "d:*": 2.0, "yesno": 3.0}
    assert calibrate.bias_for(bias, "d", "q", "yesno") == 1.0
    assert calibrate.bias_for(bias, "d", "other", "yesno") == 2.0
    assert calibrate.bias_for(bias, "ood", "q", "yesno") == 3.0  # old-style shared bias
    assert calibrate.bias_for({"d:q": 1.0}, "ood", "q", "yesno") == 0.0  # OOD: none


def test_apply_bias_moves_p_yes(records):
    preds = [p for p in fake_predictions(records) if p.type == "yesno"]
    for bias in ({"yesno": -2.0}, {f"{records[0].dataset}:*": -2.0}):
        lowered = calibrate.apply(preds, records, {}, bias=bias)
        for p, q in zip(preds, lowered, strict=True):
            assert q.probabilities["yes"] < p.probabilities["yes"]
            assert q.meta["bias"] == -2.0
    untouched = calibrate.apply(preds, records, {}, bias={"other-dataset:*": -2.0})
    for p, q in zip(preds, untouched, strict=True):
        assert q.probabilities["yes"] == pytest.approx(p.probabilities["yes"])


def test_fit_with_nothing_to_fit_is_one():
    assert calibrate.fit_temperature([]) == 1.0


# --------------------------------------------------------------------------------------
# On contract-shaped predictions
# --------------------------------------------------------------------------------------


@pytest.fixture
def records():
    return list(read_jsonl(EXAMPLES / "data.example.jsonl", DataRecord))


def fake_predictions(records, seed=0):
    """Over-confident predictions with two permutations each, as the eval harness writes."""
    rng = random.Random(seed)
    preds = []
    for r in records:
        for qid, q in r.questions.items():
            keys = answer_keys(q)
            perms = []
            for _ in range(2):
                w = [rng.random() ** 8 for _ in keys]
                perms.append({k: x / sum(w) for k, x in zip(keys, w, strict=True)})
            mean = {k: sum(p[k] for p in perms) / 2 for k in keys}
            top = max(mean, key=mean.get)
            answer = top if q.type == "choice" else int(top) if q.type == "score" else top == "yes"
            preds.append(
                Prediction(
                    system="zs",
                    state_id=r.state_id,
                    question_id=qid,
                    type=q.type,
                    answer=answer,
                    probabilities=mean,
                    value=sum(int(k) * v for k, v in mean.items()) if q.type == "score" else None,
                    meta={"per_permutation": perms},
                )
            )
    return preds


def test_apply_at_t1_changes_nothing_but_the_system(records):
    preds = fake_predictions(records)
    out = calibrate.apply(preds, records, {}, system="cal")
    for p, q in zip(preds, out, strict=True):
        assert q.system == "cal"
        assert q.answer == p.answer
        for k in p.probabilities:
            assert q.probabilities[k] == pytest.approx(p.probabilities[k])


def test_apply_softens_and_stays_valid(records):
    preds = fake_predictions(records)
    out = calibrate.apply(preds, records, {"choice": 3.0, "score": 3.0, "yesno": 3.0})
    for p, q in zip(preds, out, strict=True):
        Prediction.model_validate(q.model_dump())
        for before, after in zip(p.meta["per_permutation"], q.meta["per_permutation"], strict=True):
            assert max(after.values()) <= max(before.values()) + 1e-9
        assert q.meta["temperature"] == 3.0
        if q.type == "score":
            assert q.value is not None


def test_errored_predictions_pass_through(records):
    r = records[0]
    qid, q = next(iter(r.questions.items()))
    bad = Prediction(
        system="zs", state_id=r.state_id, question_id=qid, type=q.type, answer=None,
        probabilities=None, error="boom",
    )  # fmt: skip
    assert calibrate.items_from([bad], records) == []
    (out,) = calibrate.apply([bad], records, {"choice": 2.0}, system="cal")
    assert out.error == "boom" and out.system == "cal"


def test_fit_reports_every_type(records):
    temps, bias, report = calibrate.fit(fake_predictions(records), records)
    assert all(":" in k for k in bias)  # "<dataset>:<question>" or "<dataset>:*"
    types = {q.type for r in records for q in r.questions.values()}
    assert set(temps) == types == set(report) - {"yesno_bias"}
    json.dumps({"temperatures": temps, "bias": bias, "report": report})  # plain JSON types
    for qtype in types:
        row = report[qtype]
        assert row["nll_after"] <= row["nll_before"] + 1e-9
        assert math.isfinite(row["temperature"]) and isinstance(row["at_bound"], bool)


def test_cli_fit_then_apply(records, tmp_path):
    from snapcall.schema import write_jsonl

    write_jsonl(tmp_path / "p.jsonl", fake_predictions(records))
    data = EXAMPLES / "data.example.jsonl"
    temps = tmp_path / "t.json"
    args = ["--predictions", str(tmp_path / "p.jsonl"), "--data", str(data)]
    assert calibrate.main(["fit", *args, "--out", str(temps)]) == 0
    assert set(json.loads(temps.read_text())["temperatures"]) <= {"choice", "score", "yesno"}
    out = tmp_path / "cal.jsonl"
    assert calibrate.main(["apply", *args, "--temperatures", str(temps), "--out", str(out)]) == 0
    assert len(list(read_jsonl(out, Prediction))) == len(fake_predictions(records))

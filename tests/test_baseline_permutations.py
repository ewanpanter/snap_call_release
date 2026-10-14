"""Option-order permutations for the API baselines (baselines.common.with_permutations)."""

import pytest

from baselines.common import (
    FatalError,
    StateResult,
    option_shifts,
    rotated,
    with_permutations,
)
from snapcall.schema import DataRecord, Prediction


def record():
    return DataRecord.model_validate(
        {
            "state_id": "s1",
            "dataset": "d",
            "split": "test",
            "state": "text",
            "questions": {
                "q": {
                    "type": "choice",
                    "question": "?",
                    "options": {"a": "A", "b": "B", "c": "C", "d": "D"},
                },
                "s": {"type": "score", "question": "?", "levels": ["lo", "hi"]},
                "y": {"type": "yesno", "question": "?"},
            },
            "gold": {"q": "a", "s": 1, "y": True},
        }
    )


def fake_api(seen, prefer_first=0.7):
    """Answers every choice question with whatever option is shown FIRST (a pure position
    bias), plus fixed score / yes-no answers. Records the option order it was shown."""

    def decide(rec):
        preds = []
        for qid, q in rec.questions.items():
            if q.type == "choice":
                keys = list(q.options)
                seen.append(keys)
                rest = (1 - prefer_first) / (len(keys) - 1)
                probs = {k: (prefer_first if i == 0 else rest) for i, k in enumerate(keys)}
                preds.append(
                    Prediction(
                        system="api",
                        state_id=rec.state_id,
                        question_id=qid,
                        type="choice",
                        answer=keys[0],
                        probabilities=probs,
                        confidence=0.9,
                    )
                )
            elif q.type == "score":
                preds.append(
                    Prediction(
                        system="api",
                        state_id=rec.state_id,
                        question_id=qid,
                        type="score",
                        answer=1,
                        value=0.8,
                        probabilities={"0": 0.2, "1": 0.8},
                    )
                )
            else:
                preds.append(
                    Prediction(
                        system="api",
                        state_id=rec.state_id,
                        question_id=qid,
                        type="yesno",
                        answer=True,
                        probabilities={"yes": 0.9, "no": 0.1},
                    )
                )
        return StateResult(
            state_id=rec.state_id,
            predictions=preds,
            latency_ms=100.0,
            input_tokens=10,
            attempts=1,
            usd_cost=0.001,
        )

    return decide


def test_shifts_match_snapcall():
    assert option_shifts(4, 4) == [0, 1, 2, 3]
    assert option_shifts(10, 4) == [0, 2, 5, 7]
    assert option_shifts(3, 4) == [0, 1, 2]
    assert option_shifts(26, 1) == [0]


def test_rotated_keeps_only_choice_questions_and_their_keys():
    r = rotated(record(), 1, 4)
    assert list(r.questions) == ["q"] and r.gold == {"q": "a"}
    assert list(r.questions["q"].options) == ["b", "c", "d", "a"]
    assert rotated(record(), 4, 4) is None  # only 4 shifts for 4 options


def test_permutations_cancel_a_pure_position_bias():
    seen = []
    result = with_permutations(fake_api(seen), 4)(record())
    assert seen == [list("abcd"), list("bcda"), list("cdab"), list("dabc")]
    by_q = {p.question_id: p for p in result.predictions}
    q = by_q["q"]
    assert q.probabilities == pytest.approx({k: 0.25 for k in "abcd"})  # bias averaged away
    assert q.meta["first_perm_answer"] == "a" and len(q.meta["per_permutation"]) == 4
    assert q.meta["per_permutation"][1] == pytest.approx({"a": 0.1, "b": 0.7, "c": 0.1, "d": 0.1})
    # score and yes/no come from the original request, untouched
    assert (
        by_q["s"].answer == 1
        and by_q["y"].answer is True
        and "per_permutation" not in by_q["y"].meta
    )
    # usage is summed over the 4 requests
    assert result.attempts == 4 and result.requests == 4 and result.input_tokens == 40
    assert result.latency_ms == 400.0 and result.usd_cost == pytest.approx(0.004)


def test_one_permutation_is_the_plain_client():
    decide = fake_api([])
    assert with_permutations(decide, 1) is decide


def test_a_failed_extra_order_keeps_the_original_answer():
    seen = []
    inner = fake_api(seen)
    calls = {"n": 0}

    def flaky(rec):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("timeout")
        return inner(rec)

    result = with_permutations(flaky, 4)(record())
    q = next(p for p in result.predictions if p.question_id == "q")
    assert q.meta["n_permutations"] == 3 and result.attempts == 3


def test_a_fatal_error_still_stops_the_run():
    inner = fake_api([])
    calls = {"n": 0}

    def dead_key(rec):
        calls["n"] += 1
        if calls["n"] == 2:
            raise FatalError("401")
        return inner(rec)

    with pytest.raises(FatalError):
        with_permutations(dead_key, 4)(record())


def test_permuted_states_are_not_counted_as_retried():
    from baselines.common import summarise

    result = with_permutations(fake_api([]), 4)(record())
    assert summarise([result])["retried_states"] == 0


def test_merge_repeats_counts_changes():
    from baselines.common import merge_repeats

    def result(answer, p):
        pred = Prediction(
            system="api",
            state_id="s1",
            question_id="q",
            type="choice",
            answer=answer,
            probabilities={"a": p, "b": 1 - p},
        )
        return StateResult(state_id="s1", predictions=[pred], latency_ms=100.0, input_tokens=5)

    merged, summary = merge_repeats([[result("a", 0.6)], [result("b", 0.4)]])
    q = summary["per_question"]["q"]
    assert (q["n"], q["answer_changed"], q["probs_changed"]) == (1, 1, 1)
    assert q["max_abs_prob_change"] == pytest.approx(0.2)
    (r,) = merged
    assert r.predictions[0].answer == "a"  # pass 1 is the scored answer
    assert r.predictions[0].meta["repeats"] == [
        {"answer": "b", "probabilities": {"a": 0.4, "b": 0.6}}
    ]
    assert (r.input_tokens, r.requests, r.latency_ms) == (10, 2, 100.0)

    _, same = merge_repeats([[result("a", 0.6)], [result("a", 0.6)]])
    assert same["per_question"]["q"]["answer_changed"] == 0

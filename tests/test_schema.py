import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from snapcall.schema import (
    CONTRACT_MODELS,
    ChoiceAnswer,
    Cost,
    DataRecord,
    DecideRequest,
    DecideResponse,
    Metrics,
    Prediction,
    ScoreAnswer,
    from_typesafe_request,
    from_typesafe_response,
    json_schemas,
    predictions_from_response,
    read_jsonl,
    state_text,
    to_typesafe_request,
    to_typesafe_response,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "contract" / "examples"
SCHEMAS = ROOT / "contract" / "schemas"

JSON_EXAMPLES = {
    "decide_request": "decide_request.example.json",
    "decide_response": "decide_response.example.json",
    "metrics": "metrics.example.json",
    "cost": "cost.example.json",
}
JSONL_EXAMPLES = {
    "data_record": "data.example.jsonl",
    "prediction": "predictions.example.jsonl",
}


def _load(name):
    return json.loads((EXAMPLES / name).read_text())


def _lines(name):
    return [json.loads(line) for line in (EXAMPLES / name).read_text().splitlines() if line]


def test_committed_json_schemas_match_models():
    """Regenerate with: uv run python -m snapcall.schema contract/schemas"""
    for name, schema in json_schemas().items():
        committed = json.loads((SCHEMAS / f"{name}.schema.json").read_text())
        assert committed == schema, f"{name}.schema.json is stale; regenerate it"
    assert {p.name.removesuffix(".schema.json") for p in SCHEMAS.glob("*.schema.json")} == set(
        CONTRACT_MODELS
    )


def test_every_contract_model_has_an_example():
    assert set(JSON_EXAMPLES) | set(JSONL_EXAMPLES) == set(CONTRACT_MODELS)


@pytest.mark.parametrize("name,file", JSON_EXAMPLES.items())
def test_json_examples_validate(name, file):
    doc = _load(file)
    CONTRACT_MODELS[name].model_validate(doc)
    jsonschema.validate(doc, json.loads((SCHEMAS / f"{name}.schema.json").read_text()))


@pytest.mark.parametrize("name,file", JSONL_EXAMPLES.items())
def test_jsonl_examples_validate(name, file):
    rows = list(read_jsonl(EXAMPLES / file, CONTRACT_MODELS[name]))
    assert rows
    schema = json.loads((SCHEMAS / f"{name}.schema.json").read_text())
    for doc in _lines(file):
        jsonschema.validate(doc, schema)


def test_example_response_matches_example_request():
    req = DecideRequest.model_validate(_load("decide_request.example.json"))
    resp = DecideResponse.model_validate(_load("decide_response.example.json"))
    resp.check_against(req)


def test_jsonl_round_trip(tmp_path):
    rows = list(read_jsonl(EXAMPLES / "predictions.example.jsonl", Prediction))
    out = tmp_path / "p.jsonl"
    assert write_jsonl(out, rows) == len(rows)
    assert list(read_jsonl(out, Prediction)) == rows


def test_record_to_request_and_predictions():
    rec = next(read_jsonl(EXAMPLES / "data.example.jsonl", DataRecord))
    req = rec.to_request(debias="off")
    assert list(req.questions) == list(rec.questions)
    resp = DecideResponse.model_validate(
        {
            "model": "fake",
            "latency_ms": 10,
            "answers": {
                "queue": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {
                        "billing": 0.7,
                        "technical": 0.1,
                        "sales": 0.1,
                        "returns": 0.1,
                    },
                },
                "type": {
                    "type": "choice",
                    "choice": "request",
                    "probabilities": {
                        "incident": 0.1,
                        "request": 0.7,
                        "question": 0.1,
                        "problem": 0.1,
                    },
                },
                "priority": {
                    "type": "score",
                    "value": 2.0,
                    "level": 2,
                    "probabilities": {"0": 0.0, "1": 0.2, "2": 0.6, "3": 0.2},
                },
                "is_billing": {"type": "yesno", "p_yes": 0.8},
            },
        }
    )
    resp.check_against(req)
    preds = predictions_from_response(resp, system="fake", state_id=rec.state_id)
    assert [p.question_id for p in preds] == list(rec.questions)
    assert preds[2].answer == 2 and preds[2].value == 2.0
    assert preds[3].answer is True and preds[3].probabilities == pytest.approx(
        {"yes": 0.8, "no": 0.2}
    )


def test_state_text():
    assert state_text("hi") == "hi"
    assert state_text({"b": 1, "a": "é"}) == '{"a": "é", "b": 1}'


# ---- TypeSafe adapter ----------------------------------------------------------------


def test_typesafe_request_round_trip():
    body = _load("typesafe_request.example.json")
    req = from_typesafe_request(body)
    assert [q.type for q in req.questions.values()] == ["yesno", "choice", "score"]
    back = to_typesafe_request(req, model=body["model"])
    assert back == body


def test_typesafe_response_parses_and_renormalises():
    req = from_typesafe_request(_load("typesafe_request.example.json"))
    resp = from_typesafe_response(_load("typesafe_response.example.json"), req, latency_ms=120.0)
    resp.check_against(req)
    assert resp.answers["is_bug"].p_yes == 0.96
    team = resp.answers["team"]
    assert isinstance(team, ChoiceAnswer) and team.choice == "payments"
    assert sum(team.probabilities.values()) == pytest.approx(1.0)
    urgency = resp.answers["urgency"]
    assert isinstance(urgency, ScoreAnswer) and urgency.level == 2 and urgency.value == 1.99
    assert resp.usage.input_tokens == 476


def test_snapcall_response_in_typesafe_shape():
    req = DecideRequest.model_validate(_load("decide_request.example.json"))
    resp = DecideResponse.model_validate(_load("decide_response.example.json"))
    body = to_typesafe_response(resp, req)
    assert body["answers"]["is_billing"] == {"type": "noul", "noul": 0.94}
    assert body["answers"]["priority"]["legend"]["3"] == "critical"
    # and it parses back through the same adapter the baseline uses
    assert from_typesafe_response(body, req).answers["queue"].choice == "billing"


# ---- things the contract must reject ---------------------------------------------------


def _record(**overrides):
    rec = _lines("data.example.jsonl")[0]
    rec.update(overrides)
    return rec


@pytest.mark.parametrize(
    "gold",
    [
        {"queue": "nope", "type": "request", "priority": 2, "is_billing": True},  # bad key
        {"queue": "billing", "type": "request", "priority": 4, "is_billing": True},  # range
        {"queue": "billing", "type": "request", "priority": True, "is_billing": True},  # bool
        {"queue": "billing", "type": "request", "priority": 2, "is_billing": "yes"},  # str
        {"queue": "billing", "type": "request", "priority": 2},  # missing
    ],
)
def test_bad_gold_rejected(gold):
    with pytest.raises(ValidationError):
        DataRecord.model_validate(_record(gold=gold))


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        DataRecord.model_validate(_record(surprise=1))


def test_bad_question_id_rejected():
    with pytest.raises(ValidationError):
        DecideRequest.model_validate(
            {"state": "x", "questions": {"has space": {"type": "yesno", "question": "q"}}}
        )


def test_single_option_choice_rejected():
    with pytest.raises(ValidationError):
        DecideRequest.model_validate(
            {
                "state": "x",
                "questions": {"q": {"type": "choice", "question": "q", "options": {"a": "A"}}},
            }
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"probabilities": {"billing": 0.5, "technical": 0.1}},  # doesn't sum to 1
        {"answer": "sales"},  # not in probabilities
        {"answer": None},  # missing without error
        {"type": "yesno"},  # wrong answer type / keys
    ],
)
def test_bad_prediction_rejected(overrides):
    row = _lines("predictions.example.jsonl")[0]
    row["probabilities"] = {"billing": 0.9, "technical": 0.1}
    row.update(overrides)
    with pytest.raises(ValidationError):
        Prediction.model_validate(row)


def test_score_prediction_needs_value():
    row = _lines("predictions.example.jsonl")[2]
    row["value"] = None
    with pytest.raises(ValidationError):
        Prediction.model_validate(row)


def test_response_mismatch_detected():
    req = DecideRequest.model_validate(_load("decide_request.example.json"))
    doc = _load("decide_response.example.json")
    doc["answers"]["queue"]["probabilities"] = {"billing": 0.9, "other": 0.1}
    resp = DecideResponse.model_validate(doc)
    with pytest.raises(ValueError):
        resp.check_against(req)


def test_cost_totals_checked():
    doc = _load("cost.example.json")
    assert Cost.model_validate(doc).cost_per_1k_decisions == pytest.approx(0.08333333 / 3.2)
    doc["total"] = 5.0
    with pytest.raises(ValidationError):
        Cost.model_validate(doc)


def test_metrics_rejects_unknown_type():
    doc = _load("metrics.example.json")
    doc["by_type"]["regression"] = {"n": 1}
    with pytest.raises(ValidationError):
        Metrics.model_validate(doc)

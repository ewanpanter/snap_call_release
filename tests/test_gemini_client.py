"""Gemini baseline client, with a fake google-genai client (no SDK or network needed)."""

import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import jsonschema
import pytest

from baselines import gemini_client as gc
from baselines.common import load_records
from snapcall.schema import Cost, Prediction, ScoreQuestion, YesNoQuestion, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "contract" / "examples" / "data.example.jsonl"


class ApiError(Exception):
    """Shaped like google.genai.errors.APIError: has an int ``code``."""

    def __init__(self, code, message):
        super().__init__(f"{code} {message}")
        self.code = code


def good_answer(q) -> dict:
    keys = gc.answer_space(q)
    probs = {k: 0.0 for k in keys}
    probs[keys[0]], probs[keys[1]] = 0.6, 0.4
    return {"answer": keys[0], "probabilities": probs}


def good_text(record) -> str:
    return json.dumps({qid: good_answer(q) for qid, q in record.questions.items()})


def fake_logprobs(text: str):
    """Split the text into 3-char 'tokens', each with p=0.9 and one alternative."""
    toks = [text[i : i + 3] for i in range(0, len(text), 3)]
    lp = math.log(0.9)
    return NS(
        chosen_candidates=[NS(token=t, log_probability=lp) for t in toks],
        top_candidates=[
            NS(
                candidates=[
                    NS(token=t, log_probability=lp),
                    NS(token="zz", log_probability=math.log(0.1)),
                ]
            )
            for t in toks
        ],
    )


def fake_response(text, finish="STOP", logprobs=None):
    return NS(
        text=text,
        model_version="gemini-test-001",
        candidates=[NS(finish_reason=NS(value=finish), logprobs_result=logprobs)],
        usage_metadata=NS(
            prompt_token_count=500, candidates_token_count=120, thoughts_token_count=0
        ),
    )


class FakeGenai:
    """``client.models.generate_content``. ``script`` = exceptions to raise first, in order."""

    def __init__(self, records, script=(), logprobs=False, finish="STOP"):
        self.by_prompt = {gc.build_prompt(r): r for r in records}
        self.script = list(script)
        self.calls: list[dict] = []
        self.logprobs, self.finish = logprobs, finish
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "config": config})
        if self.script:
            raise self.script.pop(0)
        text = good_text(self.by_prompt[contents])
        lp = fake_logprobs(text) if self.logprobs and config.get("response_logprobs") else None
        return fake_response(text, self.finish, lp)


@pytest.fixture
def records():
    return load_records(DATA)


def client(fake, **kw):
    return gc.GeminiClient(fake, run_id="g-1", sleep=lambda s: None, **kw)


# --------------------------------------------------------------------------------------


def test_schema_and_prompt_cover_every_question(records):
    r = records[0]
    schema = gc.build_schema(r)
    assert schema["required"] == list(r.questions)
    prio = schema["properties"]["priority"]["properties"]
    assert prio["answer"]["enum"] == r.questions["priority"].levels  # levels answered by text
    assert schema["properties"]["is_billing"]["properties"]["answer"]["enum"] == ["yes", "no"]
    # The schema accepts a well-formed answer.
    jsonschema.validate(json.loads(good_text(r)), schema)
    prompt = gc.build_prompt(r)
    for qid, q in r.questions.items():
        assert f"[{qid}]" in prompt and q.question in prompt


def test_ambiguous_levels_fall_back_to_indices():
    q = ScoreQuestion(question="How big?", levels=["same", "same", "x" * 80])
    assert gc.answer_space(q) == ["0", "1", "2"]


def test_decide_produces_valid_predictions(records):
    fake = FakeGenai(records)
    r = records[0]
    res = client(fake).decide(r)
    assert res.error is None and res.input_tokens == 500 and res.model_version == "gemini-test-001"
    by_q = {p.question_id: p for p in res.predictions}
    assert by_q["priority"].answer == 0 and by_q["priority"].value == pytest.approx(0.4)
    assert by_q["is_billing"].answer is True
    assert by_q["is_billing"].probabilities == {"yes": 0.6, "no": 0.4}
    assert all(p.meta["prob_source"] == "verbalised" for p in res.predictions)
    cfg = fake.calls[0]["config"]
    assert cfg["temperature"] == 0 and cfg["thinking_config"] == {"thinking_budget": 0}
    assert cfg["response_logprobs"] is True


def test_logprobs_meta_points_at_the_answer_token(records):
    fake = FakeGenai(records, logprobs=True)
    res = client(fake).decide(records[0])
    p = next(p for p in res.predictions if p.question_id == "queue")
    assert p.meta["prob_source"] == "verbalised+logprobs"
    assert p.meta["answer_token_prob"] == pytest.approx(0.9)
    # The located token overlaps the start of the answer value.
    answer = list(records[0].questions["queue"].options)[0]
    text = good_text(records[0])
    start = text.index('"queue": {"answer": "') + len('"queue": {"answer": "')
    assert text[start : start + len(answer)] == answer
    assert p.meta["answer_token"] and p.meta["answer_token_top"]["zz"] == pytest.approx(0.1)


def test_stated_probabilities_are_cleaned(records):
    r = records[0]
    data = json.loads(good_text(r))
    keys = list(r.questions["queue"].options)
    data["queue"] = {"answer": keys[1], "probabilities": {keys[0]: 0.9, keys[1]: 0.3, "bogus": 1}}
    data["type"]["answer"] = "not-a-key"
    data["priority"]["probabilities"] = {}
    preds, anomalies = gc.parse_response(r, json.dumps(data), system="g", run_id=None, latency_ms=1)
    by_q = {p.question_id: p for p in preds}
    assert sum(by_q["queue"].probabilities.values()) == pytest.approx(1)
    assert {k.split(":")[0] for k in anomalies["queue"]} == {
        "prob_keys_extra", "prob_sum", "answer_not_argmax"
    }  # fmt: skip
    assert by_q["type"].error and anomalies["type"][0].startswith("bad_answer")
    assert anomalies["priority"] == ["no_probabilities"]
    assert by_q["priority"].probabilities["0"] == 1.0  # one-hot on the stated answer
    assert by_q["is_billing"].error is None


def test_falls_back_when_logprobs_unsupported(records):
    err = ApiError(400, "Unable to submit request because Logprobs is not supported for this model")
    fake = FakeGenai(records, script=[err])
    c = client(fake)
    res = c.decide(records[0])
    assert res.error is None
    assert "response_logprobs" not in fake.calls[1]["config"]
    assert not c.logprobs and c.fallbacks and "logprobs disabled" in c.fallbacks[0]
    c.decide(records[1])
    assert "response_logprobs" not in fake.calls[2]["config"]  # stays off


def test_falls_back_when_thinking_cannot_be_disabled(records):
    fake = FakeGenai(records, script=[ApiError(400, "thinking_budget 0 is not supported")])
    c = client(fake, logprobs=False)
    assert c.decide(records[0]).error is None
    assert fake.calls[1]["config"]["thinking_config"] == {"thinking_level": "MINIMAL"}


def test_retries_rate_limits_but_not_bad_requests(records):
    fake = FakeGenai(records, script=[ApiError(429, "quota"), ApiError(503, "busy")])
    res = client(fake).decide(records[0])
    assert res.error is None and res.attempts == 3

    fake = FakeGenai(records, script=[ApiError(400, "invalid schema")])
    with pytest.raises(ApiError):
        client(fake).decide(records[0])


def test_truncated_output_is_an_error(records):
    res = client(FakeGenai(records, finish="MAX_TOKENS")).decide(records[0])
    assert res.error.startswith("finish_reason=MAX_TOKENS")
    assert all(p.error for p in res.predictions)


def test_main_end_to_end(tmp_path, records, monkeypatch):
    fake = FakeGenai(records)
    monkeypatch.setattr(gc, "make_genai_client", lambda *a: fake)
    out = tmp_path / "out"
    code = gc.main(["--project", "p", "--region", "us-central1", "--data", str(DATA),
                    "--run-id", "g-1", "--out", str(out)])  # fmt: skip
    assert code == 0
    preds = list(read_jsonl(out / "predictions.jsonl", Prediction))
    assert len(preds) == sum(len(r.questions) for r in records)
    assert {p.system for p in preds} == {gc.DEFAULT_MODEL}
    schema = json.loads((ROOT / "contract/schemas/prediction.schema.json").read_text())
    for line in (out / "predictions.jsonl").read_text().splitlines():
        jsonschema.validate(json.loads(line), schema)
    cost = Cost.model_validate_json((out / "cost.json").read_text())
    p_in, p_out = gc.PRICES_USD_PER_M[gc.DEFAULT_MODEL]
    n = len(records)
    assert cost.api[0].raw_cost == pytest.approx(n * (500 * p_in + 120 * p_out) / 1e6)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_states_failed"] == 0 and summary["location"] == "us-central1"


def test_unknown_model_price_is_zero_and_says_so(tmp_path, records, monkeypatch):
    monkeypatch.setattr(gc, "make_genai_client", lambda *a: FakeGenai(records))
    out = tmp_path / "out"
    gc.main(["--project", "p", "--region", "r", "--location", "global", "--model", "gemini-9",
             "--data", str(DATA), "--run-id", "g", "--out", str(out)])  # fmt: skip
    cost = Cost.model_validate_json((out / "cost.json").read_text())
    assert cost.total == 0 and "No checked price" in cost.notes
    assert cost.api[0].input_tokens == 500 * len(records)
    assert json.loads((out / "summary.json").read_text())["location"] == "global"


def test_yesno_without_descriptions(records):
    r = records[1]
    assert isinstance(r.questions["is_billing"], YesNoQuestion)
    assert "  - yes\n" in gc.build_prompt(r) + "\n"

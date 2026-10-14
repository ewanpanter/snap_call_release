"""TypeSafe (Jev) baseline client, against a local fake of the System One API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import jsonschema
import pytest

from baselines import typesafe_client as ts
from baselines.common import FatalError, RetryableError, load_records, run_all
from snapcall.schema import (
    Cost,
    DataRecord,
    Prediction,
    ScoreQuestion,
    YesNoQuestion,
    read_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "contract" / "examples" / "data.example.jsonl"
FAKE_KEY = "test-key-not-a-secret-7f3a"


def jev_answer_for(q) -> dict:
    """A plausible, well-formed Jev answer for one question."""
    if isinstance(q, YesNoQuestion):
        return {"type": "noul", "noul": 0.9}
    if isinstance(q, ScoreQuestion):
        n = len(q.levels)
        probs = {str(i): 0.0 for i in range(n)}
        probs[str(n - 1)] = 0.8
        probs[str(n - 2)] = 0.2
        return {
            "type": "score",
            "score": n - 1 - 0.2,
            "confidence": 0.9,
            "probabilities": probs,
            "legend": {str(i): lvl for i, lvl in enumerate(q.levels)},
        }
    keys = list(q.options)
    probs = {k: 0.0 for k in keys}
    probs[keys[0]], probs[keys[1]] = 0.7, 0.3
    return {"type": "choice", "choice": keys[0], "confidence": 0.8, "probabilities": probs}


def jev_response(record: DataRecord) -> dict:
    return {
        "id": f"gen-{record.state_id}",
        "model": "jev-1.13.0-20260917",
        "answers": {qid: jev_answer_for(q) for qid, q in record.questions.items()},
        "usage": {"input_tokens": 400, "output_tokens": 60, "cost": 0.0000168},
    }


@pytest.fixture
def records() -> list[DataRecord]:
    return load_records(DATA)


@pytest.fixture
def fake_server(records):
    """Local System One API. Records what it saw; can be told to fail the first N calls."""
    by_state = {json.dumps(r.state): r for r in records}
    seen: list[dict] = []
    fail_first: dict[str, int] = {"n": 0, "status": 429}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"auth": self.headers.get("Authorization"), "body": body})
            if fail_first["n"] > 0:
                fail_first["n"] -= 1
                self.send_response(fail_first["status"])
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(b'{"error": "slow down"}')
                return
            payload = json.dumps(jev_response(by_state[json.dumps(body["state"])])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1/systemone", seen, fail_first
    server.shutdown()


def run_main(tmp_path, endpoint, monkeypatch, *extra) -> int:
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    return ts.main(
        ["--data", str(DATA), "--run-id", "t-1", "--out", str(tmp_path / "out"),
         "--endpoint", endpoint, "--max-attempts", "3", *extra]
    )  # fmt: skip


def test_end_to_end_writes_contract_outputs(tmp_path, fake_server, monkeypatch, records):
    endpoint, seen, _ = fake_server
    assert run_main(tmp_path, endpoint, monkeypatch) == 0
    out = tmp_path / "out"

    preds = list(read_jsonl(out / "predictions.jsonl", Prediction))
    assert len(preds) == sum(len(r.questions) for r in records)
    assert all(p.error is None and p.system == "typesafe-jev" for p in preds)
    schema = json.loads((ROOT / "contract/schemas/prediction.schema.json").read_text())
    for line in (out / "predictions.jsonl").read_text().splitlines():
        jsonschema.validate(json.loads(line), schema)

    cost = Cost.model_validate_json((out / "cost.json").read_text())
    assert cost.api[0].provider == "typesafe"
    assert cost.api[0].raw_cost == pytest.approx(0.0000168 * len(records))
    assert cost.api[0].requests == len(records)

    # The request is the contract adapter's shape, with the pinned model.
    assert all(s["body"]["model"] == ts.DEFAULT_MODEL for s in seen)
    assert seen[0]["body"]["questions"]["is_billing"]["type"] == "noul"


def test_key_only_in_header_never_in_outputs(tmp_path, fake_server, monkeypatch, capsys):
    endpoint, seen, _ = fake_server
    run_main(tmp_path, endpoint, monkeypatch)
    assert all(s["auth"] == f"Bearer {FAKE_KEY}" for s in seen)
    assert all(FAKE_KEY not in json.dumps(s["body"]) for s in seen)
    for f in (tmp_path / "out").iterdir():
        assert FAKE_KEY not in f.read_text(), f.name
    captured = capsys.readouterr()
    assert FAKE_KEY not in captured.out + captured.err


def test_retries_429_then_succeeds(tmp_path, fake_server, monkeypatch):
    endpoint, _, fail_first = fake_server
    fail_first["n"] = 2
    assert run_main(tmp_path, endpoint, monkeypatch, "--concurrency", "1") == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["n_states_failed"] == 0
    assert summary["retried_states"] == 1


def test_401_stops_the_run(tmp_path, fake_server, monkeypatch, records):
    endpoint, seen, fail_first = fake_server
    fail_first.update(n=99, status=401)
    assert run_main(tmp_path, endpoint, monkeypatch, "--concurrency", "1") == 1
    assert len(seen) == 1  # no pointless requests after a bad key
    preds = list(read_jsonl(tmp_path / "out" / "predictions.jsonl", Prediction))
    assert all(p.error for p in preds)
    assert any("not attempted" in p.error for p in preds)


def test_missing_key_exits_without_calling(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(ts, "post_json", lambda *a, **k: called.append(1))
    code = ts.main(["--data", str(DATA), "--run-id", "x", "--out", str(tmp_path)])
    assert code == 2 and not called


def test_repr_hides_key():
    assert FAKE_KEY not in repr(ts.TypeSafeClient(FAKE_KEY))


# --------------------------------------------------------------------------------------
# Odd responses
# --------------------------------------------------------------------------------------


def _q(records, qid):
    return records[0].questions[qid]


def test_check_answer_clean(records):
    for q in records[0].questions.values():
        assert ts.check_answer(q, jev_answer_for(q)) == []


def test_check_answer_flags_oddities(records):
    queue = _q(records, "queue")
    a = jev_answer_for(queue)
    a["choice"] = list(queue.options)[1]  # the 0.3 option
    a["probabilities"][list(queue.options)[2]] = 0.1  # sum 1.1
    del a["confidence"]
    kinds = ts.check_answer(queue, a)
    assert {k.split(":")[0] for k in kinds} == {"choice_not_argmax", "prob_sum", "no_confidence"}

    prio = _q(records, "priority")
    a = jev_answer_for(prio)
    a["score"] = 0.5
    a["legend"] = {"0": "x"}
    kinds = {k.split(":")[0] for k in ts.check_answer(prio, a)}
    assert kinds == {"score_not_expectation", "legend_mismatch"}

    yn = _q(records, "is_billing")
    assert ts.check_answer(yn, {"type": "noul", "noul": 1.4})[0].startswith("noul_out_of_range")
    assert ts.check_answer(yn, None) == ["missing_answer"]
    assert ts.check_answer(yn, {"type": "choice", "noul": 0.2}) == ["type_mismatch:choice"]


def test_bad_answer_fails_only_its_question(records):
    r = records[0]
    body = jev_response(r)
    body["answers"]["queue"]["choice"] = "not-an-option"
    del body["answers"]["is_billing"]
    preds, anomalies = ts.convert(r, body, system="s", run_id="r", latency_ms=12.0)
    by_q = {p.question_id: p for p in preds}
    assert by_q["queue"].error and by_q["is_billing"].error
    assert by_q["type"].error is None and by_q["priority"].error is None
    assert anomalies["queue"] == ["choice_not_an_option:'not-an-option'"]
    assert anomalies["is_billing"] == ["missing_answer"]
    # Raw (pre-renormalisation) numbers are kept for the report.
    assert by_q["priority"].meta["raw_score"] == pytest.approx(
        len(r.questions["priority"].levels) - 1.2
    )


def test_rounded_probabilities_are_renormalised(records):
    r = records[0]
    body = jev_response(r)
    body["answers"]["queue"]["probabilities"][list(r.questions["queue"].options)[0]] = 0.71
    preds, anomalies = ts.convert(r, body, system="s", run_id=None, latency_ms=None)
    queue = next(p for p in preds if p.question_id == "queue")
    assert sum(queue.probabilities.values()) == pytest.approx(1.0)
    assert queue.meta["raw_prob_sum"] == pytest.approx(1.01)
    assert "queue" not in anomalies  # within Jev's two-decimal rounding


def test_run_all_stops_after_fatal(records):
    client = ts.TypeSafeClient(FAKE_KEY)
    calls = []

    def decide(r):
        calls.append(r.state_id)
        raise FatalError("HTTP 402: no credits")

    results = run_all(records, decide, client.failed, concurrency=1)
    assert len(calls) == 1
    assert "fatal" in results[0].error and "not attempted" in results[1].error


def test_post_json_maps_status(monkeypatch):
    import io
    import urllib.error

    def raise_http(code):
        def urlopen(req, timeout):
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b"nope"))

        return urlopen

    for code, exc in [(429, RetryableError), (503, RetryableError), (401, FatalError),
                      (402, FatalError), (400, ValueError)]:  # fmt: skip
        monkeypatch.setattr("urllib.request.urlopen", raise_http(code))
        with pytest.raises(exc):
            ts.post_json("https://example.invalid", {}, FAKE_KEY)


def test_repeat_same_order_sends_identical_requests(tmp_path, fake_server, monkeypatch, records):
    endpoint, seen, _ = fake_server
    assert run_main(tmp_path, endpoint, monkeypatch, "--repeat-same-order", "2") == 0
    out = tmp_path / "out"
    n = len(records)
    assert len(seen) == 2 * n
    # Pass 2 starts after pass 1 ends, and asks exactly the same requests.
    key = lambda s: json.dumps(s["body"], sort_keys=True)  # noqa: E731
    assert sorted(map(key, seen[:n])) == sorted(map(key, seen[n:]))

    preds = list(read_jsonl(out / "predictions.jsonl", Prediction))
    assert len(preds) == sum(len(r.questions) for r in records)  # pass 1 only is scored
    assert all(p.system == "typesafe-jev-rep2" for p in preds)
    assert all(len(p.meta["repeats"]) == 1 for p in preds)
    summary = json.loads((out / "summary.json").read_text())
    rep = summary["repeat_same_order"]
    assert rep["passes"] == 2 and len(rep["latency_per_pass"]) == 2
    assert all(q["answer_changed"] == 0 for q in rep["per_question"].values())
    cost = Cost.model_validate_json((out / "cost.json").read_text())
    assert cost.api[0].raw_cost == pytest.approx(0.0000168 * 2 * n)


def test_repeat_and_permutations_dont_mix(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    args = ["--data", str(DATA), "--run-id", "x", "--out", str(tmp_path)]
    assert ts.main([*args, "--permutations", "4", "--repeat-same-order", "2"]) == 2

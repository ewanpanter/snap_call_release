"""The GPU job, end to end on the tiny random-weights models (CPU, seconds)."""

import argparse
import json

import pytest
import torch

from eval import equivalence, run_eval
from snapcall import masking
from snapcall.schema import DataRecord, Metrics, Prediction, read_jsonl, write_jsonl

EXAMPLES = "contract/examples"


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from tests.fixtures.tiny_qwen3 import build_tiny_qwen3

    model, tok = build_tiny_qwen3("sdpa")
    d = tmp_path_factory.mktemp("tiny-qwen3")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def data_file(tmp_path_factory):
    records = list(read_jsonl(f"{EXAMPLES}/data.example.jsonl", DataRecord))
    p = tmp_path_factory.mktemp("data") / "test.jsonl"
    write_jsonl(p, records)
    return p


def test_run_eval_all_stages(tiny_dir, data_file, tmp_path):
    out = tmp_path / "out"
    rc = run_eval.main(
        [
            "--model",
            str(tiny_dir),
            "--data",
            str(data_file),
            "--out",
            str(out),
            "--equivalence-dtypes",
            "float32",
            "bfloat16",
            "--equivalence-states",
            "1",
            "--latency-states",
            "1",
            "--latency-forwards",
            "packed",
            "cached",
            "--latency-n-questions",
            "1",
            "8",
            "--run-id",
            "t-1-1",
        ]
    )
    stages = json.loads((out / "stages.json").read_text())
    assert rc == 0, stages
    (eq,) = json.loads((out / "equivalence.json").read_text())["models"].values()
    assert eq["float32"]["packed_vs_separate"]["prob_diff_max"] < 1e-5
    assert eq["float32"]["cached_vs_separate"]["prob_diff_max"] < 1e-5
    assert "separate_vs_float32_separate" in eq["bfloat16"]
    assert eq["bfloat16"]["suggested_tolerance"] is not None

    preds = list(read_jsonl(out / "predictions.jsonl", Prediction))
    assert preds and all(p.system == f"snapcall-{tiny_dir.name}-zeroshot" for p in preds)
    assert all(p.run_id == "t-1-1" for p in preds)
    choice = [p for p in preds if p.type == "choice"]
    assert all(len(p.meta["per_permutation"]) > 1 for p in choice)
    m = Metrics.model_validate_json((out / "metrics.json").read_text())
    assert m.n_questions == len(preds) and m.order_sensitivity is not None
    timings = json.loads((out / "eval_timings.json").read_text())
    assert timings["n_decisions"] == len(preds) and timings["eval_seconds"] > 0

    rows = [json.loads(line) for line in (out / "latency.jsonl").read_text().splitlines()]
    assert {r["n_questions"] for r in rows} == {1, 8}
    assert {(r["debias"], r["forward"]) for r in rows} == {
        (d, f) for d in ("off", "auto") for f in ("packed", "cached")
    }
    assert all(r["forward_used"] == r["forward"] for r in rows)
    prior = json.loads((out / "label_prior.json").read_text())
    assert set(prior["flatness"]) == {"A-D", "1-4", "random letters"}


def test_gate_stops_the_job_when_cached_is_wrong(tiny_dir, data_file, tmp_path, monkeypatch):
    def broken_cached(model, prefix_ids, branch_ids):
        ref = masking.separate_forward(model, prefix_ids, branch_ids)
        return ref + torch.randn(ref.shape, generator=torch.Generator().manual_seed(0))

    monkeypatch.setitem(masking.FORWARDS, "cached", broken_cached)
    out = tmp_path / "out"
    rc = run_eval.main(
        [
            "--model",
            str(tiny_dir),
            "--data",
            str(data_file),
            "--out",
            str(out),
            "--equivalence-dtypes",
            "float32",
            "--equivalence-states",
            "1",
            "--gate-cached",
        ]
    )
    assert rc == 3
    assert not (out / "predictions.jsonl").exists()
    assert json.loads((out / "stages.json").read_text())["cache_gate"]["ok"] is False


def test_with_n_questions_cycles_ids():
    (record,) = [r for r in read_jsonl(f"{EXAMPLES}/data.example.jsonl", DataRecord)][:1]
    n = len(record.questions)
    qs = run_eval.with_n_questions(record, 2 * n + 1)
    assert len(qs) == 2 * n + 1 and len(set(qs)) == 2 * n + 1
    assert list(run_eval.with_n_questions(record, 1)) == list(record.questions)[:1]


def test_suggested_tolerance_rounds_up():
    sec = {"separate_vs_float32_separate": {"prob_diff_max": 0.031}}
    assert equivalence.suggested_tolerance(sec) == pytest.approx(0.07)
    sec = {"separate_vs_float32_separate": {"prob_diff_max": 0.0004}}
    assert equivalence.suggested_tolerance(sec) == pytest.approx(0.0008)
    assert equivalence.suggested_tolerance({}) is None


def test_cached_gate_rules():
    def body(dt, **sec):
        return {"models": {"m": {dt: {"cached_vs_separate": sec}}}}

    exact = {"prob_diff_max": 5e-5, "prob_diff_p95": 3e-5, "argmax_agree_rate": 1.0}
    assert run_eval.cached_gate(body("float32", **exact), "/x/m")[0]
    noisy = {"prob_diff_max": 0.4, "prob_diff_p95": 0.15, "argmax_agree_rate": 0.99}
    assert not run_eval.cached_gate(body("float32", **noisy), "/x/m")[0]
    assert run_eval.cached_gate(body("bfloat16", **noisy), "/x/m", dtype="bfloat16")[0]
    wrong = {"prob_diff_max": 0.98, "prob_diff_p95": 0.96, "argmax_agree_rate": 0.78}
    assert not run_eval.cached_gate(body("bfloat16", **wrong), "/x/m", dtype="bfloat16")[0]
    assert not run_eval.cached_gate(body("bfloat16", n_failed=3), "/x/m", dtype="bfloat16")[0]


def test_gate_dtype_reaches_the_gate(tiny_dir, data_file, tmp_path):
    """--gate-dtype bfloat16 through the CLI: the gate judges the bf16 section, and the exit
    code follows its verdict (3 = gate failed). A plumbing test: whether the tiny random model
    passes in bf16 can go either way (near-uniform answers flip under rounding); the rule
    itself is tested strictly in test_decide.py."""
    out = tmp_path / "out"
    rc = run_eval.main(
        [
            "--model",
            str(tiny_dir),
            "--data",
            str(data_file),
            "--out",
            str(out),
            "--equivalence-dtypes",
            "bfloat16",
            "--equivalence-states",
            "1",
            "--gate-cached",
            "--gate-dtype",
            "bfloat16",
            "--stages",
            "equivalence",
        ]
    )
    gate = json.loads((out / "stages.json").read_text())["cache_gate"]
    assert "in bfloat16" in gate["why"], gate
    assert rc == (0 if gate["ok"] else 3)


def test_extra_data_is_scored_into_its_own_folder(tiny_dir, data_file, tmp_path):
    """--extra-data ood=...: the same model scores a second gold file in the same job, laid
    out like the LoRA runs (OUT/ood/predictions.jsonl, metrics, eval_timings)."""
    out = tmp_path / "out"
    rc = run_eval.main(
        [
            "--model",
            str(tiny_dir),
            "--data",
            str(data_file),
            "--out",
            str(out),
            "--stages",
            "eval",
            "--extra-data",
            f"ood={data_file}",
        ]
    )
    assert rc == 0, json.loads((out / "stages.json").read_text())
    stages = json.loads((out / "stages.json").read_text())["stages"]
    assert stages["eval"]["ok"] and stages["eval:ood"]["ok"]
    for d in (out, out / "ood"):
        assert (d / "predictions.jsonl").exists() and (d / "eval_timings.json").exists()
        assert (d / "metrics.json").exists()


def test_quantised_systems_say_so_in_their_name():
    assert run_eval.default_system("/opt/models/microsoft/phi-4", "8bit") == (
        "snapcall-phi-4-8bit-zeroshot"
    )
    assert run_eval.default_system("/opt/models/microsoft/Phi-4-mini-instruct") == (
        "snapcall-phi-4-mini-instruct-zeroshot"
    )


def test_extra_data_argument():
    assert run_eval.extra_data("ood=gs://b/ood_test.jsonl") == ("ood", "gs://b/ood_test.jsonl")
    for bad in ("ood", "=x", "a b=x", "ood="):
        with pytest.raises(argparse.ArgumentTypeError):
            run_eval.extra_data(bad)

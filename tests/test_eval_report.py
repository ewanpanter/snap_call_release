import json
from pathlib import Path

import pytest

from eval import report
from snapcall.schema import DataRecord, Prediction, read_jsonl, write_jsonl

EXAMPLES = "contract/examples"


@pytest.fixture
def runs(tmp_path):
    """Two fake systems' run dirs, split from the contract's example predictions."""
    preds = list(read_jsonl(f"{EXAMPLES}/predictions.example.jsonl", Prediction))
    dirs = []
    for system in sorted({p.system for p in preds}):
        d = tmp_path / system
        d.mkdir()
        write_jsonl(d / "predictions.jsonl", [p for p in preds if p.system == system])
        dirs.append(d)
    cost = json.loads(Path(f"{EXAMPLES}/cost.example.json").read_text())
    (dirs[0] / "cost.json").write_text(json.dumps(cost))
    (dirs[0] / "eval_timings.json").write_text(
        json.dumps({"eval_seconds": 36.0, "n_decisions": 100, "input_tokens": 5000})
    )
    (dirs[0] / "latency.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "state_id": "s",
                    "n_questions": n,
                    "debias": db,
                    "forward": f,
                    "forward_used": f,
                    "latency_ms": 10.0 * n,
                    "input_tokens": 100 * n,
                    "compute_tokens": 100 * n,
                    "n_branches": n,
                }
            )
            + "\n"
            for n in (1, 2, 4, 8)
            for db in ("off", "auto")
            for f in ("packed", "cached")
        )
    )
    (dirs[0] / "_runner_launch.json").write_text(
        json.dumps(
            {
                "zone": "asia-northeast1-a",
                "search_seconds": 20.5,
                "zones_tried": 2,
                "attempts": [
                    {"zone": "x-a", "outcome": "capacity", "seconds": 8.0},
                    {"zone": "asia-northeast1-a", "outcome": "created", "seconds": 12.5},
                ],
            }
        )
    )
    return dirs


def test_report_end_to_end(runs, tmp_path):
    gold = tmp_path / "gold.jsonl"
    write_jsonl(gold, list(read_jsonl(f"{EXAMPLES}/data.example.jsonl", DataRecord)))
    out = tmp_path / "report"
    args = ["--gold", str(gold), "--out", str(out), "--no-charts"]
    for d in runs:
        args += ["--run", str(d)]
    assert report.main(args) == 0
    body = json.loads((out / "report.json").read_text())
    first = body["systems"][runs[0].name]
    # rate 0.4 GBP/h x 36 s / 100 decisions x 1000
    assert first["cost"]["steady_per_1k"] == pytest.approx(0.4 * 36 / 3600 / 100 * 1000)
    assert {c["n_questions"] for c in first["latency_curve"]} == {1, 2, 4, 8}
    assert (out / "summary.md").read_text().startswith("# SnapCall eval summary")
    assert list((out / "metrics").glob("*.json"))


def test_charts_render(runs, tmp_path):
    pytest.importorskip("matplotlib")
    gold = tmp_path / "gold.jsonl"
    records = list(read_jsonl(f"{EXAMPLES}/data.example.jsonl", DataRecord))
    write_jsonl(gold, records)
    loaded = [report.load_run(d) for d in runs]
    body = report.build(loaded, records)
    made = report.charts(body, tmp_path / "charts", [r["system"] for r in loaded])
    assert "accuracy_f1.png" in made and "capacity_search.png" in made
    assert "latency_vs_questions.png" in made


def test_question_order_comes_from_the_gold_file():
    records = list(read_jsonl(f"{EXAMPLES}/data.example.jsonl", DataRecord))
    qs, yesno = report.question_order(records)
    assert qs == list(dict.fromkeys(q for r in records for q in r.questions))
    assert yesno and all(q in qs for q in yesno)


def test_lora_run_layout_costs_the_eval_not_the_training(tmp_path):
    """The LoRA run layout: calibrated predictions at the root and in ood/, the T = 1 pass
    (with its eval timing) in uncalibrated/, the job's cost.json (training included) at the
    root."""
    preds = list(read_jsonl(f"{EXAMPLES}/predictions.example.jsonl", Prediction))
    system = preds[0].system
    run = tmp_path / "outputs-gce-1-1"
    for sub in ("", "uncalibrated", "ood"):
        (run / sub).mkdir(parents=True, exist_ok=True)
        write_jsonl(run / sub / "predictions.jsonl", [p for p in preds if p.system == system])
    cost = json.loads(Path(f"{EXAMPLES}/cost.example.json").read_text())  # 0.4 GBP/h
    (run / "cost.json").write_text(json.dumps(cost))
    (run / "train_summary.json").write_text("{}")
    timings = {"eval_seconds": 36.0, "n_decisions": 100, "input_tokens": 5000}
    (run / "uncalibrated" / "eval_timings.json").write_text(json.dumps(timings))

    for d in (run, run / "ood"):
        c = report.cost_per_1k(report.load_run(d))
        assert c["job_includes_training"] and "job_per_1k" not in c
        assert c["job_gbp"] == cost["total"]
    c = report.cost_per_1k(report.load_run(run))
    assert c["steady_per_1k"] == pytest.approx(0.4 * 36 / 3600 / 100 * 1000)

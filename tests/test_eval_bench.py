"""Throughput-mode batching and FP8 layers (eval/bench.py), on the tiny random-weights Qwen3."""

import json

import pytest
import torch

from eval import bench
from snapcall.masking import packed_forward
from snapcall.schema import DataRecord, read_jsonl, write_jsonl

EXAMPLES = "contract/examples"


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from tests.fixtures.tiny_qwen3 import build_tiny_qwen3

    model, tok = build_tiny_qwen3("sdpa")
    d = tmp_path_factory.mktemp("tiny-qwen3")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d


def records():
    return list(read_jsonl(f"{EXAMPLES}/data.example.jsonl", DataRecord))


def test_batched_tickets_match_one_at_a_time(tiny_qwen3):
    """Several tickets in one padded batch give each ticket the same logits as packing it
    alone (fp32, so any masking or padding mistake would show)."""
    model, tok = tiny_qwen3
    items = bench.prepare(tok, records())
    encoded = [e for _, _, e in items]
    with torch.inference_mode():
        batched = bench.batched_packed_logits(model, encoded)
        for e, got in zip(encoded, batched, strict=True):
            ref = packed_forward(model, e.prefix_ids, e.branch_ids)
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_throughput_sweep_reports_each_batch_size(tiny_qwen3):
    model, tok = tiny_qwen3
    tp = bench.throughput(model, tok, records(), [1, 2], check_states=4, n_tickets=2)
    assert [r["batch_size"] for r in tp["rows"]] == [1, 2]
    assert all(r["decisions_per_s"] > 0 for r in tp["rows"])
    assert tp["batched_vs_single"]["same_answer_rate"] == 1.0
    assert tp["n_tickets"] == 2  # the length-stratified sample


def test_bench_job_end_to_end_bf16(tiny_dir, tmp_path):
    data = tmp_path / "test.jsonl"
    write_jsonl(data, records())
    out = tmp_path / "out"
    rc = bench.main(
        ["--model", str(tiny_dir), "--data", str(data), "--out", str(out),
         "--precisions", "bf16", "--batch-sizes", "1", "2"]
    )  # fmt: skip
    summary = json.loads((out / "bench.json").read_text())
    assert rc == 0, summary
    v = summary["variants"]["base-bf16"]
    assert v["ok"] and v["eval"]["n_decisions"] > 0
    assert (out / "base-bf16" / "predictions.jsonl").exists()
    assert json.loads((out / "base-bf16" / "throughput.json").read_text())["rows"]


def test_one_order_packs_fewer_tokens(tiny_qwen3):
    """debias="off" packs one option order: fewer tokens per ticket than the sweep's "auto"."""
    model, tok = tiny_qwen3
    auto = bench.throughput(model, tok, records(), [1], check_states=2, debias="auto")
    off = bench.throughput(model, tok, records(), [1], check_states=2, debias="off")
    assert (auto["debias"], off["debias"]) == ("auto", "off")
    assert off["n_tokens"] < auto["n_tokens"]
    assert off["n_decisions"] == auto["n_decisions"]


def test_bench_job_one_order_writes_both_files(tiny_dir, tmp_path):
    data = tmp_path / "test.jsonl"
    write_jsonl(data, records())
    out = tmp_path / "out"
    rc = bench.main(
        ["--model", str(tiny_dir), "--data", str(data), "--out", str(out), "--skip-eval",
         "--precisions", "bf16", "--batch-sizes", "1", "--debias", "auto", "off"]
    )  # fmt: skip
    assert rc == 0
    v = json.loads((out / "bench.json").read_text())["variants"]["base-bf16"]
    assert v["throughput"]["debias"] == "auto" and v["throughput-off"]["debias"] == "off"
    assert (out / "base-bf16" / "throughput-off.json").exists()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 matmuls need an Ada/Hopper GPU",
)
def test_fp8_linear_close_to_bf16():
    lin = torch.nn.Linear(256, 512, bias=True).cuda().to(torch.bfloat16)
    x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
    ref = lin(x).float()
    got = bench.FP8Linear(lin)(x).float()
    assert ((got - ref).norm() / ref.norm()).item() < 0.05

"""Several variants in one job (eval/multi.py), on the tiny random-weights Qwen3."""

import argparse
import json

import pytest

from eval import multi
from snapcall.schema import DataRecord, Prediction, read_jsonl, write_jsonl

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
def tiny_adapter(tiny_dir, tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_dir)
    cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    d = tmp_path_factory.mktemp("tiny-adapter")
    get_peft_model(base, cfg).save_pretrained(d)
    return d


def test_parse_variant_defaults_and_system_names():
    v = multi.parse_variant("name=qwen4b,model=/m/Qwen/Qwen3-4B-Instruct-2507")
    assert v["forward"] == "packed"
    assert v["system"] == "snapcall-qwen3-4b-instruct-2507-zeroshot"
    v = multi.parse_variant("name=a,model=/m/Qwen/Qwen3-4B-Instruct-2507,adapter=gs://b/r/x=1")
    assert v["adapter"] == "gs://b/r/x=1"
    assert v["system"] == "snapcall-qwen3-4b-instruct-2507-lora-a-uncal"
    with pytest.raises(argparse.ArgumentTypeError):
        multi.parse_variant("name=x")  # no model
    with pytest.raises(argparse.ArgumentTypeError):
        multi.parse_variant("name=x,model=y,forward=sideways")
    with pytest.raises(argparse.ArgumentTypeError):
        multi.parse_variant("name=x,model=y,colour=red")


def test_multi_job_scores_each_variant(tiny_dir, tiny_adapter, tmp_path):
    data = tmp_path / "cfpb_test.jsonl"
    records = list(read_jsonl(f"{EXAMPLES}/data.example.jsonl", DataRecord))
    write_jsonl(data, records)
    out = tmp_path / "out"
    rc = multi.main(
        ["--data", str(data), "--out", str(out),
         "--variant", f"name=zs,model={tiny_dir}",
         "--variant", f"name=zs-cached,model={tiny_dir},forward=cached",
         "--variant", f"name=a,model={tiny_dir},adapter={tiny_adapter}"]
    )  # fmt: skip
    summary = json.loads((out / "multi.json").read_text())
    assert rc == 0, summary
    assert summary["n_states"] == len(records)
    for name in ("zs", "zs-cached", "a"):
        assert summary["variants"][name]["ok"], summary["variants"][name]
        preds = list(read_jsonl(out / name / "predictions.jsonl", Prediction))
        assert preds and not any(p.error for p in preds)
    systems = {
        n: list(read_jsonl(out / n / "predictions.jsonl", Prediction))[0].system
        for n in ("zs", "a")
    }
    assert systems["a"].endswith("-lora-a-uncal") and systems["zs"].endswith("-zeroshot")


def test_one_failing_variant_keeps_the_others(tiny_dir, tmp_path):
    data = tmp_path / "test.jsonl"
    write_jsonl(data, list(read_jsonl(f"{EXAMPLES}/data.example.jsonl", DataRecord)))
    out = tmp_path / "out"
    rc = multi.main(
        ["--data", str(data), "--out", str(out),
         "--variant", f"name=missing,model={tmp_path}/no-such-model",
         "--variant", f"name=zs,model={tiny_dir}"]
    )  # fmt: skip
    summary = json.loads((out / "multi.json").read_text())
    assert rc == 1
    assert summary["variants"]["missing"]["ok"] is False
    assert summary["variants"]["zs"]["ok"] is True


def test_example_args_pass_the_runners_weight_prefetch():
    """The runner pre-fetches every ``{models}/<org>/<name>``; it must be a token's tail."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "ops" / "runner"))
    import vm

    args = (
        "--data gs://b/data/cfpb_test.jsonl "
        "--variant name=gemma,forward=cached,model={models}/google/gemma-4-E4B-it "
        "--variant name=a,adapter=gs://b/runs/x/outputs/adapter,model={models}/Qwen/Qwen3-4B"
    )
    assert vm.model_repos(args) == ["google/gemma-4-E4B-it", "Qwen/Qwen3-4B"]
    for spec in args.split("--variant ")[1:]:
        assert multi.parse_variant(spec.strip())["model"].startswith("{models}/")
    with pytest.raises(ValueError):  # model in the middle: what failed on 2 Oct
        vm.model_repos("--variant name=g,model={models}/google/gemma-4-E4B-it,forward=cached")

import json
import warnings
from pathlib import Path

import pytest
import torch

from snapcall.decide import (
    CANDIDATE_LABELS,
    SMOKE_REQUEST,
    StateTooLong,
    choose_forward,
    compare_branches,
    decide,
    decide_detailed,
    equivalence_report,
    flatness,
    label_prior,
    main,
)
from snapcall.prompt import PromptProfile
from snapcall.schema import DecideRequest, DecideResponse, YesNoQuestion, predictions_from_response

EXAMPLE = Path(__file__).parent.parent / "contract" / "examples" / "decide_request.example.json"


@pytest.fixture
def request_example() -> DecideRequest:
    return DecideRequest.model_validate_json(EXAMPLE.read_text())


def probabilities(answer) -> dict[str, float]:
    if hasattr(answer, "probabilities"):
        return answer.probabilities
    return {"yes": answer.p_yes}


def assert_same_answers(a: DecideResponse, b: DecideResponse, abs_tol: float) -> None:
    for qid, answer in a.answers.items():
        expected = probabilities(b.answers[qid])
        for k, p in probabilities(answer).items():
            assert p == pytest.approx(expected[k], abs=abs_tol), (qid, k)


# --- decide ---------------------------------------------------------------------------


@pytest.mark.parametrize("forward", ["packed", "cached", "separate"])
def test_decide_end_to_end_tiny(tiny_qwen3, request_example, forward):
    model, tok = tiny_qwen3
    response = decide(model, tok, request_example, forward=forward)
    DecideResponse.model_validate(response.model_dump())
    response.check_against(request_example)
    assert response.usage.output_tokens == 0
    assert response.usage.input_tokens > 0
    assert all(a.confidence is not None for a in response.answers.values())
    # flattening into predictions.jsonl lines must validate too
    preds = predictions_from_response(response, system="snapcall-tiny", state_id="s1")
    assert [p.question_id for p in preds] == ["queue", "priority", "is_billing"]


def test_decide_forwards_agree(tiny_qwen3, request_example):
    model, tok = tiny_qwen3
    reference = decide(model, tok, request_example, forward="separate")
    for forward in ("packed", "cached"):
        assert_same_answers(decide(model, tok, request_example, forward=forward), reference, 1e-5)


def test_decide_debias_off_has_no_confidence(tiny_qwen3, request_example):
    model, tok = tiny_qwen3
    request = request_example.model_copy(update={"debias": "off"})
    response = decide(model, tok, request)
    assert all(a.confidence is None for a in response.answers.values())
    # fewer branches, fewer tokens
    assert response.usage.input_tokens < decide(model, tok, request_example).usage.input_tokens


def test_compute_tokens_vs_input_tokens(tiny_qwen3, request_example):
    """input_tokens is the billing-style count (same for every forward); compute_tokens is
    what each forward actually ran."""
    model, tok = tiny_qwen3
    runs = {
        f: decide_detailed(model, tok, request_example, forward=f)
        for f in ("packed", "cached", "separate")
    }
    billed = {r.response.usage.input_tokens for r in runs.values()}
    assert len(billed) == 1
    assert runs["packed"].compute_tokens == billed.pop()
    assert runs["packed"].compute_tokens <= runs["cached"].compute_tokens
    assert runs["cached"].compute_tokens < runs["separate"].compute_tokens


def test_decide_switches_to_cached_for_long_prompts(tiny_qwen3, request_example):
    model, tok = tiny_qwen3
    assert decide_detailed(model, tok, request_example).forward == "packed"
    short_limit = decide_detailed(model, tok, request_example, packed_max_tokens=10)
    assert short_limit.forward == "cached"


def test_decide_rejects_huge_state(tiny_qwen3, monkeypatch):
    model, tok = tiny_qwen3
    monkeypatch.setattr("snapcall.decide.MAX_STATE_TOKENS", 100)
    request = DecideRequest(state="x" * 200, questions={"q": YesNoQuestion(question="?")})
    with pytest.raises(StateTooLong, match="limit is 100"):
        decide(model, tok, request)


def test_choose_forward():
    assert choose_forward("packed", "sdpa", 100, 2048) == "packed"
    assert choose_forward("packed", "eager", 5000, 2048) == "cached"
    assert choose_forward("separate", "flash_attention_2", 5000, 2048) == "separate"
    with pytest.warns(UserWarning, match="flash_attention_2"):
        assert choose_forward("packed", "flash_attention_2", 100, 2048) == "cached"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        choose_forward("cached", "flash_attention_2", 100, 2048)  # no warning


def test_empty_think_is_switchable(tiny_qwen3, request_example):
    model, tok = tiny_qwen3
    plain = decide(model, tok, request_example, profile=PromptProfile(empty_think=False))
    think = decide(model, tok, request_example, profile=PromptProfile(empty_think=True))
    assert think.usage.input_tokens > plain.usage.input_tokens


# --- label-prior probe ----------------------------------------------------------------


def test_flatness():
    assert flatness({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) == pytest.approx(1.0)
    assert flatness({"A": 1.0, "B": 0.0}) == pytest.approx(0.0)
    assert 0 < flatness({"A": 0.7, "B": 0.3}) < 1


@pytest.mark.parametrize("name", list(CANDIDATE_LABELS))
def test_label_prior_tiny(tiny_qwen3, name):
    model, tok = tiny_qwen3
    labels = CANDIDATE_LABELS[name]
    prior = label_prior(model, tok, labels)
    assert list(prior) == list(labels)
    assert sum(prior.values()) == pytest.approx(1.0)


def test_label_prior_for_a_real_question(tiny_qwen3):
    model, tok = tiny_qwen3
    prior = label_prior(model, tok, ("A", "B"), question=YesNoQuestion(question="Refund?"))
    assert list(prior) == ["A", "B"]


# --- equivalence check / smoke test ---------------------------------------------------


@pytest.mark.parametrize("empty_think", [False, True])
def test_equivalence_report_tiny(tiny_qwen3, empty_think):
    model, tok = tiny_qwen3
    profile = PromptProfile(empty_think=empty_think)
    report = equivalence_report(model, tok, SMOKE_REQUEST, profile=profile)
    assert report["ok"], report
    assert report["packed"]["max_abs_prob_diff"] < 1e-5
    assert report["empty_think"] is empty_think


def test_smoke_cli_on_saved_tiny_model(tiny_qwen3, tmp_path):
    """The smoke-test entry point, end to end, with the tiny model saved to disk."""
    model, tok = tiny_qwen3
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    out = tmp_path / "outputs"
    assert main(["smoke", "--model", str(tmp_path / "model"), "--out", str(out)]) == 0
    report = json.loads((out / "equivalence.json").read_text())
    assert report["ok"] and report["dtype"] == "torch.float32"
    assert report["packed"]["check"].startswith("float32")

    assert main(["probe", "--model", str(tmp_path / "model"), "--out", str(out)]) == 0
    priors = json.loads((out / "label_prior.json").read_text())
    assert set(priors["flatness"]) == set(CANDIDATE_LABELS)


def test_compare_branches_float32_is_strict():
    ref = [torch.tensor([0.6, 0.4]), torch.tensor([0.5, 0.5])]
    near = [torch.tensor([0.6005, 0.3995]), torch.tensor([0.5, 0.5])]
    far = [torch.tensor([0.61, 0.39]), torch.tensor([0.5, 0.5])]
    assert compare_branches(near, ref, torch.float32)["ok"]
    assert not compare_branches(far, ref, torch.float32)["ok"]


def test_compare_branches_bf16_tolerates_rounding_but_not_wrong_answers():
    ref = [torch.tensor([0.6, 0.4])] * 100
    # One near-tie swung by 0.3 (bf16 noise): still fine at the 95th percentile.
    noisy = [torch.tensor([0.6, 0.4])] * 99 + [torch.tensor([0.45, 0.55])]
    r = compare_branches(noisy, ref, torch.bfloat16)
    assert r["ok"] and r["same_argmax_rate"] == 0.99 and not r["same_argmax"]
    assert not compare_branches(noisy, ref, torch.float32)["ok"]
    # Many changed answers: a real bug, even in bf16.
    broken = [torch.tensor([0.4, 0.6])] * 5 + [torch.tensor([0.6, 0.4])] * 95
    assert not compare_branches(broken, ref, torch.bfloat16)["ok"]


# --- real weights (CI only) -----------------------------------------------------------


@pytest.mark.real_weights
def test_decide_real(real_qwen3, request_example):
    model, tok = real_qwen3
    packed = decide_detailed(model, tok, request_example, forward="packed")
    cached = decide(model, tok, request_example, forward="cached")
    assert packed.forward == "packed"
    packed.response.check_against(request_example)
    assert_same_answers(packed.response, cached, 1e-3)


@pytest.mark.real_weights
@pytest.mark.parametrize("empty_think", [False, True])
def test_equivalence_report_real(real_qwen3, empty_think):
    """The GPU smoke-test check, float32 (max gap < 1e-3, same argmax), here on CPU."""
    model, tok = real_qwen3
    report = equivalence_report(model, tok, SMOKE_REQUEST, profile=PromptProfile(empty_think))
    assert report["ok"], report


@pytest.mark.real_weights
def test_label_prior_real(real_qwen3):
    model, tok = real_qwen3
    for labels in CANDIDATE_LABELS.values():
        prior = label_prior(model, tok, labels)
        assert sum(prior.values()) == pytest.approx(1.0)
        assert 0 < flatness(prior) <= 1


def test_choose_forward_falls_back_for_non_full_attention_models():
    with pytest.warns(UserWarning, match="linear-attention"):
        assert choose_forward("packed", "sdpa", 10, 2048, full_attention=False) == "cached"
    assert choose_forward("separate", "sdpa", 10, 2048, full_attention=False) == "separate"

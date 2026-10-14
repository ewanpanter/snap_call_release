import pytest
import torch

from snapcall.prompt import Branch, build_prompt
from snapcall.readout import (
    QuestionResult,
    agreement,
    read_branches,
    restricted_softmax,
    to_answer,
)
from snapcall.schema import (
    ChoiceQuestion,
    DecideRequest,
    ScoreQuestion,
    YesNoQuestion,
)

LABEL_IDS = [100, 101, 102, 103]  # pretend token ids of A, B, C, D
VOCAB = 200


def logits_favouring(label_index: int, strength: float = 5.0) -> torch.Tensor:
    row = torch.zeros(VOCAB)
    row[LABEL_IDS[label_index]] = strength
    return row


def test_restricted_softmax_ignores_other_tokens():
    logits = torch.zeros(VOCAB)
    logits[7] = 50.0  # a huge logit on a token that isn't a label
    logits[LABEL_IDS[1]] = 1.0
    probs = restricted_softmax(logits, LABEL_IDS[:2])
    assert probs.sum().item() == pytest.approx(1.0)
    assert probs[1] > probs[0]


def test_temperature_softens_and_sharpens():
    logits = logits_favouring(0, strength=2.0)
    base = restricted_softmax(logits, LABEL_IDS[:2])[0]
    assert restricted_softmax(logits, LABEL_IDS[:2], temperature=4.0)[0] < base
    assert restricted_softmax(logits, LABEL_IDS[:2], temperature=0.5)[0] > base


def test_agreement():
    p = {"a": 0.8, "b": 0.2}
    assert agreement([p]) is None
    assert agreement([p, p]) == pytest.approx(1.0)
    assert agreement([p, {"a": 0.2, "b": 0.8}]) == pytest.approx(0.4)


def test_agreement_never_negative():
    # Orders that disagree completely, with float sums a hair over 1 (a real CVE run
    # got -4e-9 here, which failed the answer models' confidence >= 0 check).
    p, q = {"a": 1.0 + 1e-8, "b": 0.0}, {"a": 0.0, "b": 1.0 + 1e-8}
    assert agreement([p, q]) == 0.0


def choice_request(debias) -> DecideRequest:
    q = ChoiceQuestion(question="Which?", options={"x": "X", "y": "Y", "z": "Z"})
    return DecideRequest(state="s", questions={"q": q}, debias=debias)


def test_pure_label_bias_averages_out():
    """A model that always says "A", whatever the options, knows nothing. With all three
    cyclic shifts, averaging gives a uniform answer and low confidence."""
    request = choice_request("auto")
    branches = build_prompt(request).branches
    logits = torch.stack([logits_favouring(0) for _ in branches])
    (result,) = read_branches(request, branches, logits, LABEL_IDS)
    assert list(result.probabilities) == ["x", "y", "z"]
    for p in result.probabilities.values():
        assert p == pytest.approx(1 / 3)
    assert result.confidence < 0.2


def test_consistent_answer_survives_permutation():
    """A model that picks option "y" wherever it's shown: confident, answer y."""
    request = choice_request("auto")
    branches = build_prompt(request).branches
    rows = [logits_favouring(b.keys.index("y")) for b in branches]
    (result,) = read_branches(request, branches, torch.stack(rows), LABEL_IDS)
    assert max(result.probabilities, key=result.probabilities.get) == "y"
    assert result.confidence == pytest.approx(1.0)
    assert len(result.per_permutation) == 3


def test_labels_map_back_to_keys():
    request = choice_request("off")
    branch = Branch("q", ("z", "x", "y"), "unused")
    (result,) = read_branches(request, [branch], logits_favouring(0)[None], LABEL_IDS)
    assert result.probabilities["z"] > 0.9  # label A was z in this branch
    assert result.confidence is None


def test_temperature_is_per_type():
    request = DecideRequest(
        state="s",
        questions={
            "c": ChoiceQuestion(question="?", options={"x": "X", "y": "Y"}),
            "y": YesNoQuestion(question="?"),
        },
        debias="off",
    )
    branches = build_prompt(request).branches
    logits = torch.stack([logits_favouring(0, 2.0)] * 2)
    c, y = read_branches(request, branches, logits, LABEL_IDS, temperature={"yesno": 10.0})
    assert c.probabilities["x"] > 0.8
    assert 0.5 < y.probabilities["yes"] < 0.6


def test_score_answer_is_expected_level():
    q = ScoreQuestion(question="?", levels=["low", "mid", "high"])
    probs = {"0": 0.2, "1": 0.5, "2": 0.3}
    a = to_answer(q, QuestionResult("s", "score", probs, [probs], None))
    assert a.value == pytest.approx(1.1)
    assert a.level == 1


def test_yesno_answer():
    probs = {"yes": 0.7, "no": 0.3}
    a = to_answer(YesNoQuestion(question="?"), QuestionResult("y", "yesno", probs, [probs], 0.9))
    assert a.p_yes == pytest.approx(0.7)
    assert a.confidence == 0.9

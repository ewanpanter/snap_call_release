"""The shared contract: every part of the project reads and writes these shapes.

Change it with care: the API, the data files, the eval and the baselines all depend on it.

Contents:
- SnapCall's own API: ``DecideRequest`` / ``DecideResponse`` for ``POST /v1/decide``.
- File formats: ``DataRecord`` (``data/*.jsonl``), ``Prediction`` (``predictions.jsonl``),
  ``Metrics`` (``metrics.json``), ``Cost`` (``cost.json``).
- A thin adapter to and from TypeSafe's Decisions API format, for the baseline.
- JSONL helpers and a JSON Schema exporter (``contract/schemas/``).

Conventions:
- Question ids are stable task names (``queue``, ``priority``, ``is_billing``), the same
  across every state in a dataset, so metrics can be grouped by question.
- Score levels are ordered low to high and addressed by 0-based index. Score
  probabilities are keyed by the index as a string (``"0"``, ``"1"``, ...), as in Jev.
- Yes/no probabilities are keyed ``"yes"`` and ``"no"``.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

SCHEMA_VERSION = "1"

# Tolerance on "probabilities sum to 1". Jev rounds to two decimals, so allow a little slack.
PROB_SUM_TOL = 0.02

QuestionType = Literal["choice", "score", "yesno"]
QuestionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{1,64}$")]
OptionKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{1,64}$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Split = Literal["train", "val", "test"]
State = str | dict[str, Any] | list[Any]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_distribution(probs: dict[str, float], expected_keys: Iterable[str], what: str) -> None:
    expected = set(expected_keys)
    if set(probs) != expected:
        raise ValueError(f"{what}: probability keys {sorted(probs)} != expected {sorted(expected)}")
    total = sum(probs.values())
    if not math.isclose(total, 1.0, abs_tol=PROB_SUM_TOL):
        raise ValueError(f"{what}: probabilities sum to {total:.4f}, expected 1")


def state_text(state: State) -> str:
    """The text the model reads. JSON states are serialised deterministically."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------------------


class ChoiceQuestion(_Model):
    """Pick one of the labelled options. Keys are what the answer returns."""

    type: Literal["choice"] = "choice"
    question: str = Field(min_length=1)
    options: dict[OptionKey, str] = Field(min_length=2, max_length=26)


class ScoreQuestion(_Model):
    """Place the state on an ordered scale. ``levels`` is ordered low to high."""

    type: Literal["score"] = "score"
    question: str = Field(min_length=1)
    levels: list[str] = Field(min_length=2, max_length=10)


class YesNoQuestion(_Model):
    """Is this statement true? ``yes``/``no`` optionally describe what each answer means."""

    type: Literal["yesno"] = "yesno"
    question: str = Field(min_length=1)
    yes: str | None = None
    no: str | None = None


Question = Annotated[ChoiceQuestion | ScoreQuestion | YesNoQuestion, Field(discriminator="type")]


def answer_keys(q: ChoiceQuestion | ScoreQuestion | YesNoQuestion) -> list[str]:
    """The probability keys an answer to ``q`` must use, in canonical order."""
    if isinstance(q, ChoiceQuestion):
        return list(q.options)
    if isinstance(q, ScoreQuestion):
        return [str(i) for i in range(len(q.levels))]
    return ["yes", "no"]


# --------------------------------------------------------------------------------------
# SnapCall API: POST /v1/decide
# --------------------------------------------------------------------------------------

Debias = Literal["off", "auto"] | Annotated[int, Field(ge=1, le=8)]


class DecideRequest(_Model):
    """``state`` is read once; every question is answered independently against it.

    ``debias``: ``"off"`` = one branch per question; ``"auto"`` = the default number of
    permutations per type; an int K = up to K permutations per question.
    """

    state: State
    questions: dict[QuestionId, Question] = Field(min_length=1)
    debias: Debias = "auto"
    seed: int = 0


class ChoiceAnswer(_Model):
    type: Literal["choice"] = "choice"
    choice: OptionKey
    probabilities: dict[OptionKey, Probability]
    confidence: Probability | None = None

    @model_validator(mode="after")
    def _check(self) -> ChoiceAnswer:
        if self.choice not in self.probabilities:
            raise ValueError(f"choice {self.choice!r} not in probabilities")
        _check_distribution(self.probabilities, self.probabilities, "choice")
        return self


class ScoreAnswer(_Model):
    """``value`` is the expected level index, so it can fall between levels.
    ``level`` is the most probable level index."""

    type: Literal["score"] = "score"
    value: NonNegativeFloat
    level: NonNegativeInt
    probabilities: dict[str, Probability]
    confidence: Probability | None = None

    @model_validator(mode="after")
    def _check(self) -> ScoreAnswer:
        n = len(self.probabilities)
        _check_distribution(self.probabilities, [str(i) for i in range(n)], "score")
        if self.level >= n or self.value > n - 1 + 1e-6:
            raise ValueError(f"score level/value out of range for {n} levels")
        return self


class YesNoAnswer(_Model):
    type: Literal["yesno"] = "yesno"
    p_yes: Probability
    confidence: Probability | None = None


Answer = Annotated[ChoiceAnswer | ScoreAnswer | YesNoAnswer, Field(discriminator="type")]


class Usage(_Model):
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0


class DecideResponse(_Model):
    """Answers are keyed by the request's question ids, in the same order."""

    model: str
    answers: dict[QuestionId, Answer]
    usage: Usage = Field(default_factory=Usage)
    latency_ms: NonNegativeFloat | None = None

    def check_against(self, request: DecideRequest) -> None:
        """Raise if the answers don't line up with the request's questions and options."""
        if list(self.answers) != list(request.questions):
            raise ValueError("answer ids do not match question ids")
        for qid, q in request.questions.items():
            a = self.answers[qid]
            if a.type != q.type:
                raise ValueError(f"{qid}: answer type {a.type} != question type {q.type}")
            if isinstance(a, ChoiceAnswer | ScoreAnswer):
                _check_distribution(a.probabilities, answer_keys(q), qid)


# --------------------------------------------------------------------------------------
# data/*.jsonl: one line per state, with its questions and gold answers
# --------------------------------------------------------------------------------------

# choice -> option key (str), score -> level index (int), yesno -> bool
Gold = StrictBool | StrictInt | StrictStr


def _check_gold(qid: str, q: ChoiceQuestion | ScoreQuestion | YesNoQuestion, g: Any) -> None:
    if isinstance(q, ChoiceQuestion):
        ok = isinstance(g, str) and g in q.options
    elif isinstance(q, ScoreQuestion):
        ok = isinstance(g, int) and not isinstance(g, bool) and 0 <= g < len(q.levels)
    else:
        ok = isinstance(g, bool)
    if not ok:
        raise ValueError(f"{qid}: gold {g!r} is not valid for a {q.type} question")


class DataRecord(_Model):
    state_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: Split
    state: State
    questions: dict[QuestionId, Question] = Field(min_length=1)
    gold: dict[QuestionId, Gold]
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> DataRecord:
        if set(self.gold) != set(self.questions):
            raise ValueError("gold must have exactly one entry per question")
        for qid, q in self.questions.items():
            _check_gold(qid, q, self.gold[qid])
        return self

    def to_request(self, debias: Debias = "auto", seed: int = 0) -> DecideRequest:
        return DecideRequest(state=self.state, questions=self.questions, debias=debias, seed=seed)


# --------------------------------------------------------------------------------------
# predictions.jsonl: one line per (system, state, question)
# --------------------------------------------------------------------------------------


class Prediction(_Model):
    """One system's answer to one question.

    ``answer``: choice -> option key, score -> most probable level index, yesno -> bool.
    ``value``: score only, the expected level index.
    ``latency_ms``: wall time of the whole request this question was part of (put the
    request's question count and debias setting in ``meta`` for the latency sweep).
    ``error``: set when the system failed on this question; answer/probabilities are then null.
    """

    system: str = Field(min_length=1)
    run_id: str | None = None
    state_id: str = Field(min_length=1)
    question_id: QuestionId
    type: QuestionType
    answer: Gold | None
    probabilities: dict[str, Probability] | None
    confidence: Probability | None = None
    value: NonNegativeFloat | None = None
    latency_ms: NonNegativeFloat | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Prediction:
        if self.error is not None:
            return self
        if self.answer is None or self.probabilities is None:
            raise ValueError("answer and probabilities are required unless error is set")
        a, p = self.answer, self.probabilities
        if self.type == "choice":
            if not isinstance(a, str) or a not in p:
                raise ValueError("choice answer must be an option key present in probabilities")
            _check_distribution(p, p, "choice")
        elif self.type == "score":
            if not isinstance(a, int) or isinstance(a, bool):
                raise ValueError("score answer must be a level index (int)")
            _check_distribution(p, [str(i) for i in range(len(p))], "score")
            if not 0 <= a < len(p):
                raise ValueError("score answer out of range")
            if self.value is None:
                raise ValueError("score predictions must set value")
        else:
            if not isinstance(a, bool):
                raise ValueError("yesno answer must be a bool")
            _check_distribution(p, ["yes", "no"], "yesno")
        return self


def predictions_from_response(
    response: DecideResponse,
    *,
    system: str,
    state_id: str,
    run_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> list[Prediction]:
    """Flatten one DecideResponse into prediction lines."""
    out = []
    for qid, a in response.answers.items():
        common = dict(
            system=system,
            run_id=run_id,
            state_id=state_id,
            question_id=qid,
            type=a.type,
            confidence=a.confidence,
            latency_ms=response.latency_ms,
            meta=dict(meta or {}),
        )
        if isinstance(a, ChoiceAnswer):
            out.append(Prediction(answer=a.choice, probabilities=a.probabilities, **common))
        elif isinstance(a, ScoreAnswer):
            out.append(
                Prediction(answer=a.level, probabilities=a.probabilities, value=a.value, **common)
            )
        else:
            probs = {"yes": a.p_yes, "no": 1.0 - a.p_yes}
            out.append(Prediction(answer=a.p_yes >= 0.5, probabilities=probs, **common))
    return out


# --------------------------------------------------------------------------------------
# metrics.json
# --------------------------------------------------------------------------------------


class ReliabilityBin(_Model):
    lo: Probability
    hi: Probability
    count: NonNegativeInt
    mean_confidence: Probability | None
    accuracy: Probability | None


class TypeMetrics(_Model):
    """Metrics over a set of predictions (all, one type, or one question id).

    accuracy: top answer == gold (yesno: p_yes >= 0.5). ece: expected calibration error of
    the top-answer probability, 10 equal-width bins. brier: multi-class Brier score (sum of
    squared errors over the answer keys). mae: score only, |value - gold index|.
    """

    n: NonNegativeInt
    n_errors: NonNegativeInt = 0
    accuracy: Probability | None = None
    ece: NonNegativeFloat | None = None
    brier: NonNegativeFloat | None = None
    log_loss: NonNegativeFloat | None = None
    mae: NonNegativeFloat | None = None
    reliability: list[ReliabilityBin] | None = None


class LatencyStats(_Model):
    n: NonNegativeInt
    mean_ms: NonNegativeFloat
    p50_ms: NonNegativeFloat
    p95_ms: NonNegativeFloat


class Metrics(_Model):
    schema_version: Literal["1"] = SCHEMA_VERSION
    system: str
    run_id: str | None = None
    dataset: str | None = None
    split: Split
    created_at: datetime
    n_states: NonNegativeInt
    n_questions: NonNegativeInt
    overall: TypeMetrics
    by_type: dict[QuestionType, TypeMetrics] = Field(default_factory=dict)
    by_question: dict[QuestionId, TypeMetrics] = Field(default_factory=dict)
    latency: LatencyStats | None = None
    # Fitted temperature per question type (1.0 = uncalibrated).
    temperature: dict[QuestionType, float] | None = None
    # Mean total-variation distance between permutations of the same question (0 = order-free).
    order_sensitivity: NonNegativeFloat | None = None
    # Anything component-specific that doesn't fit above. Keys should be snake_case.
    extra: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# cost.json
# --------------------------------------------------------------------------------------


class ComputeCost(_Model):
    resource: str  # e.g. "nvidia-l4", "cpu-github-runner"
    count: int = Field(default=1, ge=1)
    minutes: NonNegativeFloat
    spot: bool = False
    rate_per_hour: NonNegativeFloat  # per unit, in Cost.currency; check current pricing
    cost: NonNegativeFloat

    @model_validator(mode="after")
    def _check(self) -> ComputeCost:
        expected = self.count * self.minutes / 60 * self.rate_per_hour
        if not math.isclose(self.cost, expected, rel_tol=1e-3, abs_tol=1e-6):
            raise ValueError(f"compute cost {self.cost} != count*minutes/60*rate = {expected}")
        return self


class ApiCost(_Model):
    provider: str  # e.g. "typesafe", "vertex-gemini"
    requests: NonNegativeInt
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cost: NonNegativeFloat  # in Cost.currency
    raw_cost: NonNegativeFloat | None = None  # as billed, before FX conversion
    raw_currency: str | None = None


class Cost(_Model):
    schema_version: Literal["1"] = SCHEMA_VERSION
    run_id: str
    job: str  # workflow / script name
    system: str | None = None
    created_at: datetime
    currency: str = "GBP"
    compute: list[ComputeCost] = Field(default_factory=list)
    api: list[ApiCost] = Field(default_factory=list)
    total: NonNegativeFloat
    n_decisions: NonNegativeInt | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Cost:
        expected = sum(c.cost for c in self.compute) + sum(a.cost for a in self.api)
        if not math.isclose(self.total, expected, rel_tol=1e-3, abs_tol=1e-6):
            raise ValueError(f"total {self.total} != sum of items {expected}")
        return self

    @property
    def cost_per_1k_decisions(self) -> float | None:
        if not self.n_decisions:
            return None
        return self.total / self.n_decisions * 1000


# --------------------------------------------------------------------------------------
# TypeSafe adapter (baseline only)
#
# Format from the published Decisions API reference (OpenRouter, Sep 2026):
# request  {model, state, questions: {id: {type, instructions, criteria}}}
#   choice criteria = {key: description}, score criteria = [level, ...] (low -> high),
#   noul criteria = {"true": ..., "false": ...}
# response {id, model, answers: {id: answer}, usage: {input_tokens, output_tokens, cost}}
#   choice {choice, confidence, probabilities}, score {score, confidence, probabilities,
#   legend}, noul {noul}
# --------------------------------------------------------------------------------------

TYPESAFE_DEFAULT_MODEL = "typesafe/jev-1.13"


def to_typesafe_request(request: DecideRequest, model: str = TYPESAFE_DEFAULT_MODEL) -> dict:
    questions: dict[str, dict] = {}
    for qid, q in request.questions.items():
        if isinstance(q, ChoiceQuestion):
            questions[qid] = {"type": "choice", "instructions": q.question, "criteria": q.options}
        elif isinstance(q, ScoreQuestion):
            questions[qid] = {"type": "score", "instructions": q.question, "criteria": q.levels}
        else:
            body: dict[str, Any] = {"type": "noul", "instructions": q.question}
            if q.yes is not None or q.no is not None:
                body["criteria"] = {"true": q.yes or "Yes", "false": q.no or "No"}
            questions[qid] = body
    return {"model": model, "state": request.state, "questions": questions}


def _criterion_text(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)


def from_typesafe_request(body: dict) -> DecideRequest:
    questions: dict[str, Any] = {}
    for qid, q in body["questions"].items():
        text = _criterion_text(q["instructions"])
        crit = q.get("criteria")
        if q["type"] == "choice":
            opts = {k: _criterion_text(v) for k, v in crit.items()}
            questions[qid] = ChoiceQuestion(question=text, options=opts)
        elif q["type"] == "score":
            questions[qid] = ScoreQuestion(question=text, levels=[_criterion_text(v) for v in crit])
        elif q["type"] == "noul":
            crit = crit or {}
            yes, no = crit.get("true"), crit.get("false")
            questions[qid] = YesNoQuestion(
                question=text,
                yes=None if yes is None else _criterion_text(yes),
                no=None if no is None else _criterion_text(no),
            )
        else:
            raise ValueError(f"{qid}: unknown TypeSafe question type {q['type']!r}")
    return DecideRequest(state=body["state"], questions=questions)


def _renormalise(probs: dict[str, float]) -> dict[str, float]:
    """Jev rounds to two decimals; renormalise so downstream sums are exact."""
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()} if total > 0 else probs


def from_typesafe_response(
    body: dict, request: DecideRequest, latency_ms: float | None = None
) -> DecideResponse:
    answers: dict[str, Any] = {}
    for qid, q in request.questions.items():
        a = body["answers"][qid]
        if isinstance(q, ChoiceQuestion):
            probs = _renormalise({k: float(a["probabilities"].get(k, 0.0)) for k in q.options})
            answers[qid] = ChoiceAnswer(
                choice=a["choice"], probabilities=probs, confidence=a.get("confidence")
            )
        elif isinstance(q, ScoreQuestion):
            keys = answer_keys(q)
            probs = _renormalise({k: float(a["probabilities"].get(k, 0.0)) for k in keys})
            answers[qid] = ScoreAnswer(
                value=min(max(float(a["score"]), 0.0), len(keys) - 1),
                level=int(max(keys, key=lambda k: probs[k])),
                probabilities=probs,
                confidence=a.get("confidence"),
            )
        else:
            answers[qid] = YesNoAnswer(p_yes=float(a["noul"]), confidence=a.get("confidence"))
    usage = body.get("usage", {})
    return DecideResponse(
        model=body.get("model", "typesafe"),
        answers=answers,
        usage=Usage(
            input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0)
        ),
        latency_ms=latency_ms,
    )


def to_typesafe_response(response: DecideResponse, request: DecideRequest) -> dict:
    """SnapCall answers in TypeSafe's shape, so one client can call either system."""
    answers: dict[str, dict] = {}
    for qid, a in response.answers.items():
        if isinstance(a, ChoiceAnswer):
            answers[qid] = {
                "type": "choice",
                "choice": a.choice,
                "confidence": a.confidence,
                "probabilities": a.probabilities,
            }
        elif isinstance(a, ScoreAnswer):
            q = request.questions[qid]
            assert isinstance(q, ScoreQuestion)
            answers[qid] = {
                "type": "score",
                "score": a.value,
                "confidence": a.confidence,
                "probabilities": a.probabilities,
                "legend": {str(i): lvl for i, lvl in enumerate(q.levels)},
            }
        else:
            answers[qid] = {"type": "noul", "noul": a.p_yes}
    return {
        "model": response.model,
        "answers": answers,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


# --------------------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------------------


def read_jsonl[M: BaseModel](path: str | Path, model: type[M]) -> Iterator[M]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield model.model_validate_json(line)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e


def write_jsonl(path: str | Path, rows: Iterable[BaseModel]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json(exclude_none=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------------------
# JSON Schema export (contract/schemas/*.schema.json, checked for drift by tests)
# --------------------------------------------------------------------------------------

CONTRACT_MODELS: dict[str, type[BaseModel]] = {
    "decide_request": DecideRequest,
    "decide_response": DecideResponse,
    "data_record": DataRecord,
    "prediction": Prediction,
    "metrics": Metrics,
    "cost": Cost,
}


def json_schemas() -> dict[str, dict]:
    return {name: m.model_json_schema(mode="validation") for name, m in CONTRACT_MODELS.items()}


def export_json_schemas(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, schema in json_schemas().items():
        p = out / f"{name}.schema.json"
        p.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(p)
    return written


def export_json_schemas_cli() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "contract/schemas"
    for p in export_json_schemas(out):
        print(p)


if __name__ == "__main__":
    export_json_schemas_cli()

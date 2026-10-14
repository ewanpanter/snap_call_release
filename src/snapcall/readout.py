"""Turn next-token logits into answers. No generation, no parsing.

At the last token of a branch the model scores its whole vocabulary (~150k tokens for
Qwen3). We keep only the logits of the labels that branch offered (``A``, ``B``, ...) and
softmax over those, which gives a proper distribution over the allowed answers.

Then, per question:

1. Map each branch's label probabilities back to answer keys (the branch knows which key
   sits behind which label).
2. Average over the question's permutations. That cancels most of the model's preference
   for particular labels and positions.
3. ``confidence`` = how much the permutations agree: 1 - the mean total-variation distance
   between them. 1 means option order made no difference. With one permutation there's
   nothing to compare, so confidence is ``None``.
4. Build the typed answer: ``choice`` -> the most probable key; ``score`` -> the expected
   level index (can fall between levels) and the most probable level; ``yesno`` -> P(yes).

``decide.py`` runs the whole pipeline (prompt -> forward -> readout) for one request.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch

from snapcall.prompt import Branch
from snapcall.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecideRequest,
    QuestionType,
    ScoreAnswer,
    ScoreQuestion,
    YesNoAnswer,
    answer_keys,
)


def restricted_softmax(
    logits: torch.Tensor, label_ids: list[int], temperature: float = 1.0
) -> torch.Tensor:
    """Softmax over just ``label_ids``. ``logits``: ``[..., vocab]`` -> ``[..., n_labels]``.

    ``temperature`` > 1 softens the distribution, < 1 sharpens it (calibration).
    Keeps gradients, so LoRA training can use it as its output layer.
    """
    return torch.softmax(logits[..., label_ids].float() / temperature, dim=-1)


@dataclass(frozen=True)
class QuestionResult:
    """Everything the readout knows about one question, before it's typed into an answer."""

    question_id: str
    type: QuestionType
    probabilities: dict[str, float]  # averaged over permutations, canonical key order
    per_permutation: list[dict[str, float]]  # one distribution per branch, same key order
    confidence: float | None


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p[k] - q[k]) for k in p)


def agreement(dists: list[dict[str, float]]) -> float | None:
    """1 - mean pairwise total-variation distance. ``None`` for a single distribution."""
    pairs = list(combinations(dists, 2))
    if not pairs:
        return None
    # Clamped: when the orders disagree completely, float rounding can give -4e-9, which
    # fails the answer models' confidence >= 0 check and loses the whole state.
    return min(1.0, max(0.0, 1.0 - sum(total_variation(p, q) for p, q in pairs) / len(pairs)))


def read_branches(
    request: DecideRequest,
    branches: list[Branch],
    branch_logits: torch.Tensor,
    label_ids: list[int],
    temperature: dict[QuestionType, float] | None = None,
) -> list[QuestionResult]:
    """Logits at each branch's last token -> one ``QuestionResult`` per question."""
    per_question: dict[str, list[dict[str, float]]] = {qid: [] for qid in request.questions}
    for branch, logits in zip(branches, branch_logits, strict=True):
        q = request.questions[branch.question_id]
        t = (temperature or {}).get(q.type, 1.0)
        probs = restricted_softmax(logits, label_ids[: len(branch.keys)], t).tolist()
        # label i -> answer key branch.keys[i]; store in canonical key order.
        by_key = dict(zip(branch.keys, probs, strict=True))
        per_question[branch.question_id].append({k: by_key[k] for k in answer_keys(q)})

    results = []
    for qid, dists in per_question.items():
        q = request.questions[qid]
        mean = {k: sum(d[k] for d in dists) / len(dists) for k in answer_keys(q)}
        results.append(QuestionResult(qid, q.type, mean, dists, agreement(dists)))
    return results


def to_answer(question, result: QuestionResult) -> ChoiceAnswer | ScoreAnswer | YesNoAnswer:
    probs, conf = result.probabilities, result.confidence
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(choice=max(probs, key=probs.get), probabilities=probs, confidence=conf)
    if isinstance(question, ScoreQuestion):
        expected = sum(int(k) * p for k, p in probs.items())
        return ScoreAnswer(
            value=min(expected, len(probs) - 1),  # guard against float overshoot
            level=int(max(probs, key=probs.get)),
            probabilities=probs,
            confidence=conf,
        )
    # Only yes and no are offered, so P(yes) is already renormalised against P(no).
    return YesNoAnswer(p_yes=probs["yes"], confidence=conf)

"""TF-IDF nearest-neighbour baseline: the "is this dataset template-like?" gate.

For each test state, find the most similar train state (cosine on TF-IDF of words) and copy
its gold answers. If that scores close to the models, the test set mostly repeats the train
set (that's what exposed telco: 90% of telco states fully right), and fine-tuned scores mean
"learned this dataset", not "learned the task". Run it before anyone scores a new dataset
(Decision 16).

Stdlib only, so it runs anywhere:

    uv run --no-sync python data/nn_baseline.py outputs/data/cfpb_train.jsonl \\
        outputs/data/cfpb_test.jsonl

Prints, per question: nearest-neighbour accuracy, the majority-class baseline, and the share
of test states that are near-copies of a train state (cosine >= 0.9).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# CFPB redacts names, dates and numbers as XXXX; they carry no signal.
_TOKEN = re.compile(r"[a-z][a-z']+")
# TeleLogs states are tables of numbers, so there the numbers are the signal.
_TOKEN_WITH_NUMBERS = re.compile(r"[a-z][a-z']+|-?\d+(?:\.\d+)?")
NEAR_COPY = 0.9


def tokens(text: str, keep_numbers: bool = False) -> list[str]:
    pattern = _TOKEN_WITH_NUMBERS if keep_numbers else _TOKEN
    return [t for t in pattern.findall(text.lower()) if not set(t) <= {"x"}]


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True)


def read(path: Path, dataset: str | None) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if dataset is None or r["dataset"] == dataset]


def tfidf(docs: list[list[str]], idf: dict[str, float]) -> list[dict[str, float]]:
    """Sublinear term frequency times idf, L2-normalised. Terms without an idf are dropped."""
    out = []
    for doc in docs:
        v = {t: (1 + math.log(c)) * idf[t] for t, c in Counter(doc).items() if t in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({t: x / norm for t, x in v.items()})
    return out


def nearest(train: list[dict[str, float]], test: list[dict[str, float]]) -> list[tuple[int, float]]:
    """(index of the most similar train vector, cosine) for each test vector."""
    index: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i, v in enumerate(train):
        for t, x in v.items():
            index[t].append((i, x))
    out = []
    for v in test:
        scores: dict[int, float] = defaultdict(float)
        for t, x in v.items():
            for i, y in index.get(t, ()):
                scores[i] += x * y
        out.append(max(scores.items(), key=lambda kv: (kv[1], -kv[0])) if scores else (0, 0.0))
    return out


def nn_baseline(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    max_df: float = 0.5,
    keep_numbers: bool = False,
) -> dict[str, Any]:
    train_docs = [tokens(state_text(r["state"]), keep_numbers) for r in train_rows]
    test_docs = [tokens(state_text(r["state"]), keep_numbers) for r in test_rows]
    n = len(train_docs)
    df = Counter(t for doc in train_docs for t in set(doc))
    # Words in more than max_df of train states are near-stopwords: drop them (also speed).
    idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items() if d / n <= max_df}
    hits = nearest(tfidf(train_docs, idf), tfidf(test_docs, idf))

    questions = list(test_rows[0]["gold"])
    per_q: dict[str, Any] = {}
    for q in questions:
        majority = Counter(json.dumps(r["gold"][q]) for r in train_rows).most_common(1)[0][0]
        nn_right = [
            train_rows[i]["gold"][q] == r["gold"][q]
            for r, (i, _) in zip(test_rows, hits, strict=True)
        ]
        per_q[q] = {
            "nn_accuracy": round(sum(nn_right) / len(test_rows), 4),
            "majority_accuracy": round(
                sum(json.dumps(r["gold"][q]) == majority for r in test_rows) / len(test_rows), 4
            ),
            "majority_answer": json.loads(majority),
        }
    all_right = sum(
        all(train_rows[i]["gold"][q] == r["gold"][q] for q in questions)
        for r, (i, _) in zip(test_rows, hits, strict=True)
    )
    sims = sorted(s for _, s in hits)
    return {
        "train": len(train_rows),
        "test": len(test_rows),
        "questions": per_q,
        "nn_all_questions_right": round(all_right / len(test_rows), 4),
        "nn_cosine": {
            "p50": round(sims[len(sims) // 2], 3),
            "p90": round(sims[int(0.9 * (len(sims) - 1))], 3),
            "max": round(sims[-1], 3),
            f"share_ge_{NEAR_COPY}": round(sum(s >= NEAR_COPY for s in sims) / len(sims), 4),
        },
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        f"train {result['train']}, test {result['test']}",
        "| question | NN accuracy | majority class |",
        "|---|---|---|",
    ]
    for q, r in result["questions"].items():
        lines.append(
            f"| {q} | {r['nn_accuracy']:.3f} | {r['majority_accuracy']:.3f} "
            f"({r['majority_answer']}) |"
        )
    c = result["nn_cosine"]
    lines.append(
        f"all questions right: {result['nn_all_questions_right']:.3f}; NN cosine p50 {c['p50']},"
        f" p90 {c['p90']}, max {c['max']}, near-copies (>= {NEAR_COPY}) "
        f"{c[f'share_ge_{NEAR_COPY}']:.1%}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("train", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("--dataset", help="only rows with this `dataset` value")
    p.add_argument("--json", type=Path, help="also write the result here")
    p.add_argument(
        "--keep-numbers",
        action="store_true",
        help="count numbers as terms (for tabular states such as TeleLogs)",
    )
    a = p.parse_args(argv)
    result = nn_baseline(
        read(a.train, a.dataset), read(a.test, a.dataset), keep_numbers=a.keep_numbers
    )
    if a.json:
        a.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()

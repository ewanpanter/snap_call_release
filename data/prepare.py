"""Download the public datasets, map their labels to SnapCall questions, split, write JSONL.

Stdlib only, plus the contract models, so it runs on any machine and on a
plain GitHub runner:

    uv run python data/prepare.py --out outputs/data

Writes ``train.jsonl``, ``val.jsonl`` and ``test.jsonl`` (both datasets in each file; the
``dataset`` field tells them apart) plus ``stats.json``. Every record is a ``DataRecord``.

Datasets (pinned to a Hub revision, see ``data/README.md`` for licences):

- ``customer-support-tickets``: Tobi-Bueck/customer-support-tickets, English rows only.
  Questions: ``queue`` (choice, 10), ``type`` (choice, 4), ``priority`` (score, 3 levels),
  ``is_billing`` (yesno, derived from queue and tags).
- ``bitext-telco``: bitext/Bitext-telco-llm-chatbot-training-dataset.
  Questions: ``intent`` (choice, 26), ``category`` (choice, 7), ``wants_to_leave`` (yesno,
  derived from intent).

Sampling: each dataset is deduplicated on normalised text, then a stratified sample of
``test_size / 0.15`` records is drawn and split 70/15/15, so test gets ``test_size`` states.
``--full`` keeps every record instead.

Stratification is systematic: records are sorted by their label tuple (queue, type, priority,
... ) with a seeded random tie-break, then sampled and assigned to splits in an interleaved
pattern. That keeps every label combination in proportion across the splits, exactly for the
first key and closely for the rest, with no special case for rare combinations.

Separate sets, written on their own so the main splits never change: ``--ood-only`` (the
forgetting check, ``ood_test.jsonl``) and ``--cfpb`` (the held-out CFPB complaints domain,
``cfpb_{train,val,test}.jsonl``, Decision 16) and ``--telelogs`` (5G root-cause
analysis, ``telelogs_{train,val,test}.jsonl``, Decision 17) and ``--cve`` (Red Hat CVE
advisories, ``cve_{train,val,test}.jsonl``, Decision 17).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snapcall.schema import (
    ChoiceQuestion,
    DataRecord,
    ScoreQuestion,
    YesNoQuestion,
    write_jsonl,
)

SEED = 2609
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# A label is flagged as heavily imbalanced when its rarest value has under this share, or
# the most common value is over IMBALANCE_RATIO times the rarest.
IMBALANCE_MIN_SHARE = 0.03
IMBALANCE_RATIO = 10.0

csv.field_size_limit(10**8)


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    filename: str
    sha256: str
    licence: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.filename}"
        )


TICKETS_SOURCE = Source(
    repo="Tobi-Bueck/customer-support-tickets",
    revision="ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb",
    filename="aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    sha256="f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0",
    licence="CC BY-NC 4.0",
)

TELCO_SOURCE = Source(
    repo="bitext/Bitext-telco-llm-chatbot-training-dataset",
    revision="eacc593a96d75ec6da0887f081ff0dfc442fccab",
    filename="bitext-telco-llm-chatbot-training-dataset.csv",
    sha256="0905820a880783d1a00f92d4f519488479f9eefe770cbfb5737f9462135d6c12",
    licence="CDLA-Sharing 1.0",
)


def fetch(source: Source, cache_dir: Path) -> Path:
    """Download ``source`` into ``cache_dir`` (once) and check its hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{source.revision[:12]}-{source.filename}"
    if not path.exists():
        tmp = path.with_suffix(".part")
        print(f"downloading {source.url}", file=sys.stderr)
        with urllib.request.urlopen(source.url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source.sha256:
        raise RuntimeError(f"{path}: sha256 {digest} != pinned {source.sha256}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# --------------------------------------------------------------------------------------
# Customer-support tickets
# --------------------------------------------------------------------------------------

TICKETS = "customer-support-tickets"

# Source queue name -> (option key, description shown to the model)
QUEUES = {
    "Technical Support": ("technical_support", "Technical Support: technical issues"),
    "Product Support": ("product_support", "Product Support: problems with a specific product"),
    "Customer Service": ("customer_service", "Customer Service: general service requests"),
    "IT Support": ("it_support", "IT Support: internal IT and infrastructure"),
    "Billing and Payments": ("billing", "Billing and Payments: charges, invoices, payments"),
    "Returns and Exchanges": ("returns", "Returns and Exchanges: sending products back"),
    "Service Outages and Maintenance": (
        "outages",
        "Service Outages and Maintenance: service interruptions",
    ),
    "Sales and Pre-Sales": ("sales", "Sales and Pre-Sales: buying or pricing questions"),
    "Human Resources": ("hr", "Human Resources: employee and HR matters"),
    "General Inquiry": ("general", "General Inquiry: anything else"),
}

TYPES = {
    "Incident": ("incident", "Incident: something unexpected is broken now"),
    "Request": ("request", "Request: asking for something to be done or explained"),
    "Problem": ("problem", "Problem: an underlying issue behind repeated incidents"),
    "Change": ("change", "Change: a planned change or update"),
}

# The source README lists Low / Medium / Critical; the CSV uses low / medium / high.
PRIORITIES = ["low", "medium", "high"]

BILLING_TAGS = {"Billing", "Payment", "Refund", "Invoice"}

TICKET_QUESTIONS = {
    "queue": ChoiceQuestion(
        question="Which support queue should handle this ticket?",
        options={key: desc for key, desc in QUEUES.values()},
    ),
    "type": ChoiceQuestion(
        question="What kind of ticket is this?",
        options={key: desc for key, desc in TYPES.values()},
    ),
    "priority": ScoreQuestion(question="How urgent is this ticket?", levels=PRIORITIES),
    "is_billing": YesNoQuestion(
        question="This ticket is about billing, payments or a refund.",
        yes="The customer is asking about charges, invoices, payments or refunds",
        no="The ticket is about something else",
    ),
}


def ticket_state(row: dict[str, str]) -> str:
    # The CSV stores newlines as a literal backslash-n.
    subject = row["subject"].replace("\\n", "\n").strip()
    body = row["body"].replace("\\n", "\n").strip()
    return f"Subject: {subject}\n\n{body}" if subject else body


def ticket_tags(row: dict[str, str]) -> list[str]:
    return [row[k] for k in sorted(row) if k.startswith("tag_") and row[k]]


def ticket_record(i: int, row: dict[str, str]) -> dict[str, Any]:
    tags = ticket_tags(row)
    queue = QUEUES[row["queue"]][0]
    return dict(
        state_id=f"tickets-{i:06d}",
        dataset=TICKETS,
        state=ticket_state(row),
        questions=TICKET_QUESTIONS,
        gold={
            "queue": queue,
            "type": TYPES[row["type"]][0],
            "priority": PRIORITIES.index(row["priority"]),
            "is_billing": queue == "billing" or bool(BILLING_TAGS & set(tags)),
        },
        meta={
            "source": TICKETS_SOURCE.repo,
            "source_row": i,
            "language": row["language"],
            "version": row["version"],
            "tags": tags,
        },
    )


def load_tickets(path: Path, languages: set[str]) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(read_csv(path)):
        if row["language"] not in languages or not row["body"].strip():
            continue
        out.append(ticket_record(i, row))
    return out


# --------------------------------------------------------------------------------------
# Bitext telco
# --------------------------------------------------------------------------------------

TELCO = "bitext-telco"

INTENTS = {
    "activate_call_management_services": "Turn on a calling feature (call waiting, forwarding)",
    "activate_phone": "Activate a phone or SIM",
    "activate_roaming": "Turn on roaming",
    "cancel_plan": "Cancel a plan or contract",
    "change_plan": "Change to a different plan",
    "change_provider": "Switch to another provider",
    "check_cancellation_fee": "Ask about the cancellation fee",
    "check_excess_data_charges": "Ask about extra data charges",
    "check_mobile_payments": "Check payments made with the phone",
    "check_signal_coverage": "Check signal coverage in an area",
    "check_usage": "Check call, text or data usage",
    "customer_service": "Contact customer service",
    "deactivate_call_management_services": "Turn off a calling feature",
    "deactivate_phone": "Deactivate a phone or SIM",
    "dispute_invoice": "Dispute a charge on an invoice",
    "get_compensation": "Ask for compensation",
    "human_agent": "Speak to a human agent",
    "install_internet": "Get home internet installed",
    "invoices": "Get or view invoices",
    "pay": "Pay a bill",
    "payment_methods": "Ask which payment methods are accepted",
    "report_poor_signal_coverage": "Report poor signal",
    "report_problem": "Report a problem with the service",
    "schedule_payments": "Set up scheduled payments",
    "set_usage_limits": "Set usage limits",
    "sign_up_for_plan": "Sign up for a new plan",
}

CATEGORIES = {
    "BILLING": ("billing", "Billing: invoices and disputed charges"),
    "COMPLAINTS": ("complaints", "Complaints: reporting problems or asking for compensation"),
    "CONSUMPTION": ("consumption", "Consumption: usage, data charges and limits"),
    "CONTACT": ("contact", "Contact: reaching customer service or a person"),
    "PAYMENT": ("payment", "Payment: paying and payment methods"),
    "SERVICES": ("services", "Services: activating or setting up services"),
    "SUBSCRIPTION": ("subscription", "Subscription: signing up, changing or cancelling a plan"),
}

LEAVING_INTENTS = {"cancel_plan", "change_provider", "check_cancellation_fee", "deactivate_phone"}

TELCO_QUESTIONS = {
    "intent": ChoiceQuestion(
        question="What does the customer want to do?",
        options=dict(sorted(INTENTS.items())),
    ),
    "category": ChoiceQuestion(
        question="Which category does this message belong to?",
        options={key: desc for key, desc in CATEGORIES.values()},
    ),
    "wants_to_leave": YesNoQuestion(
        question="The customer is thinking about cancelling or leaving.",
        yes="They want to cancel, switch provider, deactivate, or ask what leaving costs",
        no="They want something else",
    ),
}


def telco_record(i: int, row: dict[str, str]) -> dict[str, Any]:
    return dict(
        state_id=f"telco-{i:06d}",
        dataset=TELCO,
        state=row["instruction"].strip(),
        questions=TELCO_QUESTIONS,
        gold={
            "intent": row["intent"],
            "category": CATEGORIES[row["category"]][0],
            "wants_to_leave": row["intent"] in LEAVING_INTENTS,
        },
        meta={"source": TELCO_SOURCE.repo, "source_row": i, "bitext_tags": row["tags"]},
    )


def load_telco(path: Path) -> list[dict[str, Any]]:
    return [telco_record(i, row) for i, row in enumerate(read_csv(path))]


# --------------------------------------------------------------------------------------
# Dedup, sample, split
# --------------------------------------------------------------------------------------


def dedupe(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop repeated states (same normalised text), keeping the first. Texts that repeat
    with *different* gold labels are dropped entirely, since no answer is right for them."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(_norm(r["state"]), []).append(r)
    out = []
    for group in groups.values():
        golds = {json.dumps(r["gold"], sort_keys=True) for r in group}
        if len(golds) == 1:
            out.append(group[0])
    out.sort(key=lambda r: r["meta"]["source_row"])
    return out, len(records) - len(out)


def strat_key(record: dict[str, Any]) -> tuple:
    """Sort key for stratification: the gold answers, in question order."""
    return tuple(str(record["gold"][q]) for q in record["questions"])


def stratified_order(records: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    tagged = [(strat_key(r), rng.random(), r) for r in records]
    tagged.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in tagged]


def systematic_sample(ordered: list[Any], n: int) -> list[Any]:
    """Every (len/n)-th item of a stratum-sorted list: a proportional stratified sample."""
    if n >= len(ordered):
        return list(ordered)
    step = len(ordered) / n
    return [ordered[int(j * step)] for j in range(n)]


def interleave(ratios: dict[str, float], n: int) -> list[str]:
    """Assign ``n`` positions to splits so every prefix is as close as possible to
    ``ratios`` (largest deficit first). Applied to a stratum-sorted list, each stratum is
    split in proportion."""
    assigned = dict.fromkeys(ratios, 0)
    out = []
    for i in range(1, n + 1):
        name = max(ratios, key=lambda s: ratios[s] * i - assigned[s])
        assigned[name] += 1
        out.append(name)
    return out


def sample_and_split(
    records: list[dict[str, Any]],
    test_size: int | None,
    seed: int,
    splits: dict[str, float] = SPLITS,
) -> list[DataRecord]:
    rng = random.Random(seed)
    ordered = stratified_order(records, rng)
    if test_size:
        ordered = systematic_sample(ordered, round(test_size / splits["test"]))
    out = []
    for r, split in zip(ordered, interleave(splits, len(ordered)), strict=True):
        out.append(DataRecord(split=split, **r))
    out.sort(key=lambda r: r.state_id)
    return out


# --------------------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------------------


def _dist(values: Iterable[Any]) -> dict[str, int]:
    c = Counter(str(v) for v in values)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def imbalance(dist: dict[str, int]) -> dict[str, Any]:
    total = sum(dist.values())
    lo, hi = min(dist.values()), max(dist.values())
    min_share = lo / total if total else 0.0
    ratio = hi / lo if lo else float("inf")
    return {
        "min_share": round(min_share, 4),
        "max_min_ratio": round(ratio, 2),
        "heavy": min_share < IMBALANCE_MIN_SHARE or ratio > IMBALANCE_RATIO,
    }


def summarise(records: list[DataRecord], extra: dict[str, Any]) -> dict[str, Any]:
    splits = extra.get("splits", SPLITS)
    stats: dict[str, Any] = {"seed": extra["seed"], "splits": splits, "datasets": {}}
    for ds in sorted({r.dataset for r in records}):
        rows = [r for r in records if r.dataset == ds]
        lengths = [len(r.state) for r in rows if isinstance(r.state, str)]
        labels = {}
        for qid, q in rows[0].questions.items():
            by_split = {
                s: _dist(r.gold[qid] for r in rows if r.split == s)
                for s in ("train", "val", "test")
            }
            overall = _dist(r.gold[qid] for r in rows)
            labels[qid] = {
                "type": q.type,
                "n_options": len(q.options)
                if isinstance(q, ChoiceQuestion)
                else len(q.levels)
                if isinstance(q, ScoreQuestion)
                else 2,
                "overall": overall,
                "by_split": by_split,
                "imbalance": imbalance(overall),
            }
        stats["datasets"][ds] = {
            **extra["datasets"][ds],
            "records": {s: sum(r.split == s for r in rows) for s in ("train", "val", "test")},
            "questions_per_record": _dist(len(r.questions) for r in rows),
            "state_chars": {
                "mean": round(statistics.mean(lengths), 1),
                "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
                "max": max(lengths),
            },
            "labels": labels,
        }
    return stats


def render(stats: dict[str, Any]) -> str:
    lines = [f"seed {stats['seed']}, splits {stats['splits']}"]
    for ds, s in stats["datasets"].items():
        lines += [
            "",
            f"## {ds}  ({s['source']}, {s['licence']})",
            f"loaded {s['loaded']}, duplicates dropped {s['duplicates_dropped']}, "
            f"records {s['records']}",
            f"questions per record {s['questions_per_record']}, state chars {s['state_chars']}",
        ]
        for qid, lab in s["labels"].items():
            flag = "  ** HEAVY IMBALANCE **" if lab["imbalance"]["heavy"] else ""
            lines.append(
                f"- {qid} ({lab['type']}, {lab['n_options']}): {lab['overall']}"
                f"  [min share {lab['imbalance']['min_share']:.1%}, "
                f"max/min {lab['imbalance']['max_min_ratio']}]{flag}"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    name: str
    source: Source
    load: Callable[[Path], list[dict[str, Any]]]
    test_size: int


def prepare(
    out_dir: Path,
    cache_dir: Path,
    *,
    seed: int = SEED,
    tickets_test: int = 750,
    telco_test: int = 250,
    full: bool = False,
    languages: set[str] | None = None,
) -> dict[str, Any]:
    langs = languages or {"en"}
    datasets = [
        Dataset(TICKETS, TICKETS_SOURCE, lambda p: load_tickets(p, langs), tickets_test),
        Dataset(TELCO, TELCO_SOURCE, load_telco, telco_test),
    ]
    records: list[DataRecord] = []
    extra: dict[str, Any] = {"seed": seed, "datasets": {}}
    for ds in datasets:
        loaded = ds.load(fetch(ds.source, cache_dir))
        unique, dropped = dedupe(loaded)
        records += sample_and_split(unique, None if full else ds.test_size, seed)
        extra["datasets"][ds.name] = {
            "source": f"{ds.source.repo}@{ds.source.revision[:12]}/{ds.source.filename}",
            "licence": ds.source.licence,
            "loaded": len(loaded),
            "duplicates_dropped": dropped,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(out_dir / f"{split}.jsonl", (r for r in records if r.split == split))
    stats = summarise(records, extra)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


# --------------------------------------------------------------------------------------
# Out-of-domain test set (forgetting check for LoRA)
# --------------------------------------------------------------------------------------
#
# Tasks with nothing to do with customer support. They're never used for training: score
# the zero-shot base model and each LoRA adapter on them, and a drop means the adapter has
# made the model worse at things it wasn't trained on. Written to ood_test.jsonl only, so
# train/val/test are untouched.

AG_NEWS = "ag-news"
AG_NEWS_SOURCE = Source(
    repo="sh0416/ag_news",  # the original AG's News CSVs (Zhang et al. 2015) as JSONL
    revision="70e3fa1915be9a8daebec5e840f20df9a8e18793",
    filename="test.jsonl",
    sha256="2db61b7d9f5806b146a7575dc8074d9f29732a879cf319dc3677f65a286a317e",
    licence="AG's corpus: non-commercial research use",
)
AG_TOPICS = {1: "world", 2: "sports", 3: "business", 4: "sci_tech"}
AG_QUESTIONS = {
    "topic": ChoiceQuestion(
        question="What is this news article mainly about?",
        options={
            "world": "World news and politics",
            "sports": "Sports",
            "business": "Business and the economy",
            "sci_tech": "Science and technology",
        },
    ),
    "is_business": YesNoQuestion(
        question="This article is mainly about business or the economy.",
        yes="It is about companies, markets, trade or the economy",
        no="It is about something else",
    ),
}


def ag_news_record(i: int, row: dict[str, Any]) -> dict[str, Any]:
    topic = AG_TOPICS[int(row["label"])]
    title = " ".join(row["title"].split())
    body = " ".join(row["description"].replace("\\n", " ").split())
    return dict(
        state_id=f"agnews-{i:06d}",
        dataset=AG_NEWS,
        state=f"{title}\n\n{body}",
        questions=AG_QUESTIONS,
        gold={"topic": topic, "is_business": topic == "business"},
        meta={"source": AG_NEWS_SOURCE.repo, "source_row": i},
    )


def load_ag_news(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [ag_news_record(i, json.loads(line)) for i, line in enumerate(f) if line.strip()]


BOOLQ = "boolq"


@dataclass(frozen=True)
class RowsSource:
    """A Hub dataset read through the dataset viewer's JSON rows API (the files are Parquet
    only, and this script is stdlib only). The rows can't be pinned to a revision, so their
    canonical JSONL is pinned by sha256 instead: a changed upstream fails loudly."""

    repo: str
    config: str
    split: str
    sha256: str
    licence: str

    def page_url(self, offset: int, length: int = 100) -> str:
        return (
            "https://datasets-server.huggingface.co/rows?"
            f"dataset={self.repo}&config={self.config}&split={self.split}"
            f"&offset={offset}&length={length}"
        )


BOOLQ_SOURCE = RowsSource(
    repo="google/boolq",
    config="default",
    split="validation",
    sha256="e8fb84fbf510b022e963cddf3a3aded04151afa0ea0ef1cc1bf22f260ddd2344",
    licence="CC BY-SA 3.0",
)


def _get_json(url: str, attempts: int = 8, headers: dict[str, str] | None = None) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except OSError as e:  # the viewer API sometimes drops a connection
            if attempt == attempts:
                raise
            print(f"retrying {url}: {e}", file=sys.stderr)
            time.sleep(min(60, 3 * 2 ** (attempt - 1)))  # 429s ease off within a minute
    raise AssertionError("unreachable")


def fetch_rows(source: RowsSource, cache_dir: Path) -> Path:
    """Every row of ``source`` as canonical JSONL in ``cache_dir`` (once), hash-checked."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{source.repo.replace('/', '__')}-{source.split}.jsonl"
    if not path.exists():
        print(f"downloading {source.repo} {source.split} via the dataset viewer", file=sys.stderr)
        rows: list[dict[str, Any]] = []
        while True:
            page = _get_json(source.page_url(len(rows)))
            rows += [r["row"] for r in page["rows"]]
            if not page["rows"] or len(rows) >= page["num_rows_total"]:
                break
        text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
        path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source.sha256:
        raise RuntimeError(f"{path}: sha256 {digest} != pinned {source.sha256}")
    return path


def boolq_record(i: int, row: dict[str, Any]) -> dict[str, Any]:
    q = row["question"].strip()
    return dict(
        state_id=f"boolq-{i:06d}",
        dataset=BOOLQ,
        state=" ".join(row["passage"].split()),
        # One question per passage, under a stable id; only its wording changes.
        questions={
            "answer": YesNoQuestion(
                question=f"{q[0].upper()}{q[1:]}?",
                yes="The passage says yes",
                no="The passage says no",
            )
        },
        gold={"answer": bool(row["answer"])},
        meta={"source": BOOLQ_SOURCE.repo, "source_row": i},
    )


def load_boolq(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [boolq_record(i, json.loads(line)) for i, line in enumerate(f) if line.strip()]


def prepare_ood(out_dir: Path, cache_dir: Path, *, seed: int = SEED, size: int = 300) -> dict:
    """Write ``ood_test.jsonl`` (``size`` states per dataset, stratified on gold) and
    ``ood_stats.json``."""
    loaders = [
        (AG_NEWS, lambda: load_ag_news(fetch(AG_NEWS_SOURCE, cache_dir)), AG_NEWS_SOURCE),
        (BOOLQ, lambda: load_boolq(fetch_rows(BOOLQ_SOURCE, cache_dir)), BOOLQ_SOURCE),
    ]
    records: list[DataRecord] = []
    extra: dict[str, Any] = {"seed": seed, "datasets": {}}
    for name, load, source in loaders:
        loaded = load()
        unique, dropped = dedupe(loaded)
        ordered = systematic_sample(stratified_order(unique, random.Random(seed)), size)
        records += sorted(
            (DataRecord(split="test", **r) for r in ordered), key=lambda r: r.state_id
        )
        extra["datasets"][name] = {
            "source": source.repo,
            "licence": source.licence,
            "loaded": len(loaded),
            "duplicates_dropped": dropped,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "ood_test.jsonl", records)
    stats = summarise(records, extra)
    (out_dir / "ood_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


# --------------------------------------------------------------------------------------
# CFPB Consumer Complaint Database: the held-out domain (Decision 16)
# --------------------------------------------------------------------------------------
#
# Real consumer complaint narratives, CC0. No system was trained on this domain, so it's the
# fair "beats Jev?" test. Written to cfpb_{train,val,test}.jsonl, so the main splits are
# untouched.
#
# Source: Decision 16 said "the CFPB search API with filters". That no longer works: the CFPB
# stopped publishing narratives on 14 Aug 2026 and removed them from the database and its API
# on 14 Sep 2026 (the API has no narrative field, and `has_narrative` is ignored). They were
# moved to the FOIA reading room as per-month CSV exports, by date received. We use the June
# and July 2026 files (after every model's training cutoff); the August file stops at 14 Aug
# and holds a single narrative. Same official data and fields, pinned by sha256.

CFPB = "cfpb-complaints"
CFPB_ARCHIVE_PAGE = (
    "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)


@dataclass(frozen=True)
class UrlSource:
    """A plain file at a URL, pinned by sha256 (``fetch`` reads ``url``, ``revision`` and
    ``filename``)."""

    url: str
    filename: str
    sha256: str
    licence: str
    revision: str = "cfpb-foia-"


CFPB_SOURCES = [
    UrlSource(
        url="https://files.consumerfinance.gov/f/documents/CCDB_Export_19_June_2026.zip",
        filename="CCDB_Export_19_June_2026.zip",
        sha256="3021c8af9eeed4971a92f15be21c7ce08cf3c1337151de255e5b12bf782049df",
        licence="CC0 1.0 (US government work)",
    ),
    UrlSource(
        url="https://files.consumerfinance.gov/f/documents/CCDB_Export_20_July_2026.zip",
        filename="CCDB_Export_20_July_2026.zip",
        sha256="aafebbaaf0962c3d29532411ac2133c01bae830b7ea6ed1931a6e42278437d18",
        licence="CC0 1.0 (US government work)",
    ),
]

# Inclusive window on "Date received" (ISO dates compare as strings).
CFPB_WINDOW = ("2026-06-01", "2026-07-31")
CFPB_SPLIT_SIZES = {"train": 3000, "val": 300, "test": 1000}
CFPB_MIN_CHARS, CFPB_MAX_CHARS = 100, 4000
CFPB_CREDIT_REPORTING_CAP = 0.25  # max share of the sample
# Company responses that aren't a final outcome: no answer for monetary_relief.
CFPB_UNRESOLVED = {"In progress", "Untimely response", ""}

# Post-2017 product names (the CFPB's current taxonomy) -> (option key, description).
PRODUCTS = {
    "Credit reporting or other personal consumer reports": (
        "credit_reporting",
        "Credit reporting or other personal consumer reports",
    ),
    "Debt collection": ("debt_collection", "Debt collection"),
    "Credit card": ("credit_card", "Credit card"),
    "Checking or savings account": ("checking_savings", "Checking or savings account"),
    "Money transfer, virtual currency, or money service": (
        "money_transfer",
        "Money transfer, virtual currency, or money service",
    ),
    "Mortgage": ("mortgage", "Mortgage"),
    "Vehicle loan or lease": ("vehicle_loan", "Vehicle loan or lease"),
    "Payday loan, title loan, personal loan, or advance loan": (
        "personal_loan",
        "Payday loan, title loan, personal loan, or advance loan",
    ),
    "Student loan": ("student_loan", "Student loan"),
    "Prepaid card": ("prepaid_card", "Prepaid card"),
    "Debt or credit management": ("debt_management", "Debt or credit management"),
}
# Older product names, consolidated into the current ones. Names that don't map cleanly
# ("Credit card or prepaid card", "Consumer Loan", ...) are dropped and counted.
PRODUCT_ALIASES = {
    "Credit reporting, credit repair services, or other personal consumer reports": (
        "Credit reporting or other personal consumer reports"
    ),
    "Credit reporting": "Credit reporting or other personal consumer reports",
    "Bank account or service": "Checking or savings account",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Payday loan, title loan, or personal loan": (
        "Payday loan, title loan, personal loan, or advance loan"
    ),
    "Payday loan": "Payday loan, title loan, personal loan, or advance loan",
    "Credit repair services": "Debt or credit management",
}

# The 26 issue options (the contract's maximum): the two most common issues within each of
# the 11 products in the June-July 2026 narratives, plus the next four most common overall.
# The descriptions are the CFPB's own issue names. Complaints with any other issue are
# dropped (coverage is in cfpb_stats.json).
ISSUES = {
    "Attempts to collect debt not owed": "debt_not_owed",
    "Written notification about debt": "debt_notification",
    "False statements or representation": "false_statements",
    "Took or threatened to take negative or legal action": "threatened_action",
    "Managing an account": "managing_account",
    "Problem with a lender or other company charging your account": "charged_by_other_company",
    "Closing an account": "closing_account",
    "Problem with a purchase shown on your statement": "purchase_on_statement",
    "Other features, terms, or problems": "card_other_problem",
    "Fees or interest": "fees_or_interest",
    "Incorrect information on your report": "incorrect_report_info",
    "Improper use of your report": "improper_report_use",
    "Problem with a company's investigation into an existing problem": "investigation_problem",
    "Confusing or missing disclosures": "missing_disclosures",
    "Didn't provide services promised": "services_not_provided",
    "Fraud or scam": "fraud_or_scam",
    "Unauthorized transactions or other transaction problem": "unauthorized_transaction",
    "Trouble during payment process": "payment_process_trouble",
    "Struggling to pay mortgage": "struggling_mortgage",
    "Problem when making payments": "problem_making_payments",
    "Problem with a purchase or transfer": "purchase_or_transfer_problem",
    "Trouble using the card": "trouble_using_card",
    "Dealing with your lender or servicer": "dealing_with_servicer",
    "Struggling to repay your loan": "struggling_repay_loan",
    "Managing the loan or lease": "managing_loan_lease",
    "Repossession": "repossession",
}
# The same issue under a different product's wording.
ISSUE_ALIASES = {
    "Closing your account": "Closing an account",
    "Struggling to pay your loan": "Struggling to repay your loan",
    "Trouble using your card": "Trouble using the card",
    "Charged fees or interest you didn't expect": "Fees or interest",
}

CFPB_QUESTIONS = {
    "product": ChoiceQuestion(
        question="Which financial product is this complaint about?",
        options={key: desc for key, desc in PRODUCTS.values()},
    ),
    "issue": ChoiceQuestion(
        question="What is the main issue in this complaint?",
        options={key: desc for desc, key in ISSUES.items()},
    ),
    "monetary_relief": YesNoQuestion(
        question="The company closed this complaint with monetary relief for the consumer.",
        yes="The company gave money back: a refund, credit, reversed fee or compensation",
        no="The company closed it with an explanation or non-monetary relief only",
    ),
}

# A narrative counts as English unless Spanish function words outnumber English ones (the
# database takes complaints in Spanish too). A relative test, because legal-template
# complaints ("15 U.S.C. 1681c-2 ...") have few function words of either kind.
_EN_TEXT = "the and to i my of in that was for it is on this have with they not be as but"
_ES_TEXT = "de la que el y los se del las por una para con es su al lo como pero muy esta"
_EN_WORDS, _ES_WORDS = set(_EN_TEXT.split()), set(_ES_TEXT.split())
_WORD = re.compile(r"[a-záéíóúñü']+")


def is_english(text: str) -> bool:
    words = _WORD.findall(text.lower())
    en = sum(w in _EN_WORDS for w in words)
    return en > 0 and en >= sum(w in _ES_WORDS for w in words)


def read_cfpb_zip(path: Path) -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(path) as z:
        (name,) = [n for n in z.namelist() if n.lower().endswith(".csv")]
        with z.open(name) as f:
            yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline=""))


def cfpb_record(row: dict[str, str]) -> dict[str, Any] | str:
    """A record dict, or the reason the row is skipped."""
    text = row["Consumer complaint narrative"].strip()
    if not text:
        return "no_narrative"
    if not CFPB_WINDOW[0] <= row["Date received"][:10] <= CFPB_WINDOW[1]:
        return "outside_window"
    if not CFPB_MIN_CHARS <= len(text) <= CFPB_MAX_CHARS:
        return "length"
    if not is_english(text):
        return "not_english"
    product = PRODUCT_ALIASES.get(row["Product"], row["Product"])
    if product not in PRODUCTS:
        return "product_unmapped"
    issue = ISSUE_ALIASES.get(row["Issue"], row["Issue"])
    if issue not in ISSUES:
        return "issue_not_in_options"
    response = row["Company response to consumer"]
    if response in CFPB_UNRESOLVED:
        return "response_unresolved"
    cid = int(row["Complaint ID"])
    return dict(
        state_id=f"cfpb-{cid}",
        dataset=CFPB,
        state=text,
        questions=CFPB_QUESTIONS,
        gold={
            "product": PRODUCTS[product][0],
            "issue": ISSUES[issue],
            "monetary_relief": response == "Closed with monetary relief",
        },
        meta={
            "source": "CFPB Consumer Complaint Database (FOIA narratives archive)",
            "source_row": cid,
            "complaint_id": cid,
            "date_received": row["Date received"][:10],
            "product": row["Product"],
            "sub_product": row["Sub-product"],
            "issue": row["Issue"],
            "sub_issue": row["Sub-issue"],
            "company_response": response,
        },
    )


def load_cfpb(paths: list[Path]) -> tuple[list[dict[str, Any]], Counter]:
    records, skipped = [], Counter()
    for path in paths:
        for row in read_cfpb_zip(path):
            r = cfpb_record(row)
            if isinstance(r, str):
                skipped[r] += 1
            else:
                records.append(r)
    return records, skipped


# Near-duplicates: complaints that are the same text with small edits, mostly template
# letters (FDCPA validation requests and the like). Exact dedup misses them, and a test
# state with a near-copy in train scores for memorising, not for the task (the NN gate
# found 40 in test). Each cluster of near-copies is collapsed to its earliest complaint.
CFPB_NEAR_DUP_COSINE = 0.8  # below the NN gate's 0.9, for margin
_NEAR_DUP_TOKEN = re.compile(r"[a-z][a-z']+")
_NEAR_DUP_TOP_TERMS = 12  # candidate pairs must share at least half of these


def _tfidf_vectors(texts: list[str], max_df: float = 0.5) -> list[dict[str, float]]:
    """Sublinear tf * idf, L2-normalised; the same weighting as ``data/nn_baseline.py``.
    Redactions (XXXX) and near-stopwords (in more than ``max_df`` of texts) are dropped."""
    docs = [
        Counter(t for t in _NEAR_DUP_TOKEN.findall(x.lower()) if not set(t) <= {"x"}) for x in texts
    ]
    n = len(docs)
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items() if c / n <= max_df}
    out = []
    for d in docs:
        v = {t: (1 + math.log(c)) * idf[t] for t, c in d.items() if t in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({t: x / norm for t, x in v.items()})
    return out


def near_dedupe(
    records: list[dict[str, Any]], threshold: float = CFPB_NEAR_DUP_COSINE
) -> tuple[list[dict[str, Any]], int]:
    """Collapse clusters of near-duplicate states (TF-IDF cosine >= ``threshold``) to the
    earliest record. Near-copies share most of their highest-weighted terms, so only pairs
    sharing half of each other's top terms are compared exactly (fast, stdlib only)."""
    records = sorted(records, key=lambda r: r["meta"]["source_row"])
    vecs = _tfidf_vectors([r["state"] for r in records])
    tops = [sorted(v, key=lambda t: (-v[t], t))[:_NEAR_DUP_TOP_TERMS] for v in vecs]
    index: dict[str, list[int]] = {}
    for i, top in enumerate(tops):
        for t in top:
            index.setdefault(t, []).append(i)
    parent = list(range(len(records)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, top in enumerate(tops):
        shared = Counter(j for t in top for j in index[t] if j < i)
        need = max(1, min(len(top), _NEAR_DUP_TOP_TERMS) // 2)
        for j, c in shared.items():
            if c >= need and root(i) != root(j):
                small, big = (
                    (vecs[i], vecs[j]) if len(vecs[i]) < len(vecs[j]) else (vecs[j], vecs[i])
                )
                if sum(x * big.get(t, 0.0) for t, x in small.items()) >= threshold:
                    parent[max(root(i), root(j))] = min(root(i), root(j))
    kept = [r for i, r in enumerate(records) if root(i) == i]
    return kept, len(records) - len(kept)


def cap_share(
    records: list[dict[str, Any]],
    product: str,
    cap: float,
    rng: random.Random,
    key: Callable[[dict[str, Any]], str] = lambda r: r["gold"]["product"],
) -> tuple[list[dict[str, Any]], int]:
    """Downsample the records whose ``key`` is ``product`` (stratified) so they're at most
    ``cap`` of the records."""
    inside = [r for r in records if key(r) == product]
    rest = [r for r in records if key(r) != product]
    limit = int(cap * len(rest) / (1 - cap))
    if len(inside) <= limit:
        return records, 0
    kept = systematic_sample(stratified_order(inside, rng), limit)
    out = sorted(rest + kept, key=lambda r: r["meta"]["source_row"])
    return out, len(inside) - limit


def prepare_cfpb(
    out_dir: Path,
    cache_dir: Path,
    *,
    seed: int = SEED,
    sizes: dict[str, int] = CFPB_SPLIT_SIZES,
    credit_reporting_cap: float = CFPB_CREDIT_REPORTING_CAP,
) -> dict[str, Any]:
    """Write ``cfpb_{train,val,test}.jsonl`` and ``cfpb_stats.json``."""
    loaded, skipped = load_cfpb([fetch(s, cache_dir) for s in CFPB_SOURCES])
    unique, dropped = dedupe(loaded)
    unique, near_dropped = near_dedupe(unique)
    capped, cap_dropped = cap_share(
        unique, "credit_reporting", credit_reporting_cap, random.Random(seed)
    )
    total = sum(sizes.values())
    splits = {s: n / total for s, n in sizes.items()}
    records = sample_and_split(capped, sizes["test"], seed, splits)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        write_jsonl(out_dir / f"cfpb_{split}.jsonl", (r for r in records if r.split == split))
    extra: dict[str, Any] = {
        "seed": seed,
        "splits": splits,
        "datasets": {
            CFPB: {
                "source": [s.url for s in CFPB_SOURCES],
                "source_sha256": [s.sha256 for s in CFPB_SOURCES],
                "archive_page": CFPB_ARCHIVE_PAGE,
                "licence": CFPB_SOURCES[0].licence,
                "filters": {
                    "date_received": list(CFPB_WINDOW),
                    "narrative": "non-empty",
                    "chars": [CFPB_MIN_CHARS, CFPB_MAX_CHARS],
                    "language": "English (more English than Spanish function words)",
                    "company_response_excluded": sorted(r for r in CFPB_UNRESOLVED if r),
                    "issue": "one of the 26 options (after ISSUE_ALIASES)",
                    "credit_reporting_cap": credit_reporting_cap,
                },
                "rows_skipped": dict(sorted(skipped.items())),
                "loaded": len(loaded),
                "duplicates_dropped": dropped,
                "near_duplicates_dropped": near_dropped,
                "near_duplicate_cosine": CFPB_NEAR_DUP_COSINE,
                "credit_reporting_capped_out": cap_dropped,
                "eligible_after_dedupe": len(capped),
            }
        },
    }
    stats = summarise(records, extra)
    (out_dir / "cfpb_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


# --------------------------------------------------------------------------------------
# TeleLogs: 5G root-cause analysis (Decision 17)
# --------------------------------------------------------------------------------------
#
# GSMA Open-Telco's ``telelogs`` benchmark: a drive-test log (KPIs per second) plus the
# engineering parameters of the nearby cells, and the root cause of a throughput drop. Every
# case lists the same 8 causes (C1-C8; Decision 17 said 16, the data has 8). We strip the
# instructions and the lettered list, so the options come only from our prompt.
#
# The 864 cases are 108 drive-test segments x 8 variants: each segment is edited once per
# cause, so siblings share almost every number and differ in the planted cause. A per-case
# split would put the 7 siblings of every test case in train, so we split by segment
# instead. Each segment holds every cause once, so every split is exactly balanced.

TELELOGS = "telelogs"
TELELOGS_SOURCE = RowsSource(
    repo="GSMA/ot-full",
    config="telelogs",
    split="test",
    sha256="d3c558f990ed2ca1750ef8cd7fd37ed4184a6744d4b54177e8d421fa6e5b0931",
    licence="MIT",
)
# Segments per split: 75 / 12 / 21 = 600 / 96 / 168 cases (Decision 17: about 600/100/164).
TELELOGS_SPLIT_SEGMENTS = {"train": 75, "val": 12, "test": 21}

# Source label -> (our key, the source's wording).
ROOT_CAUSES = {
    "C1": (
        "downtilt_too_large",
        "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    ),
    "C2": (
        "overshooting",
        "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    ),
    "C3": (
        "better_neighbour",
        "A neighboring cell provides higher throughput.",
    ),
    "C4": (
        "overlapping_coverage",
        "Non-colocated co-frequency neighboring cells cause severe overlapping coverage.",
    ),
    "C5": (
        "frequent_handovers",
        "Frequent handovers degrade performance.",
    ),
    "C6": (
        "pci_mod30_clash",
        "Neighbor cell and serving cell have the same PCI mod 30, leading to interference.",
    ),
    "C7": (
        "speed_too_high",
        "Test vehicle speed exceeds 40km/h, impacting user throughput.",
    ),
    "C8": (
        "too_few_rbs",
        "Average scheduled RBs are below 160, affecting throughput.",
    ),
}

TELELOGS_QUESTIONS = {
    "root_cause": ChoiceQuestion(
        question=(
            "In this 5G drive test, throughput drops below 600 Mbps on some road sections. "
            "What is the most likely root cause?"
        ),
        options={key: desc for key, desc in ROOT_CAUSES.values()},
    ),
}

_TELELOGS_OPTION = re.compile(r"^C\d+: .*$", re.M)
_TELELOGS_TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", re.M)


def telelogs_state(question: str) -> str:
    """The case data only: everything from "Given:" on (the cell-parameter rules, the
    drive-test table and the engineering parameters). Drops the task wording, the "enclose
    its number in \\boxed{}" instruction and the lettered C1-C8 list."""
    start = question.find("Given:")
    if start < 0:
        raise ValueError("TeleLogs case without a 'Given:' section")
    state = question[start:].strip()
    if _TELELOGS_OPTION.search(state):
        raise ValueError("TeleLogs case lists options after 'Given:'")
    return state


def telelogs_record(i: int, row: dict[str, Any]) -> dict[str, Any]:
    key, _ = ROOT_CAUSES[row["answer"].strip()]
    state = telelogs_state(row["question"])
    m = _TELELOGS_TIMESTAMP.search(state)
    if not m:
        raise ValueError(f"TeleLogs row {i}: no drive-test timestamp")
    return dict(
        state_id=f"telelogs-{i:04d}",
        dataset=TELELOGS,
        state=state,
        questions=TELELOGS_QUESTIONS,
        gold={"root_cause": key},
        meta={
            "source": TELELOGS_SOURCE.repo,
            "source_row": i,
            "source_answer": row["answer"].strip(),
            "segment": m.group(1),  # the first drive-test timestamp names the segment
        },
    )


def load_telelogs(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [telelogs_record(i, r) for i, r in enumerate(rows)]


def grouped_split(
    records: list[dict[str, Any]], group_sizes: dict[str, int], seed: int
) -> list[DataRecord]:
    """Assign whole groups (``meta["segment"]``) to splits: shuffle the groups with ``seed``
    and deal ``group_sizes[split]`` of them to each split, interleaved."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r["meta"]["segment"], []).append(r)
    if sum(group_sizes.values()) != len(groups):
        raise ValueError(f"{len(groups)} groups, but the split sizes add up to {group_sizes}")
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    total = len(names)
    ratios = {s: n / total for s, n in group_sizes.items()}
    out = [
        DataRecord(split=split, **r)
        for name, split in zip(names, interleave(ratios, total), strict=True)
        for r in groups[name]
    ]
    out.sort(key=lambda r: r.state_id)
    return out


def prepare_telelogs(
    out_dir: Path,
    cache_dir: Path,
    *,
    seed: int = SEED,
    segments: dict[str, int] = TELELOGS_SPLIT_SEGMENTS,
) -> dict[str, Any]:
    """Write ``telelogs_{train,val,test}.jsonl`` and ``telelogs_stats.json``."""
    loaded = load_telelogs(fetch_rows(TELELOGS_SOURCE, cache_dir))
    unique, dropped = dedupe(loaded)
    records = grouped_split(unique, segments, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in segments:
        write_jsonl(out_dir / f"telelogs_{split}.jsonl", (r for r in records if r.split == split))
    n = len(unique)
    extra: dict[str, Any] = {
        "seed": seed,
        "splits": {s: round(sum(r.split == s for r in records) / n, 4) for s in segments},
        "datasets": {
            TELELOGS: {
                "source": f"{TELELOGS_SOURCE.repo} ({TELELOGS_SOURCE.config}/"
                f"{TELELOGS_SOURCE.split}, rows sha256 {TELELOGS_SOURCE.sha256[:12]})",
                "licence": TELELOGS_SOURCE.licence,
                "loaded": len(loaded),
                "duplicates_dropped": dropped,
                "segments": len({r.meta["segment"] for r in records}),
                "segments_per_split": segments,
                "split_by": "drive-test segment (8 sibling cases each), not by case",
            }
        },
    }
    stats = summarise(records, extra)
    (out_dir / "telelogs_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


# --------------------------------------------------------------------------------------
# CVE advisories: "how bad is this, and from where?" (Decision 17)
# --------------------------------------------------------------------------------------
#
# Red Hat's security advisories for CVEs made public Apr-Sep 2026 (after every model's
# stated training cutoff; Gemini 3.5 Flash-Lite's is March 2026). The state is the advisory
# text with every CVSS score, vector and severity rating removed; the gold is the CVSS 3.1
# vector Red Hat publishes for the CVE, so the questions are "work out the CVSS from the
# facts", never a lookup.
#
# Decided after a blind hand-check of 30 states per question: `severity` (23/30
# exact, 30/30 within one level) and `attack_vector` (28/30) are answerable from the text.
# `needs_non_default_config` (6/30 say what the default is), `affected` (VEX justifications
# describe products, not the config in the text; 18/30 unambiguous) and `exploited_in_wild`
# (7 KEV positives in the pool) were dropped. Cisco CSAF was dropped too: its terms allow
# internal use only, and data.yml artifacts of this public repo are publicly downloadable.

CVE = "cve-advisories"
CVE_WINDOW = ("2026-04-01", "2026-10-01")  # Red Hat public_date, [start, end)
CVE_SPLIT_SIZES = {"train": 3000, "val": 300, "test": 1000}
CVE_KERNEL_CAP = 0.25  # max share of Linux kernel CVEs (raw commit messages)
CVE_CANDIDATES = 6500  # pre-sampled from the listing before fetching each CVE's text
CVE_MIN_CHARS, CVE_MAX_CHARS = 200, 6000
REDHAT_API = "https://access.redhat.com/hydra/rest/securitydata"
REDHAT_HEADERS = {"User-Agent": "snapcall-data/1.0"}
REDHAT_LICENCE = "CC BY 4.0 (Red Hat, Inc.)"

SEVERITIES = ["low", "medium", "high", "critical"]
ATTACK_VECTORS = {
    "N": ("network", "Network: the attacker can be anywhere on the network or internet"),
    "A": ("adjacent", "Adjacent: the attacker must be on the same local network or link"),
    "L": ("local", "Local: the attacker needs local access, or a user must open a file"),
    "P": ("physical", "Physical: the attacker must physically touch the device"),
}

CVE_QUESTIONS = {
    "severity": ScoreQuestion(
        question="How severe is this vulnerability (its CVSS base severity)?",
        levels=SEVERITIES,
    ),
    "attack_vector": ChoiceQuestion(
        question="From where can an attacker exploit this vulnerability?",
        options={key: desc for key, desc in ATTACK_VECTORS.values()},
    ),
}

# Anything that gives the CVSS away: vectors and scores, and severity ratings in words
# (Red Hat's "rated as having Important impact", Chromium's "security severity: High").
# Whole sentences are dropped, so no half-sentence hints remain.
_CVSS_VECTOR = re.compile(r"\(?CVSS:?\s*v?\d\.\d\s*/?[A-Za-z:/]*\)?")
_CVSS_METRIC = re.compile(r"\b(?:AV|AC|PR|UI|AT|VC|VI|VA|SC|SI|SA|C|I|A|S):[A-Z]\b")
_RATING = re.compile(
    r"CVSS|base score|security impact rating|\bseverity\b"
    r"|\b(?:rated|evaluated|classified|assessed|considered|categori[sz]ed)\b[^.]{0,80}"
    r"\b(?:low|moderate|important|critical|high|medium)\b"
    r"|\b(?:low|moderate|important|critical|high|medium)[- ](?:security )?"
    r"(?:impact|rating|risk)\b",
    re.I,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=\))\s+(?=[A-Z])")
_CVE_SKIP_STATEMENT = re.compile(
    r"duplicate (?:assignment|of)|not a (?:security )?(?:flaw|vulnerability)|"
    r"CVE (?:has been|was) rejected",
    re.I,
)
_MITIGATION_BOILERPLATE = re.compile(r"either not available or the currently available", re.I)


def redact_cvss(text: str) -> str:
    """``text`` without any sentence that mentions a CVSS vector, metric, score or a
    severity rating."""
    text = _CVSS_VECTOR.sub(" CVSSVECTOR ", text or "")
    paragraphs = []
    for para in text.split("\n"):
        kept = [
            s
            for s in _SENTENCE_END.split(para)
            if not (_RATING.search(s) or "CVSSVECTOR" in s or _CVSS_METRIC.search(s))
        ]
        paragraphs.append(" ".join(kept).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(paragraphs)).strip()


def severity_band(score: float) -> str | None:
    """CVSS 3.1 qualitative rating (0 is "none", which has no question level)."""
    if score <= 0:
        return None
    return SEVERITIES[(score >= 4.0) + (score >= 7.0) + (score >= 9.0)]


def _rh_gold(cvss3: dict[str, Any]) -> tuple[str, str] | None:
    vector = cvss3.get("cvss3_scoring_vector") or ""
    m = re.search(r"AV:([NALP])", vector)
    try:
        # per-CVE records say cvss3_base_score, listing rows cvss3_score
        score = cvss3.get("cvss3_base_score") or cvss3.get("cvss3_score") or 0
        band = severity_band(float(score))
    except ValueError:
        band = None
    if not (m and band):
        return None
    return band, ATTACK_VECTORS[m.group(1)][0]


def is_kernel(title: str) -> bool:
    return title.lower().startswith("kernel")


def cve_state(cve: dict[str, Any]) -> str:
    """Title, descriptions, Red Hat's statement and a real mitigation, with CVSS removed."""
    parts = [f"Title: {cve['bugzilla']['description'].strip()}"]
    for detail in cve.get("details") or []:  # often the upstream text, then Red Hat's copy
        detail = redact_cvss(detail)
        if detail and _norm(detail) not in {_norm(p) for p in parts}:
            parts.append(detail)
    if statement := redact_cvss(cve.get("statement") or ""):
        parts.append(f"Statement: {statement}")
    mitigation = (cve.get("mitigation") or {}).get("value") or ""
    if not _MITIGATION_BOILERPLATE.search(mitigation) and (mitigation := redact_cvss(mitigation)):
        parts.append(f"Mitigation: {mitigation}")
    return "\n\n".join(p for p in parts if p)


def cve_record(row: int, cve: dict[str, Any]) -> dict[str, Any] | str:
    """A record, or the reason the CVE is skipped."""
    gold = _rh_gold(cve.get("cvss3") or {})
    if gold is None:
        return "no CVSS 3 score"
    if not (cve.get("bugzilla") or {}).get("description"):
        return "no title"
    if _CVE_SKIP_STATEMENT.search(cve.get("statement") or ""):
        return "duplicate or not a flaw"
    state = cve_state(cve)
    if not CVE_MIN_CHARS <= len(state) <= CVE_MAX_CHARS:
        return "length"
    severity, vector = gold
    return dict(
        state_id=f"cve-{cve['name']}",
        dataset=CVE,
        state=state,
        questions=CVE_QUESTIONS,
        gold={"severity": SEVERITIES.index(severity), "attack_vector": vector},
        meta={
            "source": "Red Hat security data",
            "source_url": f"https://access.redhat.com/security/cve/{cve['name']}",
            "source_row": row,
            "cve": cve["name"],
            "public_date": cve.get("public_date"),
            "kernel": is_kernel(cve["bugzilla"]["description"]),
            "redhat_rating": cve.get("threat_severity"),
        },
    )


def redhat_listing(window: tuple[str, str] = CVE_WINDOW) -> list[dict[str, Any]]:
    """Every Red Hat CVE made public in ``window`` (summary rows, no text)."""
    rows: list[dict[str, Any]] = []
    for page in range(1, 100):
        url = (
            f"{REDHAT_API}/cve.json?after={window[0]}&before={window[1]}&per_page=1000&page={page}"
        )
        batch = _get_json(url, headers=REDHAT_HEADERS)
        if not batch:
            break
        rows += batch
    rows = [r for r in rows if window[0] <= r["public_date"][:10] < window[1]]
    return sorted(rows, key=lambda r: (r["public_date"], r["CVE"]))


def cve_candidates(
    listing: list[dict[str, Any]], n: int, kernel_cap: float, seed: int
) -> list[dict[str, Any]]:
    """Pre-sample ``n`` listing rows to fetch, stratified on the listing's own CVSS and with
    kernel CVEs at most ``kernel_cap`` of them, so the fetch fits in one CI job."""
    rows = []
    for i, r in enumerate(listing):
        if gold := _rh_gold(r):
            rows.append(
                {
                    "row": r,
                    "questions": list(CVE_QUESTIONS),
                    "gold": dict(zip(CVE_QUESTIONS, gold, strict=True)),
                    "meta": {"source_row": i},
                    "kernel": is_kernel(r.get("bugzilla_description") or ""),
                }
            )
    rng = random.Random(seed)
    kernel = [r for r in rows if r["kernel"]]
    other = [r for r in rows if not r["kernel"]]
    n_kernel = min(len(kernel), int(n * kernel_cap))
    picked = systematic_sample(stratified_order(kernel, rng), n_kernel)
    picked += systematic_sample(stratified_order(other, rng), n - n_kernel)
    picked.sort(key=lambda r: r["meta"]["source_row"])
    return [r["row"] for r in picked]


def fetch_redhat_cves(names: list[str], workers: int = 4) -> list[dict[str, Any]]:
    """Each CVE's full record from Red Hat's API, in ``names`` order."""
    import concurrent.futures

    def one(name: str) -> dict[str, Any]:
        return _get_json(f"{REDHAT_API}/cve/{name}.json", headers=REDHAT_HEADERS)

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, names))


def cve_snapshot(cache_dir: Path, seed: int, kernel_cap: float) -> Path:
    """Fetch the listing and the pre-sampled CVEs once into ``cve_raw.jsonl.gz`` (one CVE
    record per line, in listing order). Red Hat updates records, so the snapshot (not the
    live API) is what a rebuild should read; its sha256 goes into the stats."""
    import gzip

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "cve_raw.jsonl.gz"
    if not path.exists():
        listing = redhat_listing()
        names = [r["CVE"] for r in cve_candidates(listing, CVE_CANDIDATES, kernel_cap, seed)]
        print(f"fetching {len(names)} of {len(listing)} Red Hat CVEs", file=sys.stderr)
        records = fetch_redhat_cves(names)
        tmp = path.with_suffix(".part")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        tmp.rename(path)
    return path


def load_cves(path: Path) -> tuple[list[dict[str, Any]], Counter]:
    import gzip

    records, skipped = [], Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row, line in enumerate(f):
            r = cve_record(row, json.loads(line))
            if isinstance(r, str):
                skipped[r] += 1
            else:
                records.append(r)
    return records, skipped


def prepare_cve(
    out_dir: Path,
    cache_dir: Path,
    *,
    seed: int = SEED,
    sizes: dict[str, int] = CVE_SPLIT_SIZES,
    kernel_cap: float = CVE_KERNEL_CAP,
    snapshot: Path | None = None,
) -> dict[str, Any]:
    """Write ``cve_{train,val,test}.jsonl``, ``cve_stats.json`` and the raw snapshot."""
    import shutil

    raw = snapshot or cve_snapshot(cache_dir, seed, kernel_cap)
    loaded, skipped = load_cves(raw)
    unique, dropped = dedupe(loaded)
    unique, near_dropped = near_dedupe(unique)
    capped, cap_dropped = cap_share(
        unique, "kernel", kernel_cap, random.Random(seed),
        key=lambda r: "kernel" if r["meta"]["kernel"] else "other",
    )  # fmt: skip
    total = sum(sizes.values())
    splits = {s: n / total for s, n in sizes.items()}
    records = sample_and_split(capped, sizes["test"], seed, splits)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        write_jsonl(out_dir / f"cve_{split}.jsonl", (r for r in records if r.split == split))
    if raw.resolve() != (out_dir / raw.name).resolve():
        shutil.copyfile(raw, out_dir / "cve_raw.jsonl.gz")
    extra: dict[str, Any] = {
        "seed": seed,
        "splits": splits,
        "datasets": {
            CVE: {
                "source": f"{REDHAT_API}/cve/<CVE>.json",
                "source_snapshot_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "licence": REDHAT_LICENCE,
                "filters": {
                    "public_date": list(CVE_WINDOW),
                    "candidates_fetched": CVE_CANDIDATES,
                    "chars": [CVE_MIN_CHARS, CVE_MAX_CHARS],
                    "kernel_cap": kernel_cap,
                    "redacted": "sentences with CVSS vectors, metrics, scores or ratings",
                },
                "rows_skipped": dict(sorted(skipped.items())),
                "loaded": len(loaded),
                "duplicates_dropped": dropped,
                "near_duplicates_dropped": near_dropped,
                "near_duplicate_cosine": CFPB_NEAR_DUP_COSINE,
                "kernel_capped_out": cap_dropped,
                "eligible_after_dedupe": len(capped),
                "kernel_share": round(sum(r.meta["kernel"] for r in records) / len(records), 4),
            }
        },
    }
    stats = summarise(records, extra)
    (out_dir / "cve_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path("outputs/data"))
    p.add_argument("--cache", type=Path, default=Path("data/raw"), help="download cache")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tickets-test", type=int, default=750, help="test states from tickets")
    p.add_argument("--telco-test", type=int, default=250, help="test states from telco")
    p.add_argument("--full", action="store_true", help="keep every record (no subsampling)")
    p.add_argument("--languages", default="en", help="ticket languages, comma-separated")
    p.add_argument(
        "--ood-only",
        action="store_true",
        help="write only the out-of-domain forgetting-check set (ood_test.jsonl)",
    )
    p.add_argument("--ood-size", type=int, default=300, help="OOD states per dataset")
    p.add_argument(
        "--cfpb",
        action="store_true",
        help="write only the CFPB held-out domain (cfpb_{train,val,test}.jsonl)",
    )
    p.add_argument(
        "--cfpb-credit-cap",
        type=float,
        default=CFPB_CREDIT_REPORTING_CAP,
        help="max share of credit reporting in the CFPB sample",
    )
    p.add_argument(
        "--telelogs",
        action="store_true",
        help="write only the TeleLogs 5G root-cause set (telelogs_{train,val,test}.jsonl)",
    )
    p.add_argument(
        "--cve",
        action="store_true",
        help="write only the CVE advisories set (cve_{train,val,test}.jsonl)",
    )
    p.add_argument(
        "--cve-snapshot",
        type=Path,
        help="rebuild the CVE set from this cve_raw.jsonl.gz instead of fetching",
    )
    a = p.parse_args(argv)
    if a.cve:
        print(render(prepare_cve(a.out, a.cache, seed=a.seed, snapshot=a.cve_snapshot)))
        return
    if a.telelogs:
        print(render(prepare_telelogs(a.out, a.cache, seed=a.seed)))
        return
    if a.cfpb:
        print(
            render(
                prepare_cfpb(a.out, a.cache, seed=a.seed, credit_reporting_cap=a.cfpb_credit_cap)
            )
        )
        return
    if a.ood_only:
        print(render(prepare_ood(a.out, a.cache, seed=a.seed, size=a.ood_size)))
        return
    stats = prepare(
        a.out,
        a.cache,
        seed=a.seed,
        tickets_test=a.tickets_test,
        telco_test=a.telco_test,
        full=a.full,
        languages=set(a.languages.split(",")),
    )
    print(render(stats))


if __name__ == "__main__":
    main()

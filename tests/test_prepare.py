"""data/prepare.py on small synthetic CSVs (no downloads)."""

import csv
import json
import random
from collections import Counter

import pytest

from data import prepare as P
from snapcall.schema import DataRecord, read_jsonl

TICKET_COLS = ["subject", "body", "answer", "type", "queue", "priority", "language", "version"]
TICKET_COLS += [f"tag_{i}" for i in range(1, 9)]


def _write_csv(path, cols, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def _tickets_csv(path, n=600):
    rng = random.Random(0)
    queues, types = list(P.QUEUES), list(P.TYPES)
    rows = []
    for i in range(n):
        q = queues[min(int(rng.expovariate(0.5)), len(queues) - 1)]  # skewed, like the real data
        rows.append(
            dict(
                subject=f"Ticket {i}" if i % 7 else "",
                body=f"Body of ticket {i}.\\nSecond line.",
                answer="Thanks",
                type=rng.choice(types),
                queue=q,
                priority=rng.choice(P.PRIORITIES),
                language="de" if i % 5 == 0 else "en",
                version="400",
                tag_1="Refund" if i % 11 == 0 else "IT",
                **{f"tag_{k}": "" for k in range(2, 9)},
            )
        )
    rows.append(dict(rows[1], queue="Billing and Payments"))  # same text, conflicting gold
    return _write_csv(path, TICKET_COLS, rows)


def _telco_csv(path, per_intent=12):
    cat = {i: "SUBSCRIPTION" if i in P.LEAVING_INTENTS else "SERVICES" for i in P.INTENTS}
    rows = [
        dict(
            instruction=f"I want to {intent} number {j}",
            intent=intent,
            category=cat[intent],
            tags="BL",
            response="ok",
        )
        for intent in P.INTENTS
        for j in range(per_intent)
    ]
    rows.append(dict(rows[0], instruction=rows[0]["instruction"].upper()))  # duplicate
    return _write_csv(path, ["instruction", "intent", "category", "tags", "response"], rows)


@pytest.fixture
def fake_sources(tmp_path, monkeypatch):
    files = {
        P.TICKETS_SOURCE.repo: _tickets_csv(tmp_path / "tickets.csv"),
        P.TELCO_SOURCE.repo: _telco_csv(tmp_path / "telco.csv"),
    }
    monkeypatch.setattr(P, "fetch", lambda source, cache_dir: files[source.repo])
    return tmp_path


def test_interleave_hits_ratios_on_every_prefix():
    seq = P.interleave(P.SPLITS, 1000)
    assert Counter(seq) == {"train": 700, "val": 150, "test": 150}
    for n in (20, 40, 100):
        c = Counter(seq[:n])
        assert all(abs(c[s] - P.SPLITS[s] * n) <= 1 for s in P.SPLITS)


def test_systematic_sample_is_proportional():
    ordered = ["a"] * 800 + ["b"] * 200
    assert Counter(P.systematic_sample(ordered, 100)) == {"a": 80, "b": 20}
    assert P.systematic_sample(ordered, 5000) == ordered


def test_dedupe_drops_conflicting_texts_entirely():
    r = dict(state="Hello  World", gold={"q": "a"}, meta={"source_row": 0})
    same = dict(r, state="hello world", meta={"source_row": 1})
    conflict = dict(state="x", gold={"q": "a"}, meta={"source_row": 2})
    conflict2 = dict(state="X", gold={"q": "b"}, meta={"source_row": 3})
    out, dropped = P.dedupe([r, same, conflict, conflict2])
    assert out == [r] and dropped == 3


def test_ticket_mapping():
    row = dict(
        subject="Charged twice", body="Line one\\nLine two", type="Request",
        queue="Customer Service", priority="high", language="en", version="400",
        tag_1="Refund", **{f"tag_{k}": "" for k in range(2, 9)},
    )  # fmt: skip
    rec = DataRecord(split="test", **P.ticket_record(5, row))
    assert rec.state == "Subject: Charged twice\n\nLine one\nLine two"
    assert rec.gold == {"queue": "customer_service", "type": "request", "priority": 2,
                        "is_billing": True}  # fmt: skip
    assert rec.meta["tags"] == ["Refund"]


def test_prepare_end_to_end(fake_sources):
    out = fake_sources / "out"
    stats = P.prepare(out, fake_sources / "cache", tickets_test=30, telco_test=30)

    records = {s: list(read_jsonl(out / f"{s}.jsonl", DataRecord)) for s in P.SPLITS}
    assert all(r.split == s for s, rs in records.items() for r in rs)
    ids = [r.state_id for rs in records.values() for r in rs]
    assert len(ids) == len(set(ids))
    texts = [(r.dataset, r.state.lower()) for rs in records.values() for r in rs]
    assert len(texts) == len(set(texts)), "a state appears in more than one split"

    for ds, n_q in ((P.TICKETS, 4), (P.TELCO, 3)):
        counts = {s: sum(r.dataset == ds for r in records[s]) for s in P.SPLITS}
        assert counts["test"] == 30
        assert counts == stats["datasets"][ds]["records"]
        assert stats["datasets"][ds]["questions_per_record"] == {str(n_q): sum(counts.values())}

    tickets = [r for rs in records.values() for r in rs if r.dataset == P.TICKETS]
    assert all(r.meta["language"] == "en" for r in tickets)
    assert stats["datasets"][P.TICKETS]["duplicates_dropped"] == 2  # both conflicting rows
    assert stats["datasets"][P.TELCO]["duplicates_dropped"] == 1

    # Telco intents are balanced, so every intent should reach the test split.
    test_intents = {r.gold["intent"] for r in records["test"] if r.dataset == P.TELCO}
    assert len(test_intents) >= 25
    assert json.loads((out / "stats.json").read_text()) == stats


def test_prepare_is_deterministic(fake_sources):
    a = P.prepare(fake_sources / "a", fake_sources, tickets_test=30, telco_test=30)
    b = P.prepare(fake_sources / "b", fake_sources, tickets_test=30, telco_test=30)
    assert a == b
    for s in P.SPLITS:
        assert (fake_sources / "a" / f"{s}.jsonl").read_bytes() == (
            fake_sources / "b" / f"{s}.jsonl"
        ).read_bytes()
    c = P.prepare(fake_sources / "c", fake_sources, seed=1, tickets_test=30, telco_test=30)
    assert (fake_sources / "a" / "test.jsonl").read_bytes() != (
        fake_sources / "c" / "test.jsonl"
    ).read_bytes()
    assert c["datasets"][P.TICKETS]["records"] == a["datasets"][P.TICKETS]["records"]


def test_full_keeps_everything(fake_sources):
    stats = P.prepare(fake_sources / "out", fake_sources, full=True)
    n = sum(stats["datasets"][P.TELCO]["records"].values())
    assert n == len(P.INTENTS) * 12


def test_imbalance_flag():
    assert P.imbalance({"a": 99, "b": 1})["heavy"]
    assert not P.imbalance({"a": 60, "b": 40})["heavy"]


# --- out-of-domain forgetting-check set ------------------------------------------------


def test_ag_news_record():
    r = P.ag_news_record(
        7, {"label": 3, "title": "Stocks  rally", "description": "Shares rose.\\nMore."}
    )
    rec = DataRecord(split="test", **r)
    assert rec.state == "Stocks rally\n\nShares rose. More."
    assert rec.gold == {"topic": "business", "is_business": True}
    assert set(rec.questions) == {"topic", "is_business"}


def test_boolq_record():
    row = {"question": "is the sky blue", "passage": "The sky  is blue.", "answer": True}
    rec = DataRecord(split="test", **P.boolq_record(0, row))
    assert rec.questions["answer"].question == "Is the sky blue?"
    assert rec.gold == {"answer": True} and rec.state == "The sky is blue."


def test_fetch_rows_pages_and_checks_hash(tmp_path, monkeypatch):
    rows = [{"question": f"q{i}", "passage": "p", "answer": i % 2 == 0} for i in range(5)]

    def fake_get(url):
        offset = int(url.split("offset=")[1].split("&")[0])
        return {"rows": [{"row": r} for r in rows[offset : offset + 2]], "num_rows_total": 5}

    monkeypatch.setattr(P, "_get_json", fake_get)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    import hashlib

    good = P.RowsSource(
        "x/y", "default", "validation", hashlib.sha256(text.encode()).hexdigest(), ""
    )
    path = P.fetch_rows(good, tmp_path)
    assert path.read_text() == text
    bad = P.RowsSource("x/z", "default", "validation", "0" * 64, "")
    with pytest.raises(RuntimeError, match="sha256"):
        P.fetch_rows(bad, tmp_path)


# --- CFPB held-out domain (Decision 16) --------------------------------------------------

CFPB_COLS = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company", "State", "ZIP code",
    "Tags", "Submitted via", "Date sent to company", "Company response to consumer",
    "Timely response?", "Complaint ID",
]  # fmt: skip

_ENGLISH = (
    "I called the bank and they said that my account was closed but I did not ask for it, "
    "and now I cannot get my money back from them"
)


def _cfpb_zip(path, rows):
    import io
    import zipfile

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CFPB_COLS)
    w.writeheader()
    w.writerows({c: r.get(c, "") for c in CFPB_COLS} for r in rows)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(path.stem + ".csv", buf.getvalue())
    return path


_VOCAB = [f"{a}{b}" for a in ("lo", "ma", "ri", "su", "te", "vo") for b in "bcdfgkmnpt"]


def _words(rng, k=25):
    """Random filler, so fixture complaints aren't near-duplicates of each other."""
    return " ".join(rng.choice(_VOCAB) + rng.choice(_VOCAB) for _ in range(k))


def _cfpb_rows(n=1500, start=0):
    rng = random.Random(start)
    products = list(P.PRODUCTS)
    issues = list(P.ISSUES)
    rows = []
    for i in range(n):
        # Credit reporting is ~60% of complaints, as in the full database.
        product = products[0] if rng.random() < 0.6 else rng.choice(products[1:])
        relief = rng.random() < 0.1
        rows.append(
            dict.fromkeys(CFPB_COLS, "")
            | {
                "Date received": f"2026-0{6 + i % 2}-{1 + i % 28:02d}",
                "Product": product,
                "Issue": issues[i % len(issues)],
                "Consumer complaint narrative": f"{_ENGLISH} {_words(rng)} ({start + i}).",
                "Company response to consumer": "Closed with monetary relief"
                if relief
                else "Closed with explanation",
                "Complaint ID": str(start + i),
            }
        )
    return rows


def _cfpb_edge_rows():
    ok = dict(_cfpb_rows(1, start=900000)[0])
    return [
        dict(ok, **{"Complaint ID": "900001", "Consumer complaint narrative": ""}),
        dict(ok, **{"Complaint ID": "900002", "Date received": "2026-05-31"}),
        dict(ok, **{"Complaint ID": "900003", "Consumer complaint narrative": "Too short."}),
        dict(ok, **{"Complaint ID": "900004", "Consumer complaint narrative": "El banco "
                    "cerró mi cuenta y no me dieron una explicación de la razón por la que lo "
                    "hicieron, quiero que me devuelvan el dinero que tenía en la cuenta."}),
        dict(ok, **{"Complaint ID": "900005", "Company response to consumer": "In progress"}),
        dict(ok, **{"Complaint ID": "900006", "Issue": "Some rare issue"}),
        dict(ok, **{"Complaint ID": "900007", "Product": "Consumer Loan"}),
        # An alias (old product name, synonym issue) that maps cleanly.
        dict(ok, **{"Complaint ID": "900008", "Product": "Bank account or service",
                    "Issue": "Closing your account",
                    "Consumer complaint narrative": _ENGLISH + " (alias)."}),
    ]  # fmt: skip


@pytest.fixture
def fake_cfpb(tmp_path, monkeypatch):
    june = _cfpb_zip(tmp_path / "june.zip", _cfpb_rows(1500) + _cfpb_edge_rows())
    dup = dict(_cfpb_rows(1500)[5], **{"Complaint ID": "990005"})  # same text as id 5
    july = _cfpb_zip(tmp_path / "july.zip", _cfpb_rows(500, start=2000) + [dup])
    files = {P.CFPB_SOURCES[0].url: june, P.CFPB_SOURCES[1].url: july}
    monkeypatch.setattr(P, "fetch", lambda source, cache_dir: files[source.url])
    return tmp_path


SMALL = {"train": 300, "val": 30, "test": 100}


def test_cfpb_record_filters():
    reasons = [P.cfpb_record(r) for r in _cfpb_edge_rows()]
    assert reasons[:7] == [
        "no_narrative", "outside_window", "length", "not_english", "response_unresolved",
        "issue_not_in_options", "product_unmapped",
    ]  # fmt: skip
    rec = DataRecord(split="test", **reasons[7])
    assert rec.gold == {
        "product": "checking_savings", "issue": "closing_account", "monetary_relief": False
    }  # fmt: skip
    assert rec.meta["product"] == "Bank account or service"


def test_cfpb_questions_fit_the_contract():
    assert len(P.CFPB_QUESTIONS["issue"].options) == 26
    assert len(P.CFPB_QUESTIONS["product"].options) == len(P.PRODUCTS)
    assert set(P.ISSUE_ALIASES.values()) <= set(P.ISSUES)
    assert set(P.PRODUCT_ALIASES.values()) <= set(P.PRODUCTS)


def test_prepare_cfpb_end_to_end(fake_cfpb):
    out = fake_cfpb / "out"
    stats = P.prepare_cfpb(out, fake_cfpb, sizes=SMALL)
    records = {s: list(read_jsonl(out / f"cfpb_{s}.jsonl", DataRecord)) for s in SMALL}
    assert {s: len(rs) for s, rs in records.items()} == SMALL
    ds = stats["datasets"][P.CFPB]
    assert ds["records"] == SMALL
    assert ds["duplicates_dropped"] == 1  # id 5 and its exact copy: one is kept
    assert ds["near_duplicates_dropped"] == 0  # the fixture texts are all distinct
    assert ds["rows_skipped"] == {
        "issue_not_in_options": 1, "length": 1, "no_narrative": 1, "not_english": 1,
        "outside_window": 1, "product_unmapped": 1, "response_unresolved": 1,
    }  # fmt: skip
    every = [r for rs in records.values() for r in rs]
    assert len({r.state for r in every}) == len(every)
    share = sum(r.gold["product"] == "credit_reporting" for r in every) / len(every)
    assert ds["credit_reporting_capped_out"] > 0 and share <= P.CFPB_CREDIT_REPORTING_CAP + 0.01
    assert all(P.CFPB_WINDOW[0] <= r.meta["date_received"] <= P.CFPB_WINDOW[1] for r in every)
    assert ds["filters"]["date_received"] == list(P.CFPB_WINDOW)
    assert json.loads((out / "cfpb_stats.json").read_text()) == stats


def test_prepare_cfpb_is_deterministic(fake_cfpb):
    P.prepare_cfpb(fake_cfpb / "a", fake_cfpb, sizes=SMALL)
    P.prepare_cfpb(fake_cfpb / "b", fake_cfpb, sizes=SMALL)
    for s in SMALL:
        a = (fake_cfpb / "a" / f"cfpb_{s}.jsonl").read_bytes()
        assert a == (fake_cfpb / "b" / f"cfpb_{s}.jsonl").read_bytes()


def test_cap_share_leaves_small_products_alone():
    recs = [{"gold": {"product": "a" if i < 10 else "b"}, "meta": {"source_row": i}}
            for i in range(100)]  # fmt: skip
    for r in recs:
        r["questions"] = {"product": None}
    assert P.cap_share(recs, "a", 0.25, random.Random(0)) == (recs, 0)
    out, dropped = P.cap_share(recs, "b", 0.5, random.Random(0))
    assert dropped == 80 and sum(r["gold"]["product"] == "b" for r in out) == 10


def test_near_dedupe_collapses_template_letters():
    rng = random.Random(3)
    letter = "I am disputing this debt under the FDCPA and request validation " + _words(rng)
    recs = [
        {"state": letter + " account ending one", "meta": {"source_row": 7}},
        {"state": letter.upper() + " account ending two", "meta": {"source_row": 3}},
        {"state": _ENGLISH + " " + _words(rng), "meta": {"source_row": 5}},
        {"state": _ENGLISH + " " + _words(rng), "meta": {"source_row": 9}},
    ]
    kept, dropped = P.near_dedupe(recs)
    assert dropped == 1
    assert [r["meta"]["source_row"] for r in kept] == [3, 5, 9]  # earliest copy kept


# --- TeleLogs 5G root cause (Decision 17) ------------------------------------------------

_TELELOGS_HEAD = (
    "Analyze the 5G wireless network drive-test user plane data and engineering parameters.\n"
    "Identify the reason for the throughput dropping below 600Mbps in certain road sections.\n"
    "From the following 8 potential root causes, select the most likely one and enclose its "
    "number in \\boxed{{}} in the final answer.\n\n"
    + "".join(f"{c}: {desc}\n" for c, (_, desc) in P.ROOT_CAUSES.items())
    + "\n"
)


def _telelogs_rows(segments=6):
    rng = random.Random(17)
    rows = []
    for s in range(segments):
        for c in P.ROOT_CAUSES:
            table = "\n".join(
                f"2025-05-07 10:{s:02d}:{t:02d}.000000|{rng.uniform(0, 1200):.2f}|{c}"
                for t in range(5)
            )
            body = f"Given:\n- downtilt 255 is 6 degrees.\n\nUser plane data:\n{table}\n"
            rows.append({"question": _TELELOGS_HEAD + body, "answer": c})
    return rows


@pytest.fixture
def fake_telelogs(tmp_path, monkeypatch):
    path = tmp_path / "telelogs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _telelogs_rows()), encoding="utf-8")
    monkeypatch.setattr(P, "fetch_rows", lambda source, cache_dir: path)
    return tmp_path


def test_telelogs_record_strips_task_and_lettered_options():
    rec = P.telelogs_record(3, _telelogs_rows(1)[3])
    assert rec["state"].startswith("Given:")
    assert "C1:" not in rec["state"] and "boxed" not in rec["state"]
    assert rec["gold"] == {"root_cause": "overlapping_coverage"}
    assert rec["meta"]["segment"] == "2025-05-07 10:00:00"
    assert set(P.TELELOGS_QUESTIONS["root_cause"].options) == {k for k, _ in P.ROOT_CAUSES.values()}
    DataRecord(split="test", **rec)  # fits the contract


def test_telelogs_state_rejects_cases_without_data():
    with pytest.raises(ValueError, match="Given"):
        P.telelogs_state("C1: something\n")


def test_prepare_telelogs_splits_by_segment(fake_telelogs):
    out = fake_telelogs / "out"
    sizes = {"train": 4, "val": 1, "test": 1}
    stats = P.prepare_telelogs(out, fake_telelogs, segments=sizes)
    records = {s: list(read_jsonl(out / f"telelogs_{s}.jsonl", DataRecord)) for s in sizes}
    assert {s: len(r) for s, r in records.items()} == {"train": 32, "val": 8, "test": 8}
    segs = {s: {r.meta["segment"] for r in rs} for s, rs in records.items()}
    assert sum(len(v) for v in segs.values()) == 6  # no segment is in two splits
    for rs in records.values():  # every segment holds each cause once: exactly balanced
        assert len(set(Counter(r.gold["root_cause"] for r in rs).values())) == 1
    assert stats["datasets"]["telelogs"]["segments"] == 6
    assert json.loads((out / "telelogs_stats.json").read_text()) == stats


def test_prepare_telelogs_is_deterministic(fake_telelogs):
    sizes = {"train": 4, "val": 1, "test": 1}
    P.prepare_telelogs(fake_telelogs / "a", fake_telelogs, segments=sizes)
    P.prepare_telelogs(fake_telelogs / "b", fake_telelogs, segments=sizes)
    for s in sizes:
        a = (fake_telelogs / "a" / f"telelogs_{s}.jsonl").read_bytes()
        assert a == (fake_telelogs / "b" / f"telelogs_{s}.jsonl").read_bytes()


def test_grouped_split_checks_group_count(fake_telelogs):
    with pytest.raises(ValueError, match="groups"):
        P.prepare_telelogs(fake_telelogs / "c", fake_telelogs, segments={"train": 1})


# --- CVE advisories (Decision 17) --------------------------------------------------------


def test_redact_cvss_drops_every_score_vector_and_rating():
    text = (
        "A remote attacker can crash the server. CVSS 3.1 Base Score 7.5 (Availability "
        "impacts). CVSS Vector: (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H).\n"
        "The attack works over the network (AV:N). This flaw has been rated as having "
        "Important impact. Red Hat considers this a moderate severity issue. "
        "(Chromium security severity: High) Updates are available."
    )
    out = P.redact_cvss(text)
    assert out == "A remote attacker can crash the server.\nUpdates are available."


def test_severity_band_matches_cvss_ratings():
    bands = [P.severity_band(s) for s in (0.0, 0.1, 3.9, 4.0, 6.9, 7.0, 8.9, 9.0, 10.0)]
    assert bands == [None, "low", "low", "medium", "medium", "high", "high", "critical", "critical"]


def _rh_cve(i, *, vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", score="9.8", **kw):
    words = random.Random(i).sample(
        [
            "buffer",
            "overflow",
            "parser",
            "crafted",
            "request",
            "remote",
            "attacker",
            "heap",
            "read",
            "write",
            "memory",
            "server",
            "client",
            "library",
            "header",
            "cookie",
            "token",
            "session",
            "archive",
            "image",
            "font",
            "socket",
            "driver",
            "module",
            "packet",
            "handler",
            "queue",
            "timer",
            "cache",
            "index",
        ],
        12,
    )
    text = f"A flaw in the {' '.join(words)} component allows an attacker to cause harm."
    return {
        "name": f"CVE-2026-{1000 + i}",
        "public_date": "2026-05-01T00:00:00Z",
        "threat_severity": "Important",
        "bugzilla": {"description": f"pkg{i}: {words[0]} flaw"},
        "cvss3": {"cvss3_base_score": score, "cvss3_scoring_vector": vector},
        "details": [text, text],  # Red Hat often repeats the upstream text
        "statement": "This flaw has been rated as having Important impact.",
        "mitigation": {"value": "Disable the module until updated."},
        **kw,
    }


def test_cve_record_gold_and_state():
    rec = P.cve_record(0, _rh_cve(0))
    assert rec["gold"] == {"severity": 3, "attack_vector": "network"}
    assert rec["state"].count("A flaw in the") == 1  # repeated details collapsed
    assert "Statement" not in rec["state"]  # it held only the rating
    assert rec["state"].endswith("Mitigation: Disable the module until updated.")
    assert rec["meta"]["source_url"].endswith("/security/cve/CVE-2026-1000")
    DataRecord(split="test", **rec)  # fits the contract


def test_cve_record_skips():
    assert P.cve_record(0, _rh_cve(0, cvss3={})) == "no CVSS 3 score"
    assert P.cve_record(0, _rh_cve(0, score="0.0")) == "no CVSS 3 score"
    assert P.cve_record(0, _rh_cve(0, bugzilla={})) == "no title"
    dup = _rh_cve(0, statement="This CVE is a duplicate assignment of CVE-2026-1.")
    assert P.cve_record(0, dup) == "duplicate or not a flaw"
    assert P.cve_record(0, _rh_cve(0, details=["short"])) == "length"


def test_cve_candidates_cap_kernel():
    listing = [
        {
            "CVE": f"CVE-2026-{i}",
            "bugzilla_description": "kernel: x" if i % 2 else "lib: x",
            "cvss3_score": "5.5",
            "cvss3_scoring_vector": "CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H",
        }
        for i in range(400)
    ]
    picked = P.cve_candidates(listing, 100, 0.25, seed=1)
    assert len(picked) == 100
    assert sum(r["bugzilla_description"].startswith("kernel") for r in picked) == 25


def _snapshot(path, cves):
    import gzip

    with gzip.open(path, "wt", encoding="utf-8") as f:
        for c in cves:
            f.write(json.dumps(c) + "\n")
    return path


def test_prepare_cve_from_snapshot(tmp_path):
    vectors = ["AV:N", "AV:L", "AV:A"]
    cves = [
        _rh_cve(
            i,
            vector=f"CVSS:3.1/{vectors[i % 3]}/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            score=str([3.1, 5.5, 7.5, 9.8][i % 4]),
        )
        for i in range(120)
    ]
    snap = _snapshot(tmp_path / "cve_raw.jsonl.gz", cves)
    sizes = {"train": 60, "val": 10, "test": 20}
    stats = P.prepare_cve(tmp_path / "out", tmp_path, sizes=sizes, snapshot=snap)
    records = {s: list(read_jsonl(tmp_path / "out" / f"cve_{s}.jsonl", DataRecord)) for s in sizes}
    assert {s: len(r) for s, r in records.items()} == sizes
    for rs in records.values():
        for r in rs:
            assert "CVSS" not in r.state and "Important impact" not in r.state
    ds = stats["datasets"]["cve-advisories"]
    assert ds["licence"].startswith("CC BY 4.0")
    assert (tmp_path / "out" / "cve_raw.jsonl.gz").read_bytes() == snap.read_bytes()
    P.prepare_cve(tmp_path / "again", tmp_path, sizes=sizes, snapshot=snap)
    for s in sizes:
        a = (tmp_path / "out" / f"cve_{s}.jsonl").read_bytes()
        assert a == (tmp_path / "again" / f"cve_{s}.jsonl").read_bytes()

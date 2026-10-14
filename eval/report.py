"""Metrics and charts for every system, from downloaded outputs (CPU, no GCP).

    uv run --group eval python -m eval.report --gold test.jsonl --out report/ \\
        --run outputs-gce-123-1 --run outputs-gemini-456-1 --run outputs-typesafe-789-1 ...

Each ``--run`` is a directory as ``gh run download`` leaves it. It may hold (all optional
except ``predictions.jsonl``):

- ``predictions.jsonl``: any system's contract predictions
- ``cost.json``: the job's cost (runner: whole VM, boot to delete; baselines: API spend)
- ``eval_timings.json``: SnapCall only, seconds spent in the eval loop (steady-state cost)
- ``latency.jsonl``, ``label_prior.json``, ``equivalence.json``: from ``eval.run_eval``
- ``_runner_launch.json``, ``_runner_timings.json``: where the runner found an L4, and how long

Writes ``report.json`` (every number the charts use), ``metrics/<system>[_<dataset>].json``,
``summary.md`` and ``charts/*.png``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.metrics import compute_all, coverage_curve, join
from snapcall.schema import Cost, DataRecord, Prediction, read_jsonl


def question_order(records: list[DataRecord]) -> tuple[list[str], list[str]]:
    """Question ids in data order (by dataset, then as in each record), and the yes/no ones.
    Taken from the gold file, so any test set works (support tickets, telco, OOD)."""
    qs: dict[str, str] = {}
    for r in records:
        for qid, q in r.questions.items():
            qs.setdefault(qid, q.type)
    return list(qs), [q for q, t in qs.items() if t == "yesno"]


# Reference categorical palette (dataviz skill, light mode), in fixed slot order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Reliability bins with fewer items than this are left off the chart (too noisy to read).
MIN_BIN = 15
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_run(d: Path) -> dict:
    """One system's outputs. LoRA runs (train_lora) nest them: the root (or ``ood/``) holds the
    calibrated predictions, ``uncalibrated/`` the same pass at T = 1 (with its eval timing),
    and the run root the job's ``cost.json`` (training included) and ``train_summary.json``."""

    def j(name, *dirs):
        for base in dirs or (d,):
            p = base / name
            if p.exists():
                return json.loads(p.read_text())
        return None

    preds = list(read_jsonl(d / "predictions.jsonl", Prediction))
    systems = {p.system for p in preds}
    if len(systems) != 1:
        raise ValueError(f"{d}: expected one system, got {systems}")
    latency = None
    if (d / "latency.jsonl").exists():
        latency = [json.loads(x) for x in (d / "latency.jsonl").read_text().splitlines()]
    cost = j("cost.json", d, d.parent)
    trained = j("train_summary.json", d, d.parent) is not None
    return {
        "dir": str(d),
        "system": systems.pop(),
        "predictions": preds,
        "cost": Cost.model_validate(cost).model_dump(mode="json") if cost else None,
        # Calibration is a rescale after the forward pass: the T = 1 pass's timing applies.
        "eval_timings": j("eval_timings.json", d, d / "uncalibrated"),
        "trained": trained,
        "latency": latency,
        "label_prior": j("label_prior.json"),
        "equivalence": j("equivalence.json"),
        "launch": j("_runner_launch.json"),
        "runner_timings": j("_runner_timings.json"),
    }


def cost_per_1k(run: dict) -> dict:
    """GBP per 1,000 decisions. SnapCall: steady-state (the eval loop at the VM's hourly
    rate) and whole-job (boot to delete, incl. install and weights). Baselines: API spend.
    For a LoRA run the job also trained the adapter, so ``job_per_1k`` is left out (the job's
    total is still in ``job_gbp``, flagged by ``job_includes_training``)."""
    cost, n = run["cost"], None
    out: dict = {}
    if cost and run.get("trained"):
        out["job_gbp"] = cost["total"]
        out["job_includes_training"] = True
    elif cost:
        n = cost.get("n_decisions") or len(run["predictions"])
        out["job_gbp"] = cost["total"]
        out["job_per_1k"] = cost["total"] / n * 1000 if n else None
    t = run["eval_timings"]
    if t and cost and cost["compute"]:
        rate = cost["compute"][0]["rate_per_hour"]
        out["rate_gbp_per_hour"] = rate
        out["steady_per_1k"] = rate * t["eval_seconds"] / 3600 / t["n_decisions"] * 1000
        out["decisions_per_second"] = t["n_decisions"] / t["eval_seconds"]
        out["tokens_per_second"] = t["input_tokens"] / t["eval_seconds"]
    return out


def latency_curve(rows: list[dict]) -> list[dict]:
    """Mean and p50 latency per (forward, debias, n_questions)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["forward"], r["debias"], r["n_questions"])].append(r)
    out = []
    for (fwd, debias, n_q), rs in sorted(groups.items()):
        ms = np.array([r["latency_ms"] for r in rs])
        out.append(
            {
                "forward": fwd,
                "debias": debias,
                "n_questions": n_q,
                "mean_ms": float(ms.mean()),
                "p50_ms": float(np.median(ms)),
                "input_tokens": float(np.mean([r["input_tokens"] for r in rs])),
                "compute_tokens": float(np.mean([r["compute_tokens"] for r in rs])),
                "n_branches": float(np.mean([r["n_branches"] for r in rs])),
                "forward_used": sorted({r["forward_used"] for r in rs}),
            }
        )
    return out


def build(runs: list[dict], records: list[DataRecord]) -> dict:
    questions, yesno = question_order(records)
    report: dict = {"systems": {}, "questions": questions, "yesno": yesno}
    for run in runs:
        ms = compute_all(run["predictions"], records, system=run["system"])
        m = ms["all"]
        per_q = {}
        for qid, tm in m.by_question.items():
            per_q[qid] = {
                "accuracy": tm.accuracy,
                "ece": tm.ece,
                "brier": tm.brier,
                "log_loss": tm.log_loss,
                "mae": tm.mae,
                "n_errors": tm.n_errors,
                "macro_f1": m.extra.get(f"macro_f1__{qid}"),
                "auroc": m.extra.get(f"auroc__{qid}"),
                "ap": m.extra.get(f"ap__{qid}"),
                "recall": m.extra.get(f"recall__{qid}"),
                "precision": m.extra.get(f"precision__{qid}"),
                "positive_rate": m.extra.get(f"positive_rate__{qid}"),
                "order_tv": m.extra.get(f"order_tv__{qid}"),
                "order_flip_rate": m.extra.get(f"order_flip_rate__{qid}"),
                "first_perm_accuracy": m.extra.get(f"first_perm_accuracy__{qid}"),
                "reliability": [b.model_dump() for b in tm.reliability or []],
            }
        by_type = {
            t: {
                "accuracy": tm.accuracy,
                "ece": tm.ece,
                "brier": tm.brier,
                "reliability": [b.model_dump() for b in tm.reliability or []],
            }
            for t, tm in m.by_type.items()
        }
        report["systems"][run["system"]] = {
            "dir": run["dir"],
            "overall": m.overall.model_dump(exclude={"reliability"}),
            "macro_f1_mean": m.extra.get("macro_f1_mean"),
            "order_sensitivity": m.order_sensitivity,
            "order_flip_rate": m.extra.get("order_flip_rate"),
            "first_perm_accuracy": m.extra.get("first_perm_accuracy"),
            "latency": m.latency.model_dump() if m.latency else None,
            "by_question": per_q,
            "by_type": by_type,
            "by_dataset": {
                ds: {"accuracy": x.overall.accuracy, "ece": x.overall.ece}
                for ds, x in ms.items()
                if ds != "all"
            },
            "cost": cost_per_1k(run),
            "coverage": coverage_curve(join(run["predictions"], records)),
            "eval_timings": run["eval_timings"],
            "latency_curve": latency_curve(run["latency"]) if run["latency"] else None,
            "label_prior": run["label_prior"],
            "equivalence": run["equivalence"],
            "launch": run["launch"],
            "runner_timings": run["runner_timings"],
        }
        run["metrics"] = ms
    return report


# --------------------------------------------------------------------------------------
# Charts (matplotlib, PNG, for the post)
# --------------------------------------------------------------------------------------


def _style(plt):
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9,
            "legend.frameon": False,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def colours(systems: list[str]) -> dict[str, str]:
    return {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(systems)}


def grouped_bars(ax, report, systems, metric, questions, col, title, ylim=(0, 1)):
    n = len(systems)
    width = 0.8 / n
    x = np.arange(len(questions))
    for i, s in enumerate(systems):
        qs = report["systems"][s]["by_question"]
        vals = [qs.get(q, {}).get(metric) for q in questions]
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.9, color=col[s], label=s)
    ax.set_xticks(x, questions)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)


def charts(report: dict, out: Path, systems: list[str]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    out.mkdir(parents=True, exist_ok=True)
    col = colours(systems)
    made = []
    qs = [
        q
        for q in report["questions"]
        if any(q in report["systems"][s]["by_question"] for s in systems)
    ]

    def save(fig, name):
        fig.savefig(out / name)
        plt.close(fig)
        made.append(name)

    # 1. accuracy and macro-F1 per question
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    grouped_bars(axes[0], report, systems, "accuracy", qs, col, "Accuracy per question")
    grouped_bars(
        axes[1],
        report,
        systems,
        "macro_f1",
        qs,
        col,
        "Macro-F1 per question (rare classes count as much as common ones)",
    )
    axes[0].legend(ncol=min(4, len(systems)), loc="upper left", bbox_to_anchor=(0, 1.35))
    save(fig, "accuracy_f1.png")

    # 2. calibration: ECE and Brier per question
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ece_max = max(
        (
            (report["systems"][s]["by_question"].get(q, {}).get("ece") or 0)
            for s in systems
            for q in qs
        ),
        default=0.0,
    )
    grouped_bars(
        axes[0],
        report,
        systems,
        "ece",
        qs,
        col,
        "ECE (lower is better)",
        (0, max(0.05, ece_max * 1.1)),
    )
    grouped_bars(
        axes[1], report, systems, "brier", qs, col, "Brier score (lower is better)", (0, None)
    )
    axes[0].legend(ncol=min(4, len(systems)), loc="upper left", bbox_to_anchor=(0, 1.35))
    save(fig, "calibration.png")

    # 3. reliability diagrams, one per system, all questions pooled by type
    n = len(systems)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.9), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], systems, strict=True):
        ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (3, 3)))
        for t, marker in (("choice", "o"), ("score", "s"), ("yesno", "^")):
            bins = report["systems"][s]["by_type"].get(t, {}).get("reliability", [])
            pts = [
                (b["mean_confidence"], b["accuracy"], b["count"])
                for b in bins
                if b["count"] >= MIN_BIN
            ]
            if pts:
                x, y, c = zip(*pts, strict=True)
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    ms=4,
                    lw=1.5,
                    label=t,
                    color=PALETTE[["choice", "score", "yesno"].index(t)],
                )
        ax.set_title(s, fontsize=8, loc="left")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1)
        ax.set_xlabel("stated probability")
    axes[0][0].set_ylabel(f"share correct (bins with >= {MIN_BIN} answers)")
    axes[0][0].legend(loc="upper left", fontsize=7)
    save(fig, "reliability.png")

    # 4. yes/no: AUROC and average precision (the ~10% positives make accuracy useless)
    yq = [q for q in report["yesno"] if q in qs]
    if yq:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
        grouped_bars(axes[0], report, systems, "auroc", yq, col, "Yes/no AUROC", (0.4, 1))
        grouped_bars(
            axes[1],
            report,
            systems,
            "ap",
            yq,
            col,
            "Yes/no average precision (chance = positive rate)",
            (0, 1),
        )
        for q_i, q in enumerate(yq):
            pr = next(
                (
                    report["systems"][s]["by_question"][q]["positive_rate"]
                    for s in systems
                    if q in report["systems"][s]["by_question"]
                ),
                None,
            )
            if pr is not None:
                axes[1].hlines(pr, q_i - 0.45, q_i + 0.45, colors=INK_2, lw=1, ls=(0, (3, 3)))
        axes[0].legend(ncol=min(4, len(systems)), loc="upper left", bbox_to_anchor=(0, 1.45))
        save(fig, "yesno_ranking.png")

    # 5. order sensitivity (SnapCall only): debias off (first order) vs on, per question
    snap = [s for s in systems if report["systems"][s]["order_sensitivity"] is not None]
    if snap:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        width = 0.8 / len(snap)
        x = np.arange(len(qs))
        for i, s in enumerate(snap):
            bq = report["systems"][s]["by_question"]
            off = [bq.get(q, {}).get("first_perm_accuracy") or np.nan for q in qs]
            on = [bq.get(q, {}).get("accuracy") or np.nan for q in qs]
            xs = x + (i - (len(snap) - 1) / 2) * width
            axes[0].vlines(xs, off, on, colors=col[s], lw=2)
            axes[0].scatter(xs, off, s=18, facecolors=SURFACE, edgecolors=col[s], zorder=3)
            axes[0].scatter(xs, on, s=18, color=col[s], zorder=3, label=s)
            flips = [bq.get(q, {}).get("order_flip_rate") or np.nan for q in qs]
            axes[1].bar(xs, flips, width * 0.9, color=col[s], label=s)
        axes[0].set_title(
            "Accuracy: one order (open) vs averaged (filled)",
            loc="left",
            fontsize=9,
        )
        axes[1].set_title("Answer changes with option order (share)", loc="left", fontsize=9)
        for ax in axes:
            ax.set_xticks(x, qs, rotation=30, ha="right")
            ax.set_ylim(0, 1)
            ax.grid(axis="x", visible=False)
        axes[1].legend(fontsize=7)
        save(fig, "order_sensitivity.png")

    # 6. latency vs questions per state
    lat = [s for s in systems if report["systems"][s]["latency_curve"]]
    if lat:
        fig, axes = plt.subplots(
            1, len(lat), figsize=(3.4 * len(lat), 3.2), squeeze=False, sharey=True
        )
        styles = {"off": "-", "auto": (0, (4, 2))}
        fwd_col = {"packed": PALETTE[0], "cached": PALETTE[1], "separate": PALETTE[2]}
        for ax, s in zip(axes[0], lat, strict=True):
            curve = report["systems"][s]["latency_curve"]
            for fwd in ("packed", "cached", "separate"):
                for debias in ("off", "auto"):
                    pts = [
                        (c["n_questions"], c["p50_ms"])
                        for c in curve
                        if c["forward"] == fwd and c["debias"] == debias
                    ]
                    if pts:
                        xx, yy = zip(*pts, strict=True)
                        ax.plot(
                            xx,
                            yy,
                            color=fwd_col[fwd],
                            ls=styles[debias],
                            lw=2,
                            marker="o",
                            ms=4,
                            label=f"{fwd}, debias {debias}",
                        )
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
            ax.set_xlabel("questions per state")
            ax.set_title(s, fontsize=8, loc="left")
        axes[0][0].set_ylabel("p50 latency (ms)")
        axes[0][0].legend(fontsize=6.5)
        save(fig, "latency_vs_questions.png")

    # 7. cost per 1k decisions and latency per request (two charts, log scales)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(systems) + 1.4))
    y = np.arange(len(systems))
    cost = []
    for s in systems:
        c = report["systems"][s]["cost"]
        cost.append(c.get("steady_per_1k") or c.get("job_per_1k") or np.nan)
    axes[0].barh(y, cost, color=[col[s] for s in systems], height=0.6)
    axes[0].set_xscale("log")
    axes[0].set_title("GBP per 1,000 decisions (log scale)", loc="left")
    lat_ms = [(report["systems"][s]["latency"] or {}).get("p50_ms", np.nan) for s in systems]
    axes[1].barh(y, lat_ms, color=[col[s] for s in systems], height=0.6)
    axes[1].set_xscale("log")
    axes[1].set_title("p50 latency per state, all questions (ms, log)", loc="left")
    for ax in axes:
        ax.set_yticks(y, systems)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    save(fig, "cost_latency.png")

    # 8. capacity search on the burst runner
    runs = [(s, report["systems"][s]["launch"]) for s in systems if report["systems"][s]["launch"]]
    if runs:
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(runs) + 1.2))
        for i, (_s, launch) in enumerate(runs):
            left = 0.0
            for a in launch["attempts"]:
                ok = a["outcome"] == "created"
                ax.barh(
                    i,
                    a["seconds"],
                    left=left,
                    height=0.55,
                    edgecolor=SURFACE,
                    lw=1.5,
                    color=PALETTE[2] if ok else GRID,
                )
                left += a["seconds"]
            ax.text(
                left + 1,
                i,
                f"{launch['zone']}  ({launch['search_seconds']:.0f}s, "
                f"{launch['zones_tried']} zone(s))",
                va="center",
                fontsize=8,
                color=INK,
            )
        ax.set_yticks(range(len(runs)), [s for s, _ in runs])
        ax.invert_yaxis()
        ax.set_xlabel("seconds from first create call (grey = zone out of L4s, green = created)")
        ax.set_title("Finding an L4: capacity search per job", loc="left")
        ax.grid(axis="y", visible=False)
        save(fig, "capacity_search.png")
    return made


# --------------------------------------------------------------------------------------
# Markdown summary
# --------------------------------------------------------------------------------------


def _f(v, fmt="{:.3f}"):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def summary_md(report: dict, systems: list[str]) -> str:
    S = report["systems"]
    lines = ["# SnapCall eval summary", ""]
    lines += [
        "## Overall",
        "",
        "| system | accuracy | macro-F1 (mean) | ECE | Brier | p50 ms/state | GBP / 1k |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in systems:
        o, c = S[s]["overall"], S[s]["cost"]
        lines.append(
            f"| {s} | {_f(o['accuracy'])} | {_f(S[s]['macro_f1_mean'])} | {_f(o['ece'])} | "
            f"{_f(o['brier'])} | {_f((S[s]['latency'] or {}).get('p50_ms'), '{:.0f}')} | "
            f"{_f(c.get('steady_per_1k') or c.get('job_per_1k'), '{:.4f}')} |"
        )
    for metric in ("accuracy", "macro_f1", "ece"):
        lines += [
            "",
            f"## {metric} per question",
            "",
            "| system | " + " | ".join(report["questions"]) + " |",
            "|---|" + "---|" * len(report["questions"]),
        ]
        for s in systems:
            row = [_f(S[s]["by_question"].get(q, {}).get(metric)) for q in report["questions"]]
            lines.append(f"| {s} | " + " | ".join(row) + " |")
    lines += [
        "",
        "## Yes/no ranking",
        "",
        "| system | question | AUROC | AP | recall | precision |",
        "|---|---|---|---|---|---|",
    ]
    for s in systems:
        for q in report["yesno"]:
            x = S[s]["by_question"].get(q)
            if x:
                lines.append(
                    f"| {s} | {q} | {_f(x['auroc'])} | {_f(x['ap'])} | "
                    f"{_f(x['recall'])} | {_f(x['precision'])} |"
                )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.report")
    p.add_argument("--gold", required=True, help="test.jsonl (data/prepare.py output)")
    p.add_argument("--run", action="append", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("report"))
    p.add_argument("--no-charts", action="store_true")
    a = p.parse_args(argv)
    records = list(read_jsonl(a.gold, DataRecord))
    runs = [load_run(d) for d in a.run]
    report = build(runs, records)
    systems = [r["system"] for r in runs]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    mdir = a.out / "metrics"
    mdir.mkdir(exist_ok=True)
    for run in runs:
        for ds, m in run["metrics"].items():
            suffix = "" if ds == "all" else f"_{ds}"
            (mdir / f"{run['system']}{suffix}.json").write_text(m.model_dump_json(indent=2))
    (a.out / "summary.md").write_text(summary_md(report, systems))
    if not a.no_charts:
        charts(report, a.out / "charts", systems)
    print((a.out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

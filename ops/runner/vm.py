"""Burst-runner helper that runs on the VM (Decision 9). Stdlib only, Python 3.10+.

``startup.sh`` writes this file from the VM's metadata before anything else, so it works even
if downloading the code fails. It runs under the image's own ``python3``.

    vm.py models --args "smoke --model {models}/Qwen/Qwen3-0.6B" --zone asia-east1-a \\
        --buckets us-central1=P-models,europe-west4=P-models-europe-west4 --dest /opt/models
        --args-out FILE
        # fetches each {models}/<HF repo> (regional bucket if this region has one, else
        # Hugging Face) and writes the args, {models} replaced by --dest, to FILE
    vm.py finish --outputs DIR --exit-code N --timings FILE ...
        # writes DIR/cost.json (unless the module wrote one), DIR/_runner_timings.json and
        # prints done.json

Progress goes to stderr, so stdout carries only what the caller asked for.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODELS = "{models}"
# Not datetime.UTC: that needs 3.11, and the image's python3 may be 3.10.
UTC_TZ = timezone.utc  # noqa: UP017
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def log(msg: str) -> None:
    print(f"[vm] {msg}", file=sys.stderr, flush=True)


def region_of(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def parse_buckets(text: str) -> dict[str, str]:
    pairs = (item.split("=", 1) for item in text.split(",") if "=" in item)
    return {region.strip(): bucket.strip() for region, bucket in pairs}


def model_repos(args: str) -> list[str]:
    """HF repo ids referenced as ``{models}/<org>/<name>`` in the job args."""
    repos = []
    for token in shlex.split(args):
        if MODELS in token:
            repo = token.split(f"{MODELS}/", 1)[-1].strip("/")
            if not REPO_RE.match(repo):
                raise ValueError(f"{token!r}: expected {MODELS}/<org>/<name>")
            repos.append(repo)
    return list(dict.fromkeys(repos))


def rewrite_args(args: str, dest: str) -> str:
    return args.replace(MODELS, dest.rstrip("/"))


def fetch(repo: str, dest: Path, bucket: str | None, hf: str, run=subprocess.run) -> str:
    """Put ``repo`` at ``dest/repo``. Regional bucket first (Google network, no egress), else
    Hugging Face (ungated, no token). Returns where it came from."""
    target = dest / repo
    target.mkdir(parents=True, exist_ok=True)
    if bucket:
        cmd = ["gcloud", "storage", "rsync", "-r", f"gs://{bucket}/{repo}", str(target)]
        r = run(cmd, stdout=sys.stderr)
        if r.returncode == 0 and any(target.iterdir()):
            return f"gs://{bucket}/{repo}"
        log(f"no copy of {repo} in gs://{bucket}; using Hugging Face")
    r = run([hf, "download", repo, "--local-dir", str(target)], stdout=sys.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"couldn't download {repo} from Hugging Face")
    return f"hf://{repo}"


def cmd_models(a: argparse.Namespace) -> int:
    bucket = parse_buckets(a.buckets).get(region_of(a.zone))
    sources = {}
    for repo in model_repos(a.args):
        t0 = time.monotonic()
        sources[repo] = fetch(repo, Path(a.dest), bucket, a.hf)
        log(f"{repo} from {sources[repo]} in {time.monotonic() - t0:.0f}s")
    if a.record:
        Path(a.record).write_text(json.dumps(sources, indent=2) + "\n")
    Path(a.args_out).write_text(rewrite_args(a.args, a.dest))
    return 0


def read_timings(path: Path) -> dict[str, float]:
    """``name=<epoch seconds>`` lines, as startup.sh appends them."""
    out: dict[str, float] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            name, _, value = line.partition("=")
            try:
                out[name.strip()] = float(value)
            except ValueError:
                continue
    return out


def phases(t: dict[str, float]) -> dict[str, float]:
    """Seconds spent in each phase, from the timestamps that exist."""
    steps = [
        ("boot_to_script", "boot", "script_start"),
        ("gpu_driver_wait", "script_start", "gpu_ready"),
        ("install", "gpu_ready", "installed"),
        ("weights", "installed", "weights_ready"),
        ("module", "module_start", "module_end"),
        ("total_billed_estimate", "boot", "finish"),
    ]
    return {name: round(t[b] - t[a], 1) for name, a, b in steps if a in t and b in t}


def cost_json(
    run_id: str, module: str, machine_type: str, rate: float, minutes: float, zone: str
) -> dict:
    minutes = round(minutes, 3)
    cost = minutes / 60 * rate
    return {
        "schema_version": "1",
        "run_id": run_id,
        "job": f"gce:{module}",
        "created_at": datetime.now(UTC_TZ).isoformat(),
        "currency": "GBP",
        "compute": [
            {
                "resource": machine_type,
                "count": 1,
                "minutes": minutes,
                "spot": False,
                "rate_per_hour": rate,
                "cost": cost,
            }
        ],
        "api": [],
        "total": cost,
        "notes": (
            f"Written by ops/runner/vm.py: VM boot to self-delete x the us-central1 on-demand "
            f"rate for {machine_type} (whole VM incl. L4). Ran in {zone}; other regions cost up "
            "to ~25% more. Excludes the boot disk (~£0.02/h) and the seconds before boot."
        ),
    }


def cmd_finish(a: argparse.Namespace) -> int:
    outputs = Path(a.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    t = read_timings(Path(a.timings))
    t.setdefault("finish", time.time())
    boot = t.get("boot", t["finish"])
    cost_path = outputs / "cost.json"
    if cost_path.exists():
        log(f"{cost_path} written by the module; leaving it")
    else:
        minutes = (t["finish"] - boot) / 60
        body = cost_json(a.run_id, a.module, a.machine_type, a.rate, minutes, a.zone)
        cost_path.write_text(json.dumps(body, indent=2) + "\n")
    timings = {
        "zone": a.zone,
        "phases_seconds": phases(t),
        "timestamps": {k: datetime.fromtimestamp(v, UTC_TZ).isoformat() for k, v in t.items()},
    }
    (outputs / "_runner_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    done = {
        "run_id": a.run_id,
        "vm": a.vm,
        "zone": a.zone,
        "exit_code": a.exit_code,
        "finished_at": datetime.fromtimestamp(t["finish"], UTC_TZ).isoformat(),
        "phases_seconds": timings["phases_seconds"],
    }
    print(json.dumps(done, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models")
    m.add_argument("--args", required=True)
    m.add_argument("--zone", required=True)
    m.add_argument("--buckets", default="")
    m.add_argument("--dest", default="/opt/models")
    m.add_argument("--hf", default=os.environ.get("SNAPCALL_HF_CLI", "hf"))
    m.add_argument("--args-out", required=True, help="write the rewritten args here")
    m.add_argument("--record", default="", help="write {repo: source} JSON here")
    f = sub.add_parser("finish")
    f.add_argument("--outputs", required=True)
    f.add_argument("--timings", required=True)
    f.add_argument("--exit-code", type=int, required=True)
    f.add_argument("--run-id", required=True)
    f.add_argument("--module", required=True)
    f.add_argument("--machine-type", required=True)
    f.add_argument("--rate", type=float, required=True)
    f.add_argument("--zone", required=True)
    f.add_argument("--vm", required=True)
    a = p.parse_args(argv)
    return cmd_models(a) if a.cmd == "models" else cmd_finish(a)


if __name__ == "__main__":
    sys.exit(main())

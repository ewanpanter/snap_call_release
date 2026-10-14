"""Submit one Vertex AI custom job running ``python -m MODULE ARGS`` in the thin image.

LEGACY (superseded by the burst runner, ops/runner/; Decision 9). Called by vertex-job.yml::

    uv run python scripts/submit_job.py --project P --region R --image IMAGE \\
      --module MODULE --args "ARGS" --run-id RUN_ID --output-uri gs://.../runs/RUN_ID/outputs \\
      --service-account SA [--gpu [--machine-type g2-standard-4|g2-standard-8]] \\
      --timeout 2h [--models-dir /gcs/BUCKET] [--spot | --flex-start [--max-wait 24h]] \\
      [--wait] [--dry-run]

- Returns as soon as Vertex accepts the job and prints its resource name (Decision 2).
  ``--wait`` blocks until it finishes and exits non-zero unless it succeeded.
- The job's display name is exactly RUN_ID: fetch-results.yml finds the job by it.
- ``--gpu`` picks a g2 machine (1x L4, implied by the type; default g2-standard-4, use
  g2-standard-8 for 8B). Without it the job runs on a CPU machine (n1-standard-4).
- ``--timeout`` is capped at 2h. It's run time; queueing is extra.
- ``--flex-start`` (GPU only): Dynamic Workload Scheduler. When no L4 is free, Vertex
  queues the job for up to ``--max-wait`` (default and cap 24h, under the janitor's 24h15m
  limit for jobs that haven't started; queueing is free) and starts it when one is. Once
  running it isn't preemptible. Needs the region's *preemptible* L4 Vertex quota, which DWS
  uses "only in name".
- ``--models-dir``: the job region's weights (``/gcs/<models bucket>``). ``{models}`` in
  ``--args`` is replaced with it, and the job gets it as ``SNAPCALL_MODELS_DIR``, so a
  command like ``smoke --model {models}/Qwen/Qwen3-4B-Instruct-2507`` works in any region.
- ``--dry-run`` prints the job spec and submits nothing.

Stdlib only, so the runner needs nothing beyond the project env. Submission goes through
``gcloud ai custom-jobs create --config``, which uses the runner's WIF credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

GPU_MACHINES = ("g2-standard-4", "g2-standard-8")  # both carry exactly 1x L4
CPU_MACHINE = "n1-standard-4"
MAX_TIMEOUT_S = 2 * 3600  # run time
MAX_WAIT_S = 24 * 3600  # flex-start queueing (free); janitor allows 24h15m

# Estimated USD/hour for Vertex custom training in us-central1 (europe-west4 is roughly 10%
# more), whole machine including the L4: (on-demand, spot). Flex-start is billed at DWS
# pricing, usually below on-demand; the on-demand rate is used as an upper estimate. For
# cost.json only; check current pricing. Billing is the truth.
RATES_USD = {
    "g2-standard-4": (0.81, 0.33),
    "g2-standard-8": (0.98, 0.39),
    "n1-standard-4": (0.22, 0.05),
}
USD_TO_GBP = 0.75

RUN_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,15}-[0-9]+-[0-9]+$")  # as fetch-results.yml
MODULE_RE = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$")  # as vertex-job.yml
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def parse_timeout(text: str, cap_s: int = MAX_TIMEOUT_S) -> int:
    """``90m`` / ``2h`` / ``3600s`` / ``3600`` -> seconds, at most ``cap_s`` (default 2h)."""
    m = re.fullmatch(r"(\d+)([smh]?)", text.strip())
    if not m:
        raise ValueError(f"bad timeout {text!r} (use e.g. 90m or 2h)")
    seconds = int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]
    if not 0 < seconds <= cap_s:
        raise ValueError(f"must be between 1s and {cap_s // 3600}h, got {text!r}")
    return seconds


def rate_gbp_per_hour(machine_type: str, spot: bool) -> float:
    on_demand, spot_rate = RATES_USD[machine_type]
    return round((spot_rate if spot else on_demand) * USD_TO_GBP, 4)


def build_job_spec(
    *,
    image: str,
    module: str,
    args: str,
    run_id: str,
    output_uri: str,
    service_account: str,
    gpu: bool,
    machine_type: str | None,
    timeout_s: int,
    spot: bool,
    models_dir: str | None = None,
    flex_start: bool = False,
    max_wait_s: int = MAX_WAIT_S,
) -> dict:
    """The ``CustomJobSpec`` (Vertex v1 REST field names) for one single-replica job."""
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"bad run id {run_id!r} (expected e.g. exp-123456789-1)")
    if not MODULE_RE.match(module):
        raise ValueError(f"bad module name {module!r}")
    if flex_start and not gpu:
        raise ValueError("--flex-start is for GPU jobs only")
    if flex_start and spot:
        raise ValueError("choose --spot or --flex-start, not both")
    if "{models}" in args and not models_dir:
        raise ValueError("--args uses {models} but no --models-dir was given")
    if models_dir:
        args = args.replace("{models}", models_dir.rstrip("/"))
    if not output_uri.startswith("gs://"):
        raise ValueError(f"--output-uri must be a gs:// URI, got {output_uri!r}")
    if gpu:
        machine_type = machine_type or GPU_MACHINES[0]
        if machine_type not in GPU_MACHINES:
            raise ValueError(f"--machine-type must be one of {GPU_MACHINES}")
        machine_spec = {
            "machineType": machine_type,
            "acceleratorType": "NVIDIA_L4",
            "acceleratorCount": 1,
        }
        resource = "nvidia-l4"
    else:
        if machine_type:
            raise ValueError("--machine-type is only for --gpu runs")
        machine_type = CPU_MACHINE
        machine_spec = {"machineType": machine_type}
        resource = f"vertex-{machine_type}"

    env = {
        "SNAPCALL_RUN_ID": run_id,
        "SNAPCALL_OUTPUT_URI": output_uri.rstrip("/"),
        "SNAPCALL_MACHINE_TYPE": machine_type,
        "SNAPCALL_COMPUTE_RESOURCE": resource,
        "SNAPCALL_GPU": "true" if gpu else "false",
        "SNAPCALL_COMPUTE_COUNT": "1",
        "SNAPCALL_SPOT": "true" if spot else "false",
        "SNAPCALL_RATE_PER_HOUR": str(rate_gbp_per_hour(machine_type, spot)),
    }
    if models_dir:
        env["SNAPCALL_MODELS_DIR"] = models_dir.rstrip("/")
    scheduling = {"timeout": f"{timeout_s}s", "strategy": "SPOT" if spot else "STANDARD"}
    if flex_start:
        scheduling.update(strategy="FLEX_START", maxWaitDuration=f"{max_wait_s}s")
    return {
        "workerPoolSpecs": [
            {
                "machineSpec": machine_spec,
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": image,
                    "args": [module, *shlex.split(args)],
                    "env": [{"name": k, "value": v} for k, v in env.items()],
                },
            }
        ],
        "scheduling": scheduling,
        "serviceAccount": service_account,
    }


def _gcloud(cmd: list[str]) -> str:
    return subprocess.run(["gcloud", *cmd], check=True, capture_output=True, text=True).stdout


def submit(project: str, region: str, run_id: str, spec: dict, gpu: bool) -> str:
    """Create the job and return its resource name. Doesn't wait for it to run."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)  # JSON is valid YAML
    labels = f"snapcall=1,run_id={run_id},gpu={'true' if gpu else 'false'}"
    try:
        out = _gcloud(
        [
            "ai", "custom-jobs", "create",
            "--project", project,
            "--region", region,
            "--display-name", run_id,
            "--config", f.name,
            "--labels", labels,
            "--format", "value(name)",
        ])  # fmt: skip
    finally:
        os.unlink(f.name)
    name = out.strip()
    if not name.startswith("projects/"):
        raise RuntimeError(f"unexpected gcloud output: {name!r}")
    return name


def wait(name: str, region: str, poll_s: int = 60) -> str:
    while True:
        state = _gcloud(
            ["ai", "custom-jobs", "describe", name, "--region", region, "--format", "value(state)"]
        ).strip()
        print(f"{time.strftime('%H:%M:%S')} {state}", flush=True)
        if state in TERMINAL:
            return state
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="submit_job.py", description=__doc__.split("\n")[0])
    p.add_argument("--project", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--module", required=True, help="run as `python -m MODULE`")
    p.add_argument("--args", default="", help="one string, split shell-style")
    p.add_argument("--run-id", required=True, help="also the Vertex display name")
    p.add_argument("--output-uri", required=True, help="gs://.../runs/RUN_ID/outputs")
    p.add_argument("--service-account", required=True)
    p.add_argument("--gpu", action="store_true", help="1x L4 on a g2 machine")
    p.add_argument("--machine-type", choices=GPU_MACHINES, help="GPU runs only")
    p.add_argument("--timeout", default="2h", help="at most 2h")
    p.add_argument("--models-dir", help="weights in the job's region, e.g. /gcs/BUCKET")
    p.add_argument("--spot", action="store_true")
    p.add_argument("--flex-start", action="store_true", help="GPU: queue until an L4 is free")
    p.add_argument("--max-wait", default="24h", help="flex-start queueing limit, at most 24h")
    p.add_argument("--wait", action="store_true", help="block; non-zero exit unless it succeeds")
    p.add_argument("--dry-run", action="store_true", help="print the job spec, submit nothing")
    a = p.parse_args(argv)

    try:
        spec = build_job_spec(
            image=a.image,
            module=a.module,
            args=a.args,
            run_id=a.run_id,
            output_uri=a.output_uri,
            service_account=a.service_account,
            gpu=a.gpu,
            machine_type=a.machine_type,
            timeout_s=parse_timeout(a.timeout),
            spot=a.spot,
            models_dir=a.models_dir,
            flex_start=a.flex_start,
            max_wait_s=parse_timeout(a.max_wait, MAX_WAIT_S),
        )
    except ValueError as e:
        p.error(str(e))

    if a.dry_run:
        print(json.dumps(spec, indent=2))
        return 0

    try:
        name = submit(a.project, a.region, a.run_id, spec, a.gpu)
    except subprocess.CalledProcessError as e:
        print(f"gcloud failed ({e.returncode}):\n{e.stderr}", file=sys.stderr)
        return 1
    print(name)
    if not a.wait:
        return 0
    state = wait(name, a.region)
    return 0 if state == "JOB_STATE_SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())

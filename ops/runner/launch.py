"""Burst runner launcher (Decision 9): find an L4 somewhere and start one GPU VM on it.

Runs on the GitHub runner (run-gpu.yml) as ci-experiment. Stdlib only; talks to GCP through
the ``gcloud`` CLI. The capacity search *is* the create call: zones are tried in
``RUNNER_REGIONS`` order (every L4 zone of a region before the next region), a stockout
comes back in seconds, and the first create that succeeds wins. Nothing is created to
probe and then deleted.

Guardrails enforced here, before and on every create:
- refuse if ``--max-vms`` VMs labelled ``snapcall=runner`` already exist anywhere;
- every VM gets ``--max-run-duration`` (at most 2h) with ``--instance-termination-action
  DELETE``, so Google deletes it even if everything else fails;
- a create that fails for an unrecognised reason stops the search (a config or permission
  error would fail the same way in all 40-odd zones), after deleting the VM if the failed
  call left one behind.

Writes ``launch.json`` (``--out``): where the VM landed, and every attempt with its timing.

    python ops/runner/launch.py --project P --regions "asia-northeast1 ..." --run-id gce-1-1 \\
        --scratch gs://P-scratch --code-uri gs://P-scratch/runs/gce-1-1/code.tar.gz \\
        --module snapcall.decide --args "smoke --model {models}/Qwen/Qwen3-0.6B" --out launch.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME_PREFIX = "snapcall-runner"  # IAM only lets CI create VMs with this prefix (infra/bootstrap)
LABEL = "snapcall=runner"  # the janitor and the max-VMs check look for this
GPU_MACHINES = ("g2-standard-4", "g2-standard-8")  # both carry exactly 1x L4
MAX_RUN = "2h"  # hard cap; Google deletes the VM at this point whatever it's doing
IMAGE_PROJECT = "deeplearning-platform-release"
# Deep Learning VM "common" images: NVIDIA driver + CUDA, no framework (we install our own
# pinned torch with uv). Newest matching family wins; override with --image-family.
IMAGE_FAMILY_RE = r"^common-cu[0-9]+-ubuntu-(2204|2404)-nvidia-[0-9]+$"

# Estimated USD/hour, Compute Engine on-demand, us-central1, whole VM including the L4.
# Other regions cost up to ~25% more (Tokyo, Seoul, Zurich). For cost.json only; Billing is
# the truth.
RATES_USD = {"g2-standard-4": 0.71, "g2-standard-8": 0.85}
USD_TO_GBP = 0.75

RUN_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,15}-[0-9]+-[0-9]+$")  # as fetch-results.yml
MODULE_RE = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$")
REGION_RE = re.compile(r"^[a-z]+-[a-z]+[0-9]+$")

# Create errors, by what they mean for the search. Checked in this order.
GLOBAL_QUOTA = ("GPUS_ALL_REGIONS",)  # a GPU is already in use somewhere: stop
CAPACITY = (  # no L4 free in this zone right now: next zone
    "ZONE_RESOURCE_POOL_EXHAUSTED",
    "does not have enough resources available",
    "resource pool exhausted",
    "STOCKOUT",
)
REGION_QUOTA = ("QUOTA_EXCEEDED", "Quota '")  # a regional quota: skip the rest of the region
ZONE_UNUSABLE = (  # this zone can't host it (or is briefly unwell): next zone
    "is not available in zone",
    "does not exist in zone",
    "not supported in zone",
    "ZONE_NOT_READY",
    "Internal error",
    "backendError",
    "Error 503",
)


@dataclass
class Attempt:
    zone: str
    # created | capacity | region_quota | zone_unusable | global_quota | image_missing | fatal
    outcome: str
    seconds: float
    error: str = ""


@dataclass
class Launch:
    run_id: str
    vm: str
    machine_type: str
    image: str
    started_at: str
    zone: str | None = None
    region: str | None = None
    search_seconds: float | None = None
    zones_tried: int = 0
    attempts: list[Attempt] = field(default_factory=list)
    result: str = "searching"  # created | no_capacity | refused | failed


def rate_gbp_per_hour(machine_type: str) -> float:
    return round(RATES_USD[machine_type] * USD_TO_GBP, 4)


def vm_name(run_id: str) -> str:
    return f"{NAME_PREFIX}-{run_id}"


def region_of(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def order_zones(regions: Sequence[str], l4_zones: Sequence[str]) -> list[str]:
    """Every L4 zone of each region, regions in try-list order, zones alphabetical."""
    by_region: dict[str, list[str]] = {}
    for z in l4_zones:
        by_region.setdefault(region_of(z), []).append(z)
    return [z for r in regions for z in sorted(by_region.get(r, []))]


# The image list can name an image the create call then can't find (1 Oct 2026: a just-published
# common-cu129-ubuntu-2204 image). Not fatal: the search moves on to the next-newest image.
IMAGE_MISSING_RE = re.compile(
    r"global/images/[^'\s]*'?\.? (?:was not found|cannot be found)"
    r"|referenced image resource cannot be found"
)


def classify(error: str) -> str:
    if IMAGE_MISSING_RE.search(error):
        return "image_missing"
    if any(s in error for s in GLOBAL_QUOTA):
        return "global_quota"
    if any(s in error for s in CAPACITY):
        return "capacity"
    if any(s in error for s in REGION_QUOTA):
        return "region_quota"
    if any(s in error for s in ZONE_UNUSABLE):
        return "zone_unusable"
    return "fatal"


def models_buckets(project: str, home: str, vertex_regions: Sequence[str]) -> str:
    """``region=bucket,...`` for the regions that have a weights copy (infra/main naming)."""
    pairs = [
        f"{r}={project}-models" if r == home else f"{r}={project}-models-{r}"
        for r in dict.fromkeys([home, *vertex_regions])
    ]
    return ",".join(pairs)


Run = Callable[[list[str]], subprocess.CompletedProcess]


def _gcloud(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["gcloud", *args], capture_output=True, text=True)


class Gcloud:
    def __init__(self, project: str, run: Run = _gcloud):
        self.project = project
        self.run = run

    def _ok(self, args: list[str]) -> str:
        r = self.run([*args, "--project", self.project])
        if r.returncode != 0:
            raise RuntimeError(f"gcloud {' '.join(args[:3])} failed: {r.stderr.strip()[-800:]}")
        return r.stdout

    def runner_vms(self) -> list[str]:
        out = self._ok(
            [
                "compute",
                "instances",
                "list",
                f"--filter=labels.{LABEL}",
                "--format=value(name,zone.basename(),status)",
            ]
        )
        return [line for line in out.splitlines() if line.strip()]

    def l4_zones(self) -> list[str]:
        out = self._ok(
            [
                "compute",
                "accelerator-types",
                "list",
                "--filter=name=nvidia-l4",
                "--format=value(zone.basename())",
            ]
        )
        return sorted({z.strip() for z in out.splitlines() if z.strip()})

    def candidate_images(
        self, family_re: str, server_filter: str = "family~^common-cu", limit: int = 3
    ) -> list[str]:
        """Usable images in matching families, newest first: status READY and not deprecated."""
        r = self.run(
            [
                "compute",
                "images",
                "list",
                "--project",
                IMAGE_PROJECT,
                "--no-standard-images",
                f"--filter={server_filter}",  # thousands of images otherwise; regex below
                "--format=json(name,family,creationTimestamp,status,deprecated)",
            ]
        )
        if r.returncode != 0:
            raise RuntimeError(f"can't list {IMAGE_PROJECT} images: {r.stderr.strip()[-800:]}")
        pattern = re.compile(family_re)
        images = [
            i
            for i in json.loads(r.stdout or "[]")
            if pattern.match(i.get("family") or "")
            and i.get("status", "READY") == "READY"
            and not (i.get("deprecated") or {}).get("state")
        ]
        if not images:
            raise RuntimeError(
                f"no usable {IMAGE_PROJECT} image in a family matching {family_re!r}"
            )
        images.sort(key=lambda i: i["creationTimestamp"], reverse=True)
        return [i["name"] for i in images[:limit]]

    def newest_image(self, family_re: str, server_filter: str = "family~^common-cu") -> str:
        return self.candidate_images(family_re, server_filter)[0]

    def create(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return self.run([*cmd, "--project", self.project])

    def exists(self, name: str, zone: str) -> bool:
        r = self.run(
            ["compute", "instances", "describe", name, "--zone", zone, "--format=value(name)"]
            + ["--project", self.project]
        )
        return r.returncode == 0

    def delete(self, name: str, zone: str) -> None:
        self.run(
            ["compute", "instances", "delete", name, "--zone", zone, "--quiet"]
            + ["--project", self.project]
        )


def create_cmd(
    *,
    name: str,
    zone: str,
    machine_type: str,
    image: str,
    service_account: str,
    network: str,
    run_id: str,
    metadata_files: dict[str, Path],
    disk_gb: int,
) -> list[str]:
    """The ``gcloud compute instances create`` call. Guardrails 1 (max run + DELETE) live here."""
    return [
        "compute",
        "instances",
        "create",
        name,
        "--zone",
        zone,
        "--machine-type",
        machine_type,  # g2 = 1x L4 built in, no --accelerator
        "--provisioning-model",
        "STANDARD",  # on-demand, not Spot (fewer interruptions, faster capacity)
        "--maintenance-policy",
        "TERMINATE",  # GPUs can't live-migrate
        "--no-restart-on-failure",
        "--max-run-duration",
        MAX_RUN,
        "--instance-termination-action",
        "DELETE",
        "--image",
        image,
        "--image-project",
        IMAGE_PROJECT,
        "--boot-disk-size",
        f"{disk_gb}GB",
        "--boot-disk-type",
        "pd-balanced",
        "--network",
        network,  # snapcall-runner: no firewall rules, so no inbound traffic
        "--service-account",
        service_account,
        "--scopes",
        "cloud-platform",
        "--labels",
        f"{LABEL},run_id={run_id}",
        "--metadata",
        "block-project-ssh-keys=true",
        "--metadata-from-file",
        ",".join(f"{k}={v}" for k, v in metadata_files.items()),
        "--format",
        "value(name)",
    ]


def search(
    gc: Gcloud,
    launch: Launch,
    zones: Sequence[str],
    make_cmd: Callable[[str], list[str]],
    deadline_s: float,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
    fallback_images: Sequence[str] = (),
) -> Launch:
    """Try each zone in turn until one create succeeds. Fills ``launch`` in place.

    ``make_cmd`` builds the create call for a zone from ``launch.image``. If the image turns
    out not to exist, ``launch.image`` moves to the next of ``fallback_images`` and the same
    zone is tried again."""
    start = clock()
    skip_region = None
    fallbacks = list(fallback_images)
    i = 0
    while i < len(zones):
        zone = zones[i]
        i += 1
        if region_of(zone) == skip_region:
            continue
        if clock() - start > deadline_s:
            log(f"search deadline ({deadline_s:.0f}s) reached")
            break
        t0 = clock()
        r = gc.create(make_cmd(zone))
        seconds = round(clock() - t0, 1)
        launch.zones_tried += 1
        if r.returncode == 0:
            launch.attempts.append(Attempt(zone, "created", seconds))
            launch.zone, launch.region = zone, region_of(zone)
            launch.search_seconds = round(clock() - start, 1)
            launch.result = "created"
            log(f"{zone}: created in {seconds}s (search {launch.search_seconds}s)")
            return launch
        error = r.stderr.strip()
        outcome = classify(error)
        launch.attempts.append(Attempt(zone, outcome, seconds, error[-500:]))
        log(f"{zone}: {outcome} after {seconds}s")
        if outcome == "image_missing" and fallbacks:
            log(f"image {launch.image} can't be used; trying {fallbacks[0]}")
            launch.image = fallbacks.pop(0)
            i -= 1  # same zone, next image
            continue
        if outcome == "region_quota":
            skip_region = region_of(zone)
        elif outcome in ("global_quota", "image_missing", "fatal"):
            # A fatal error might still have left a VM (e.g. the call timed out): clean up.
            if outcome == "fatal" and gc.exists(launch.vm, zone):
                log(f"{zone}: deleting the VM the failed call left behind")
                gc.delete(launch.vm, zone)
            launch.result = "failed"
            launch.search_seconds = round(clock() - start, 1)
            return launch
    launch.result = "no_capacity"
    launch.search_seconds = round(clock() - start, 1)
    return launch


def metadata(args: argparse.Namespace, tmp: Path) -> dict[str, Path]:
    """Metadata values go through files: args can hold commas, quotes and spaces."""
    values = {
        "snapcall-run-id": args.run_id,
        "snapcall-module": args.module,
        "snapcall-args": args.args,
        "snapcall-code-uri": args.code_uri,
        "snapcall-status-uri": f"{args.scratch}/runs/{args.run_id}",
        "snapcall-output-uri": f"{args.scratch}/runs/{args.run_id}/outputs",
        "snapcall-models-buckets": args.models_buckets,
        "snapcall-machine-type": args.machine_type,
        "snapcall-rate-gbp-per-hour": str(rate_gbp_per_hour(args.machine_type)),
        "snapcall-git-sha": args.git_sha,
    }
    files = {"startup-script": HERE / "startup.sh", "snapcall-vm-py": HERE / "vm.py"}
    for key, value in values.items():
        path = tmp / key
        path.write_text(value)
        files[key] = path
    return files


def validate(args: argparse.Namespace) -> list[str]:
    problems = []
    if not RUN_ID_RE.match(args.run_id):
        problems.append(f"bad run id {args.run_id!r}")
    if not MODULE_RE.match(args.module):
        problems.append(f"bad module name {args.module!r}")
    if args.machine_type not in GPU_MACHINES:
        problems.append(f"machine type must be one of {GPU_MACHINES}")
    bad = [r for r in args.regions.split() if not REGION_RE.match(r)]
    if bad or not args.regions.split():
        problems.append(f"bad or empty region list: {bad or args.regions!r}")
    if args.max_vms < 1:
        problems.append("max VMs must be at least 1")
    if args.image_family and not re.fullmatch(r"[a-z0-9-]{1,63}", args.image_family):
        problems.append(f"bad image family {args.image_family!r}")
    return problems


def main(argv: list[str] | None = None, run: Run = _gcloud) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--project", required=True)
    p.add_argument("--regions", required=True, help="space-separated try-list (RUNNER_REGIONS)")
    p.add_argument("--max-vms", type=int, default=1, help="RUNNER_MAX_VMS")
    p.add_argument("--run-id", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--args", default="")
    p.add_argument("--machine-type", default="g2-standard-4")
    p.add_argument("--scratch", required=True, help="gs://<project>-scratch")
    p.add_argument("--code-uri", required=True, help="gs:// URI of this commit's code tarball")
    p.add_argument("--models-buckets", default="", help="region=bucket,... (weights copies)")
    p.add_argument("--git-sha", default="")
    p.add_argument("--service-account", default=None, help="default rt-runner@<project>")
    p.add_argument("--network", default="snapcall-runner")
    p.add_argument("--image-family", default="", help="exact DLVM family (default: newest)")
    p.add_argument("--disk-gb", type=int, default=150)
    p.add_argument("--deadline-minutes", type=float, default=20)
    p.add_argument("--out", type=Path, default=Path("launch.json"))
    args = p.parse_args(argv)

    problems = validate(args)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2

    gc = Gcloud(args.project, run)
    existing = gc.runner_vms()
    if len(existing) >= args.max_vms:
        print(
            f"refusing: {len(existing)} runner VM(s) already exist"
            f" (RUNNER_MAX_VMS={args.max_vms}):\n"
            + "\n".join(f"  {line}" for line in existing)
            + "\nWait for it to finish (fetch-results.yml). A stopped one is deleted by the"
            " janitor within 15 minutes.",
            file=sys.stderr,
        )
        return 3

    if args.image_family:
        images = gc.candidate_images(
            f"^{re.escape(args.image_family)}$", f"family={args.image_family}"
        )
    else:
        images = gc.candidate_images(IMAGE_FAMILY_RE)
    image = images[0]
    zones = order_zones(args.regions.split(), gc.l4_zones())
    if not zones:
        print(f"no L4 zones in {args.regions!r}", file=sys.stderr)
        return 2
    launch = Launch(
        run_id=args.run_id,
        vm=vm_name(args.run_id),
        machine_type=args.machine_type,
        image=image,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    print(
        f"image {image} (fallbacks: {images[1:]}); {len(zones)} L4 zones to try: {' '.join(zones)}",
        flush=True,
    )
    sa = args.service_account or f"rt-runner@{args.project}.iam.gserviceaccount.com"

    with tempfile.TemporaryDirectory() as tmp:
        files = metadata(args, Path(tmp))

        def make_cmd(zone: str) -> list[str]:
            return create_cmd(
                name=launch.vm,
                zone=zone,
                machine_type=args.machine_type,
                image=launch.image,
                service_account=sa,
                network=args.network,
                run_id=args.run_id,
                metadata_files=files,
                disk_gb=args.disk_gb,
            )

        search(
            gc,
            launch,
            zones,
            make_cmd,
            args.deadline_minutes * 60,
            log=_flush_print,
            fallback_images=images[1:],
        )

    args.out.write_text(json.dumps(asdict(launch), indent=2) + "\n")
    if launch.result == "created":
        return 0
    last = launch.attempts[-1] if launch.attempts else None
    if last and last.outcome == "global_quota":
        print("GPUS_ALL_REGIONS is used up: another GPU is running somewhere.", file=sys.stderr)
    elif last and last.outcome in ("fatal", "image_missing"):
        print(f"create failed in {last.zone}, stopping the search:\n{last.error}", file=sys.stderr)
    else:
        print(f"no L4 free in any of {launch.zones_tried} zones tried", file=sys.stderr)
    return 1


def _flush_print(msg: str) -> None:
    print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())

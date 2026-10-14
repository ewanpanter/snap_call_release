"""GCE burst runner (ops/runner/, Decision 9): launcher search + guardrails, VM helper."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from snapcall.schema import Cost

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ops" / "runner"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


launch = _load("runner_launch", RUNNER / "launch.py")
vm = _load("runner_vm", RUNNER / "vm.py")

TRY_LIST = (
    "asia-northeast1 asia-northeast3 asia-east1 asia-southeast1 asia-south1 europe-west4 "
    "europe-west1 europe-west2 europe-west3 europe-west6 us-east4 us-east1 us-central1 "
    "us-west1 us-west4 northamerica-northeast1 northamerica-northeast2"
)
STOCKOUT = (
    "ERROR: (gcloud.compute.instances.create) Could not fetch resource:\n"
    " - The zone 'projects/p/zones/asia-northeast1-a' does not have enough resources "
    "available to fulfill the request. '(resource type:compute)'."
)
EXHAUSTED = "ERROR: code: ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS"


def done(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, out, err)


# --- zone order and error classes ---------------------------------------------------------


def test_zones_follow_the_try_list_and_finish_each_region_first():
    l4 = ["us-central1-a", "asia-northeast1-c", "asia-east1-a", "asia-northeast1-a", "x-y1-a"]
    assert launch.order_zones(["asia-northeast1", "asia-east1", "us-central1"], l4) == [
        "asia-northeast1-a",
        "asia-northeast1-c",
        "asia-east1-a",
        "us-central1-a",
    ]


def test_regions_without_l4_are_skipped():
    assert launch.order_zones(["europe-west6", "us-east4"], ["us-east4-c"]) == ["us-east4-c"]


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (STOCKOUT, "capacity"),
        (EXHAUSTED, "capacity"),
        ("Quota 'GPUS_ALL_REGIONS' exceeded.  Limit: 1.0 globally.", "global_quota"),
        ("Quota 'NVIDIA_L4_GPUS' exceeded.  Limit: 1.0 in region asia-east1.", "region_quota"),
        ("QUOTA_EXCEEDED: Quota 'CPUS' exceeded", "region_quota"),
        (
            "Invalid value for field 'resource.machineType': ... does not exist in zone",
            "zone_unusable",
        ),
        ("Internal error. Please try again", "zone_unusable"),
        ("Required 'compute.instances.create' permission for 'projects/p/...'", "fatal"),
        (
            "The resource 'projects/deeplearning-platform-release/global/images/x' was not found",
            "image_missing",
        ),
        (  # as seen on 2026-10-01
            "Invalid value for field 'resource.disks[0].initializeParams.sourceImage': "
            "'https://compute.googleapis.com/compute/v1/projects/deeplearning-platform-release/"
            "global/images/common-cu129-ubuntu-2204-nvidia-580-v20261001'. The referenced image "
            "resource cannot be found.",
            "image_missing",
        ),
    ],
)
def test_classify(error, outcome):
    assert launch.classify(error) == outcome


# --- the create call carries the guardrails -----------------------------------------------


def make_cmd(zone="asia-east1-a", tmp=Path("/tmp")):
    return launch.create_cmd(
        name="snapcall-runner-gce-1-1",
        zone=zone,
        machine_type="g2-standard-4",
        image="common-cu128-v20260901",
        service_account="rt-runner@p.iam.gserviceaccount.com",
        network="snapcall-runner",
        run_id="gce-1-1",
        metadata_files={"startup-script": tmp / "s.sh", "snapcall-args": tmp / "a"},
        disk_gb=150,
    )


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_every_vm_gets_a_2h_hard_deadline_and_is_deleted_at_it():
    cmd = make_cmd()
    assert flag(cmd, "--max-run-duration") == "2h"
    assert flag(cmd, "--instance-termination-action") == "DELETE"
    assert "--no-restart-on-failure" in cmd


def test_vm_is_labelled_named_and_isolated_as_iam_and_the_janitor_expect():
    cmd = make_cmd()
    assert cmd[3].startswith("snapcall-runner-")  # IAM name condition (infra/bootstrap)
    assert "snapcall=runner" in flag(cmd, "--labels").split(",")  # janitor, max-VMs check
    assert flag(cmd, "--network") == "snapcall-runner"
    assert flag(cmd, "--service-account").startswith("rt-runner@")
    assert flag(cmd, "--provisioning-model") == "STANDARD"
    assert flag(cmd, "--metadata") == "block-project-ssh-keys=true"


def test_vm_name_fits_the_iam_condition():
    import re

    tf = (ROOT / "infra/bootstrap/main.tf").read_text()
    assert re.search(rf'runner_prefix\s*=\s*"{launch.NAME_PREFIX}"', tf)
    name = launch.vm_name("gce-123456789012-3")
    assert re.fullmatch(r"snapcall-runner-[a-z0-9-]+", name) and len(name) <= 63


# --- the search -----------------------------------------------------------------------------


class FakeGcloud:
    """Answers create calls per zone from a script; records everything."""

    def __init__(self, results: dict[str, tuple[int, str]], leftover=False):
        self.results = results
        self.creates: list[str] = []
        self.deleted: list[str] = []
        self.leftover = leftover

    def create(self, cmd):
        zone = flag(cmd, "--zone")
        self.creates.append(zone)
        rc, err = self.results.get(zone, (0, ""))
        return done(rc, err=err)

    def exists(self, name, zone):
        return self.leftover

    def delete(self, name, zone):
        self.deleted.append(zone)


class Clock:
    def __init__(self, step=5.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def new_launch():
    return launch.Launch("gce-1-1", "snapcall-runner-gce-1-1", "g2-standard-4", "img", "now")


def run_search(results, zones, deadline=1200, **kw):
    gc = FakeGcloud(results, **kw)
    result = launch.search(
        gc, new_launch(), zones, lambda z: make_cmd(z), deadline, clock=Clock(), log=lambda m: None
    )
    return gc, result


ZONES = ["asia-northeast1-a", "asia-northeast1-b", "asia-east1-a", "europe-west4-a"]


def test_moves_on_after_stockouts_and_stops_at_the_first_success():
    gc, result = run_search(
        {"asia-northeast1-a": (1, STOCKOUT), "asia-northeast1-b": (1, EXHAUSTED)}, ZONES
    )
    assert gc.creates == ["asia-northeast1-a", "asia-northeast1-b", "asia-east1-a"]
    assert (result.result, result.zone, result.region) == ("created", "asia-east1-a", "asia-east1")
    assert [a.outcome for a in result.attempts] == ["capacity", "capacity", "created"]
    assert result.zones_tried == 3 and result.search_seconds > 0


def test_regional_quota_skips_the_rest_of_that_region():
    gc, result = run_search({"asia-northeast1-a": (1, "Quota 'NVIDIA_L4_GPUS' exceeded")}, ZONES)
    assert gc.creates == ["asia-northeast1-a", "asia-east1-a"]
    assert result.zone == "asia-east1-a"


def test_global_gpu_quota_stops_the_search_at_once():
    gc, result = run_search({"asia-northeast1-a": (1, "Quota 'GPUS_ALL_REGIONS' exceeded")}, ZONES)
    assert gc.creates == ["asia-northeast1-a"] and result.result == "failed"


def test_an_unknown_error_stops_the_search_and_cleans_up_a_leftover_vm():
    gc, result = run_search(
        {"asia-northeast1-a": (1, "PERMISSION_DENIED: nope")}, ZONES, leftover=True
    )
    assert gc.creates == ["asia-northeast1-a"] and result.result == "failed"
    assert gc.deleted == ["asia-northeast1-a"]


def test_no_capacity_anywhere():
    gc, result = run_search({z: (1, STOCKOUT) for z in ZONES}, ZONES)
    assert gc.creates == ZONES and result.result == "no_capacity" and result.zone is None


def test_search_gives_up_at_the_deadline():
    gc, result = run_search({z: (1, STOCKOUT) for z in ZONES}, ZONES, deadline=12)
    assert len(gc.creates) < len(ZONES) and result.result == "no_capacity"


# --- main(): the max-VMs refusal and the whole flow against a fake gcloud CLI ---------------


IMAGES = [
    ("old", "common-cu124-ubuntu-2204-nvidia-550", "2026-01-01"),
    ("new", "common-cu128-ubuntu-2204-nvidia-570", "2026-09-01"),
    ("tf", "tf-2-17-cu124-ubuntu-2204", "2026-09-20"),
]
IMAGE_MISSING = (
    "Invalid value for field 'resource.disks[0].initializeParams.sourceImage': 'https://compute."
    "googleapis.com/compute/v1/projects/deeplearning-platform-release/global/images/{}'. "
    "The referenced image resource cannot be found."
)


def fake_cli(existing="", create_errors=None, images=None, missing_images=()):
    """A stand-in for `gcloud` that answers by subcommand."""
    create_errors = create_errors or {}
    images = IMAGES if images is None else images
    calls = []

    def run(args):
        calls.append(args)
        if args[:3] == ["compute", "instances", "list"]:
            return done(out=existing)
        if args[:3] == ["compute", "accelerator-types", "list"]:
            return done(out="us-central1-a\nasia-northeast1-a\nasia-northeast1-c\neurope-west4-b\n")
        if args[:3] == ["compute", "images", "list"]:
            rows = [
                {"name": i[0], "family": i[1], "creationTimestamp": i[2], **(i[3] if i[3:] else {})}
                for i in images
            ]
            return done(out=json.dumps(rows))
        if args[:3] == ["compute", "instances", "create"]:
            zone = flag(args, "--zone")
            if flag(args, "--image") in missing_images:
                return done(1, err=IMAGE_MISSING.format(flag(args, "--image")))
            return done(1, err=create_errors[zone]) if zone in create_errors else done(out="vm")
        if args[:3] == ["compute", "instances", "describe"]:
            return done(1, err="not found")
        raise AssertionError(f"unexpected gcloud call {args}")

    return run, calls


def main_args(tmp_path, *extra):
    return [
        "--project", "p", "--regions", TRY_LIST, "--run-id", "gce-7-1",
        "--module", "snapcall.decide", "--args", "smoke --model {models}/Qwen/Qwen3-0.6B",
        "--scratch", "gs://p-scratch", "--code-uri", "gs://p-scratch/runs/gce-7-1/code.tar.gz",
        "--models-buckets", "us-central1=p-models", "--out", str(tmp_path / "launch.json"),
        *extra,
    ]  # fmt: skip


def test_refuses_when_a_runner_vm_already_exists(tmp_path, capsys):
    run, calls = fake_cli(existing="snapcall-runner-gce-6-1\tasia-east1-a\tRUNNING\n")
    assert launch.main(main_args(tmp_path), run=run) == 3
    assert not [c for c in calls if c[:3] == ["compute", "instances", "create"]]
    assert "RUNNER_MAX_VMS=1" in capsys.readouterr().err
    list_call = calls[0]
    assert "--filter=labels.snapcall=runner" in list_call


def test_max_vms_above_one_allows_another(tmp_path):
    run, _ = fake_cli(existing="snapcall-runner-gce-6-1\tasia-east1-a\tRUNNING\n")
    assert launch.main(main_args(tmp_path, "--max-vms", "2"), run=run) == 0


def test_main_tries_the_try_list_and_writes_launch_json(tmp_path):
    run, calls = fake_cli(create_errors={"asia-northeast1-a": STOCKOUT})
    assert launch.main(main_args(tmp_path), run=run) == 0
    creates = [c for c in calls if c[:3] == ["compute", "instances", "create"]]
    assert [flag(c, "--zone") for c in creates] == ["asia-northeast1-a", "asia-northeast1-c"]
    assert flag(creates[0], "--image") == "new"  # newest common-cu* family, not tf-*
    assert all(flag(c, "--project") == "p" for c in creates)
    record = json.loads((tmp_path / "launch.json").read_text())
    assert record["zone"] == "asia-northeast1-c" and record["result"] == "created"
    assert record["vm"] == "snapcall-runner-gce-7-1"
    assert [a["outcome"] for a in record["attempts"]] == ["capacity", "created"]


def test_metadata_carries_the_job_and_the_scripts(tmp_path):
    args = NS(
        run_id="gce-7-1",
        module="snapcall.decide",
        args='smoke --model {models}/Qwen/Qwen3-0.6B --note "a, b"',
        code_uri="gs://p-scratch/runs/gce-7-1/code.tar.gz",
        scratch="gs://p-scratch",
        models_buckets="us-central1=p-models",
        machine_type="g2-standard-8",
        git_sha="abc",
    )
    files = launch.metadata(args, tmp_path)
    assert files["startup-script"] == RUNNER / "startup.sh"
    assert files["snapcall-vm-py"] == RUNNER / "vm.py"
    assert files["snapcall-args"].read_text() == args.args  # commas survive: it's a file
    assert files["snapcall-output-uri"].read_text() == "gs://p-scratch/runs/gce-7-1/outputs"
    assert float(files["snapcall-rate-gbp-per-hour"].read_text()) == launch.rate_gbp_per_hour(
        "g2-standard-8"
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--run-id", "Bad_ID"],
        ["--module", "os; rm -rf /"],
        ["--machine-type", "a2-highgpu-1g"],
        ["--regions", "us-central1 $(evil)"],
        ["--max-vms", "0"],
        ["--image-family", "x,y=z"],
    ],
)
def test_bad_inputs_are_rejected_before_any_gcloud_call(tmp_path, extra):
    run, calls = fake_cli()
    assert launch.main(main_args(tmp_path) + extra, run=run) == 2
    assert calls == []


def test_models_buckets_follow_infra_main_naming():
    assert launch.models_buckets("p", "us-central1", ["us-central1", "europe-west4"]) == (
        "us-central1=p-models,europe-west4=p-models-europe-west4"
    )


# --- VM helper --------------------------------------------------------------------------------


def test_model_repos_and_rewrite():
    args = "smoke --model {models}/Qwen/Qwen3-0.6B --other={models}/Qwen/Qwen3-4B-Instruct-2507"
    assert vm.model_repos(args) == ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B-Instruct-2507"]
    assert vm.rewrite_args(args, "/opt/models/") == args.replace("{models}", "/opt/models")
    with pytest.raises(ValueError):
        vm.model_repos("--model {models}/../../etc")


def test_weights_come_from_the_regional_bucket_when_there_is_one(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        (tmp_path / "Qwen/Qwen3-0.6B/config.json").write_text("{}")
        return done()

    assert vm.fetch("Qwen/Qwen3-0.6B", tmp_path, "p-models", "hf", run=run) == (
        "gs://p-models/Qwen/Qwen3-0.6B"
    )
    assert calls[0][:3] == ["gcloud", "storage", "rsync"]


def test_weights_fall_back_to_hugging_face(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return done(1 if cmd[0] == "gcloud" else 0)

    assert vm.fetch("Qwen/Qwen3-0.6B", tmp_path, "p-models", "/opt/venv/bin/hf", run=run) == (
        "hf://Qwen/Qwen3-0.6B"
    )
    target = str(tmp_path / "Qwen/Qwen3-0.6B")
    assert calls[1] == ["/opt/venv/bin/hf", "download", "Qwen/Qwen3-0.6B", "--local-dir", target]
    calls.clear()
    vm.fetch("Qwen/Qwen3-0.6B", tmp_path, None, "hf", run=run)  # region without a bucket
    assert [c[0] for c in calls] == ["hf"]


def test_bucket_for_the_vms_region():
    buckets = vm.parse_buckets("us-central1=p-models,europe-west4=p-models-europe-west4")
    assert buckets.get(vm.region_of("europe-west4-b")) == "p-models-europe-west4"
    assert buckets.get(vm.region_of("asia-northeast1-a")) is None


def finish(tmp_path, exit_code=0, timings="boot=1000\nscript_start=1030\nfinish=1720\n"):
    (tmp_path / "t").write_text(timings)
    out = tmp_path / "outputs"
    rc = vm.main(
        ["finish", "--outputs", str(out), "--timings", str(tmp_path / "t"),
         "--exit-code", str(exit_code), "--run-id", "gce-7-1", "--module", "snapcall.decide",
         "--machine-type", "g2-standard-4", "--rate", "0.5325", "--zone", "asia-east1-a",
         "--vm", "snapcall-runner-gce-7-1"]
    )  # fmt: skip
    return rc, out


def test_finish_writes_a_valid_cost_json_from_boot_to_finish(tmp_path, capsys):
    rc, out = finish(tmp_path)
    assert rc == 0
    cost = Cost.model_validate_json((out / "cost.json").read_text())
    assert cost.compute[0].minutes == 12.0 and cost.compute[0].resource == "g2-standard-4"
    assert cost.job == "gce:snapcall.decide" and "asia-east1-a" in cost.notes
    done_json = json.loads(capsys.readouterr().out)
    assert done_json["exit_code"] == 0 and done_json["zone"] == "asia-east1-a"
    assert done_json["phases_seconds"] == {"boot_to_script": 30.0, "total_billed_estimate": 720.0}


def test_finish_keeps_a_cost_json_the_module_wrote(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "cost.json").write_text("module's own")
    finish(tmp_path)
    assert (out / "cost.json").read_text() == "module's own"


def test_finish_survives_missing_timings(tmp_path):
    rc, out = finish(tmp_path, exit_code=1, timings="")
    assert rc == 0
    assert Cost.model_validate_json((out / "cost.json").read_text()).total == 0


# --- startup script and workflows ---------------------------------------------------------------


def test_startup_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(RUNNER / "startup.sh")], check=True)


def test_startup_script_always_deletes_itself():
    sh = (RUNNER / "startup.sh").read_text()
    assert "trap finish EXIT" in sh and "trap on_term TERM" in sh
    finish_body = sh.split("finish() {", 1)[1].split("\n}\n", 1)[0]
    assert finish_body.rstrip().endswith("delete_self")
    assert 'gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet' in sh
    assert "poweroff" in sh  # fallback: the janitor deletes stopped runner VMs


def test_startup_script_reads_the_pins_the_dockerfile_has():
    dockerfile = (ROOT / "docker/Dockerfile.base").read_text()
    for pin in ("TORCH_VERSION", "TORCH_INDEX", "PEFT_VERSION", "ACCELERATE_VERSION"):
        assert f"ARG {pin}=" in dockerfile
    assert "COPY --from=ghcr.io/astral-sh/uv:" in dockerfile


def test_run_gpu_draft_calls_the_launcher_with_the_guardrail_variables():
    wf = yaml.safe_load((ROOT / ".github/workflows/run-gpu.yml").read_text())
    job = wf["jobs"]["launch"]
    assert job["concurrency"]["group"] == "runner-launch"
    assert job["env"]["RUNNER_MAX_VMS"] == "${{ vars.RUNNER_MAX_VMS || '1' }}"
    script = "\n".join(s.get("run", "") for s in job["steps"])
    assert "ops/runner/launch.py" in script and '--max-vms "$RUNNER_MAX_VMS"' in script
    assert "${{" not in script  # inputs reach the shell through env only


def test_fetch_results_draft_handles_runner_runs():
    wf = yaml.safe_load((ROOT / ".github/workflows/fetch-results.yml").read_text())
    assert wf["jobs"]["fetch"]["if"] == "needs.locate.outputs.kind == 'vertex'"
    assert wf["jobs"]["fetch-runner"]["if"] == "needs.locate.outputs.kind == 'runner'"
    runs = "\n".join(s.get("run", "") for s in wf["jobs"]["fetch-runner"]["steps"])
    assert "done.json" in runs and "vanished" in runs
    assert "${{" not in runs


# --- startup.sh end to end, with stub curl/gcloud/uv/nvidia-smi/hf ----------------------------

STUBS = {
    # Metadata server from files; the uv installer becomes a no-op script.
    "curl": """#!/usr/bin/env python3
import os, sys
url = sys.argv[-1]
if "metadata.google.internal" in url:
    path = os.path.join(os.environ["FAKE_MD"], url.split("/instance/", 1)[1])
    if not os.path.isfile(path):
        sys.exit(22)
    sys.stdout.write(open(path).read())
else:
    print("true")
""",
    # gs://bucket/path <-> $FAKE_GCS/bucket/path; every call is logged.
    "gcloud": """#!/usr/bin/env python3
import os, shutil, sys
args = [a for a in sys.argv[1:] if a != "-q"]
open(os.environ["FAKE_LOG"], "a").write(" ".join(args) + "\\n")
local = lambda p: os.path.join(os.environ["FAKE_GCS"], p[5:]) if p.startswith("gs://") else p
if args[:2] == ["storage", "cp"]:
    src, dst = local(args[2]), local(args[3])
    if not os.path.exists(src):
        sys.exit(1)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
elif args[:2] == ["storage", "rsync"]:
    shutil.copytree(local(args[3]), local(args[4]), dirs_exist_ok=True)
elif args[:3] == ["compute", "instances", "delete"]:
    pass
else:
    sys.exit(2)
""",
    "nvidia-smi": '#!/bin/bash\n[ -n "$FAKE_NO_GPU" ] && exit 9\necho "NVIDIA L4, 23034 MiB"\n',
    "uv": """#!/bin/bash
case "$1" in
  venv)
    d="${@: -1}"; mkdir -p "$d/bin"
    cp "$(dirname "$0")/venv-python" "$d/bin/python"
    printf '#!/bin/bash\\nmkdir -p "$4" && echo {} > "$4/config.json"\\n' > "$d/bin/hf"
    chmod +x "$d/bin/python" "$d/bin/hf" ;;
  export)
    for ((i=1; i<=$#; i++)); do [ "${!i}" = -o ] && j=$((i+1)) && touch "${!j}"; done ;;
esac
exit 0
""",
    # The venv's python: the torch GPU check passes, everything else is the real python3.
    "venv-python": """#!/bin/bash
case "$*" in *torch.cuda*) echo fake-cuda; exit 0 ;; esac
exec python3 "$@"
""",
    "poweroff": '#!/bin/bash\necho poweroff >> "$FAKE_LOG"\n',
}

FAKE_JOB = """import json, os, pathlib, sys
pathlib.Path("outputs").mkdir(exist_ok=True)
pathlib.Path("outputs/result.json").write_text(json.dumps({"argv": sys.argv[1:]}))
sys.exit(int(os.environ.get("FAKEJOB_RC", "0")))
"""


def boot_vm(tmp_path, env_extra=None, args="--model {models}/Qwen/Qwen3-0.6B --tag 'a b'"):
    import os
    import stat
    import tarfile

    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        (stubs / name).write_text(body)
        (stubs / name).chmod(stat.S_IRWXU)
    code = tmp_path / "code"
    (code / "src/fakejob").mkdir(parents=True, exist_ok=True)
    (code / "src/fakejob/__main__.py").write_text(FAKE_JOB)
    (code / "docker").mkdir(exist_ok=True)
    (code / "docker/Dockerfile.base").write_text((ROOT / "docker/Dockerfile.base").read_text())
    gcs = tmp_path / "gcs"
    (gcs / "p-scratch/runs/gce-7-1").mkdir(parents=True, exist_ok=True)
    with tarfile.open(gcs / "p-scratch/runs/gce-7-1/code.tar.gz", "w:gz") as tar:
        tar.add(code, arcname=".")
    md = tmp_path / "md"
    (md / "attributes").mkdir(parents=True, exist_ok=True)
    values = {
        "name": "snapcall-runner-gce-7-1",
        "zone": "projects/123/zones/asia-east1-a",
        "attributes/snapcall-run-id": "gce-7-1",
        "attributes/snapcall-module": "fakejob",
        "attributes/snapcall-args": args,
        "attributes/snapcall-code-uri": "gs://p-scratch/runs/gce-7-1/code.tar.gz",
        "attributes/snapcall-status-uri": "gs://p-scratch/runs/gce-7-1",
        "attributes/snapcall-output-uri": "gs://p-scratch/runs/gce-7-1/outputs",
        "attributes/snapcall-models-buckets": "us-central1=p-models",
        "attributes/snapcall-machine-type": "g2-standard-4",
        "attributes/snapcall-rate-gbp-per-hour": "0.5325",
        "attributes/snapcall-vm-py": (RUNNER / "vm.py").read_text(),
    }
    for key, value in values.items():
        (md / key).write_text(value)
    env = {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "SNAPCALL_ROOT": str(tmp_path / "root"),
        "SNAPCALL_GPU_WAIT_TRIES": "1",
        "FAKE_MD": str(md),
        "FAKE_GCS": str(gcs),
        "FAKE_LOG": str(tmp_path / "gcloud.log"),
        **(env_extra or {}),
    }
    subprocess.run(["bash", str(RUNNER / "startup.sh")], env=env, timeout=120, check=False)
    run_dir = gcs / "p-scratch/runs/gce-7-1"
    calls = (tmp_path / "gcloud.log").read_text().splitlines()
    return run_dir, calls


DELETE = "compute instances delete snapcall-runner-gce-7-1 --zone asia-east1-a --quiet"


def test_startup_runs_the_job_saves_everything_then_deletes_the_vm(tmp_path):
    run_dir, calls = boot_vm(tmp_path)
    done_json = json.loads((run_dir / "done.json").read_text())
    assert done_json["exit_code"] == 0 and done_json["zone"] == "asia-east1-a"
    result = json.loads((run_dir / "outputs/result.json").read_text())
    models = tmp_path / "root/opt/models/Qwen/Qwen3-0.6B"
    assert result["argv"] == ["--model", str(models), "--tag", "a b"]
    assert (models / "config.json").exists()  # no bucket in asia-east1 -> Hugging Face
    Cost.model_validate_json((run_dir / "outputs/cost.json").read_text())
    for name in ("_runner_log.txt", "_nvidia_smi.txt", "_runner_timings.json"):
        assert (run_dir / "outputs" / name).exists(), name
    phases = json.loads((run_dir / "outputs/_runner_timings.json").read_text())["phases_seconds"]
    assert {"install", "weights", "module"} <= set(phases)
    # Outputs first, done.json after them, delete last.
    rsync = next(i for i, c in enumerate(calls) if c.startswith("storage rsync"))
    marker = next(i for i, c in enumerate(calls) if c.endswith("/done.json"))
    assert rsync < marker < calls.index(DELETE) == len(calls) - 1


def test_a_failing_job_still_saves_and_deletes(tmp_path):
    run_dir, calls = boot_vm(tmp_path, {"FAKEJOB_RC": "3"})
    assert json.loads((run_dir / "done.json").read_text())["exit_code"] == 3
    assert (run_dir / "outputs/result.json").exists()
    assert calls[-1] == DELETE


def test_no_gpu_fails_fast_and_still_deletes(tmp_path):
    run_dir, calls = boot_vm(tmp_path, {"FAKE_NO_GPU": "1"})
    assert json.loads((run_dir / "done.json").read_text())["exit_code"] == 1
    assert not (run_dir / "outputs/result.json").exists()
    assert "no working NVIDIA driver" in (run_dir / "outputs/_runner_log.txt").read_text()
    assert calls[-1] == DELETE


def test_a_second_boot_never_reruns_the_job(tmp_path):
    boot_vm(tmp_path)
    (tmp_path / "gcloud.log").unlink()
    (tmp_path / "gcs/p-scratch/runs/gce-7-1/done.json").unlink()
    run_dir, calls = boot_vm(tmp_path)
    assert calls == [DELETE] and not (run_dir / "done.json").exists()


def test_an_explicit_image_family_is_used_as_is(tmp_path):
    run, calls = fake_cli()
    args = main_args(tmp_path, "--image-family", "common-cu124-ubuntu-2204-nvidia-550")
    assert launch.main(args, run=run) == 0
    image_list = next(c for c in calls if c[:3] == ["compute", "images", "list"])
    assert "--filter=family=common-cu124-ubuntu-2204-nvidia-550" in image_list
    create = next(c for c in calls if c[:3] == ["compute", "instances", "create"])
    assert flag(create, "--image") == "old"


def test_venv_uses_uvs_python_which_has_headers():
    # The image's python3.12 has no Python.h, and Triton compiles against it on the GPU.
    sh = (RUNNER / "startup.sh").read_text()
    assert 'UV_PYTHON_PREFERENCE=only-managed uv venv --python 3.12 "$VENV"' in sh
    assert "Python.h" in sh


def test_run_gpu_regions_input_overrides_the_try_list_for_one_run_only():
    wf = yaml.safe_load((ROOT / ".github/workflows/run-gpu.yml").read_text())
    on = wf.get("on", wf.get(True))
    for trigger in ("workflow_dispatch", "workflow_call"):
        assert on[trigger]["inputs"]["regions"]["default"] == ""
    env = wf["jobs"]["launch"]["env"]
    assert env["RUNNER_REGIONS"] == "${{ inputs.regions || vars.RUNNER_REGIONS }}"


# --- images ---------------------------------------------------------------------------------

NEWEST = ("newest", "common-cu129-ubuntu-2204-nvidia-580", "2026-10-01")


def test_a_missing_image_falls_back_to_the_next_newest_in_the_same_zone(tmp_path):
    run, calls = fake_cli(images=[*IMAGES, NEWEST], missing_images={"newest"})
    assert launch.main(main_args(tmp_path), run=run) == 0
    creates = [c for c in calls if c[:3] == ["compute", "instances", "create"]]
    assert [(flag(c, "--zone"), flag(c, "--image")) for c in creates] == [
        ("asia-northeast1-a", "newest"),
        ("asia-northeast1-a", "new"),
    ]
    record = json.loads((tmp_path / "launch.json").read_text())
    assert record["result"] == "created" and record["image"] == "new"
    assert [a["outcome"] for a in record["attempts"]] == ["image_missing", "created"]


def test_every_candidate_image_missing_stops_the_search(tmp_path, capsys):
    run, calls = fake_cli(images=[*IMAGES, NEWEST], missing_images={"newest", "new", "old"})
    assert launch.main(main_args(tmp_path), run=run) == 1
    creates = [c for c in calls if c[:3] == ["compute", "instances", "create"]]
    assert [flag(c, "--image") for c in creates] == ["newest", "new", "old"]  # one zone only
    assert "stopping the search" in capsys.readouterr().err


def test_images_that_are_not_ready_or_are_deprecated_are_never_picked(tmp_path):
    images = [
        *IMAGES,
        ("pending", "common-cu129-ubuntu-2404-nvidia-580", "2026-10-02", {"status": "PENDING"}),
        (
            "deprecated",
            "common-cu129-ubuntu-2404-nvidia-580",
            "2026-10-03",
            {"status": "READY", "deprecated": {"state": "DEPRECATED"}},
        ),
    ]
    run, calls = fake_cli(images=images)
    assert launch.main(main_args(tmp_path), run=run) == 0
    create = next(c for c in calls if c[:3] == ["compute", "instances", "create"])
    assert flag(create, "--image") == "new"

"""scripts/submit_job.py: job spec, CLI contract with vertex-job.yml, gcloud call."""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("submit_job", ROOT / "scripts" / "submit_job.py")
sj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sj)

BASE = dict(
    image="us-central1-docker.pkg.dev/p/snapcall/snapcall:abc",
    module="snapcall.decide",
    args="smoke --model '/gcs/b/Qwen/Qwen3-4B-Instruct-2507'",
    run_id="exp-123-1",
    output_uri="gs://p-scratch/runs/exp-123-1/outputs",
    service_account="rt-train@p.iam.gserviceaccount.com",
    timeout_s=7200,
    spot=True,
)
CLI = [
    "--project", "p", "--region", "us-central1", "--image", BASE["image"],
    "--module", "snapcall.decide", "--args", "smoke --model /gcs/b/m",
    "--run-id", "exp-123-1", "--output-uri", BASE["output_uri"],
    "--service-account", BASE["service_account"],
]  # fmt: skip


def env_of(spec):
    return {e["name"]: e["value"] for e in spec["workerPoolSpecs"][0]["containerSpec"]["env"]}


def test_gpu_spec_defaults_to_g2_standard_4():
    spec = sj.build_job_spec(**BASE, gpu=True, machine_type=None)
    (pool,) = spec["workerPoolSpecs"]
    assert pool["machineSpec"] == {
        "machineType": "g2-standard-4",
        "acceleratorType": "NVIDIA_L4",
        "acceleratorCount": 1,
    }
    assert pool["replicaCount"] == 1
    assert pool["containerSpec"]["args"] == [
        "snapcall.decide", "smoke", "--model", "/gcs/b/Qwen/Qwen3-4B-Instruct-2507",
    ]  # fmt: skip
    assert spec["scheduling"] == {"timeout": "7200s", "strategy": "SPOT"}
    assert spec["serviceAccount"] == BASE["service_account"]
    env = env_of(spec)
    assert env["SNAPCALL_RUN_ID"] == "exp-123-1"
    assert env["SNAPCALL_OUTPUT_URI"] == BASE["output_uri"]
    assert env["SNAPCALL_SPOT"] == "true"
    assert env["SNAPCALL_GPU"] == "true"
    assert float(env["SNAPCALL_RATE_PER_HOUR"]) == sj.rate_gbp_per_hour("g2-standard-4", True)


def test_g2_standard_8_and_on_demand():
    spec = sj.build_job_spec(**{**BASE, "spot": False}, gpu=True, machine_type="g2-standard-8")
    assert spec["workerPoolSpecs"][0]["machineSpec"]["machineType"] == "g2-standard-8"
    assert spec["scheduling"]["strategy"] == "STANDARD"
    assert env_of(spec)["SNAPCALL_SPOT"] == "false"


def test_cpu_spec_has_no_accelerator():
    spec = sj.build_job_spec(**BASE, gpu=False, machine_type=None)
    assert spec["workerPoolSpecs"][0]["machineSpec"] == {"machineType": sj.CPU_MACHINE}
    assert env_of(spec)["SNAPCALL_GPU"] == "false"


@pytest.mark.parametrize(
    ("over", "gpu", "machine"),
    [
        ({}, False, "g2-standard-4"),  # machine type without --gpu
        ({}, True, "a2-highgpu-1g"),
        ({"run_id": "Exp-1-1"}, True, None),
        ({"run_id": "exp-1"}, True, None),
        ({"module": "os; rm -rf /"}, True, None),
        ({"output_uri": "/tmp/out"}, True, None),
    ],
)
def test_bad_inputs_rejected(over, gpu, machine):
    with pytest.raises(ValueError):
        sj.build_job_spec(**{**BASE, **over}, gpu=gpu, machine_type=machine)


@pytest.mark.parametrize(("text", "s"), [("2h", 7200), ("90m", 5400), ("600s", 600), ("60", 60)])
def test_parse_timeout(text, s):
    assert sj.parse_timeout(text) == s


@pytest.mark.parametrize("text", ["3h", "121m", "0", "2d", "abc", ""])
def test_parse_timeout_rejects(text):
    with pytest.raises(ValueError):
        sj.parse_timeout(text)


def test_every_machine_has_a_rate():
    for m in (*sj.GPU_MACHINES, sj.CPU_MACHINE):
        assert sj.rate_gbp_per_hour(m, True) < sj.rate_gbp_per_hour(m, False)


def test_run_id_regex_matches_fetch_results():
    wf = (ROOT / ".github" / "workflows" / "fetch-results.yml").read_text()
    assert sj.RUN_ID_RE.pattern in wf


def test_dry_run_prints_spec(capsys):
    assert sj.main([*CLI, "--gpu", "--timeout", "2h", "--spot", "--dry-run"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["workerPoolSpecs"][0]["machineSpec"]["acceleratorType"] == "NVIDIA_L4"


def test_cli_rejects_long_timeout():
    with pytest.raises(SystemExit):
        sj.main([*CLI, "--timeout", "3h", "--dry-run"])


class FakeGcloud:
    def __init__(self, states=()):
        self.calls, self.configs, self.states = [], [], list(states)

    def __call__(self, cmd, **kw):
        assert cmd[0] == "gcloud" and kw.get("check")
        self.calls.append(cmd)
        if cmd[3] == "create":
            self.configs.append(json.loads(Path(cmd[cmd.index("--config") + 1]).read_text()))
            out = "projects/1/locations/us-central1/customJobs/42\n"
        else:
            out = self.states.pop(0) + "\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


def test_submit_returns_immediately_with_display_name_run_id(monkeypatch, capsys):
    fake = FakeGcloud()
    monkeypatch.setattr(sj.subprocess, "run", fake)
    assert sj.main([*CLI, "--gpu", "--machine-type", "g2-standard-8", "--spot"]) == 0
    assert capsys.readouterr().out.strip() == "projects/1/locations/us-central1/customJobs/42"
    (cmd,) = fake.calls  # no describe/poll: Decision 2
    assert cmd[1:4] == ["ai", "custom-jobs", "create"]
    assert cmd[cmd.index("--display-name") + 1] == "exp-123-1"
    assert cmd[cmd.index("--region") + 1] == "us-central1"
    assert fake.configs[0]["workerPoolSpecs"][0]["machineSpec"]["machineType"] == "g2-standard-8"
    assert not Path(cmd[cmd.index("--config") + 1]).exists()  # temp file cleaned up


@pytest.mark.parametrize(("final", "rc"), [("JOB_STATE_SUCCEEDED", 0), ("JOB_STATE_FAILED", 1)])
def test_wait_polls_until_terminal(monkeypatch, final, rc):
    fake = FakeGcloud(states=["JOB_STATE_PENDING", "JOB_STATE_RUNNING", final])
    monkeypatch.setattr(sj.subprocess, "run", fake)
    monkeypatch.setattr(sj.time, "sleep", lambda s: None)
    assert sj.main([*CLI, "--wait"]) == rc
    assert [c[3] for c in fake.calls] == ["create", "describe", "describe", "describe"]


def test_gcloud_error_is_reported(monkeypatch, capsys):
    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd, stderr="PERMISSION_DENIED")

    monkeypatch.setattr(sj.subprocess, "run", boom)
    assert sj.main(CLI) == 1
    assert "PERMISSION_DENIED" in capsys.readouterr().err


def test_cli_matches_vertex_job_workflow():
    """Every flag vertex-job.yml passes is one submit_job.py accepts."""
    wf = (ROOT / ".github" / "workflows" / "vertex-job.yml").read_text()
    call = wf[wf.index("name: submit Vertex job") :].split("- name:")[0]
    flags = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", call))
    assert flags == {
        "--project", "--region", "--image", "--module", "--args", "--run-id",
        "--output-uri", "--service-account", "--gpu", "--machine-type", "--timeout", "--models-dir",
        "--flex-start", "--max-wait",
    }  # fmt: skip
    for f in flags:
        assert f in (ROOT / "scripts" / "submit_job.py").read_text()


def test_models_placeholder_is_filled_from_models_dir():
    spec = sj.build_job_spec(
        **{**BASE, "args": "smoke --model {models}/Qwen/Qwen3-4B-Instruct-2507"},
        gpu=True,
        machine_type=None,
        models_dir="/gcs/p-models-europe-west4/",
    )
    args = spec["workerPoolSpecs"][0]["containerSpec"]["args"]
    assert args[1:] == [
        "smoke",
        "--model",
        "/gcs/p-models-europe-west4/Qwen/Qwen3-4B-Instruct-2507",
    ]
    assert env_of(spec)["SNAPCALL_MODELS_DIR"] == "/gcs/p-models-europe-west4"


def test_models_placeholder_without_models_dir_is_an_error():
    with pytest.raises(ValueError, match="models-dir"):
        sj.build_job_spec(**{**BASE, "args": "--model {models}/x"}, gpu=True, machine_type=None)


def test_no_models_dir_means_no_env_var():
    spec = sj.build_job_spec(**BASE, gpu=True, machine_type=None)
    assert "SNAPCALL_MODELS_DIR" not in env_of(spec)


def test_flex_start_queues_for_a_gpu():
    spec = sj.build_job_spec(
        **{**BASE, "spot": False}, gpu=True, machine_type=None, flex_start=True, max_wait_s=3600
    )
    assert spec["scheduling"] == {
        "timeout": "7200s",
        "strategy": "FLEX_START",
        "maxWaitDuration": "3600s",
    }
    assert env_of(spec)["SNAPCALL_SPOT"] == "false"


@pytest.mark.parametrize(("over", "gpu"), [({"spot": False}, False), ({"spot": True}, True)])
def test_flex_start_needs_gpu_and_excludes_spot(over, gpu):
    with pytest.raises(ValueError):
        sj.build_job_spec(**{**BASE, **over}, gpu=gpu, machine_type=None, flex_start=True)


def test_cli_max_wait_is_capped_at_24h(capsys):
    assert sj.main([*CLI, "--gpu", "--flex-start", "--max-wait", "24h", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["scheduling"]["maxWaitDuration"] == "86400s"
    with pytest.raises(SystemExit):
        sj.main([*CLI, "--gpu", "--flex-start", "--max-wait", "25h", "--dry-run"])

"""docker/entrypoint.py: run the module, write cost.json, copy outputs/ to the GCS mount."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from snapcall.schema import Cost

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("entrypoint", ROOT / "docker" / "entrypoint.py")
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)

JOB = """
import json, pathlib, sys
out = pathlib.Path("outputs")
out.mkdir(exist_ok=True)
(out / "result.json").write_text(json.dumps({"args": sys.argv[1:]}))
if "--own-cost" in sys.argv:
    (out / "cost.json").write_text("module wrote this")
sys.exit(int(sys.argv[sys.argv.index("--rc") + 1]) if "--rc" in sys.argv else 0)
"""


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    """A fake job module, a fake /gcs mount with one bucket, and the env submit_job sets."""
    (tmp_path / "fakejob.py").write_text(JOB)
    (tmp_path / "gcs" / "bucket").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    pythonpath = os.pathsep.join([str(tmp_path), str(ROOT / "src"), str(ROOT)])
    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "SNAPCALL_RUN_ID": "exp-1-1",
        "SNAPCALL_OUTPUT_URI": "gs://bucket/runs/exp-1-1/outputs",
        "SNAPCALL_GCS_MOUNT": str(tmp_path / "gcs"),
        "SNAPCALL_COMPUTE_RESOURCE": "nvidia-l4",
        "SNAPCALL_COMPUTE_COUNT": "1",
        "SNAPCALL_SPOT": "true",
        "SNAPCALL_RATE_PER_HOUR": "0.25",
    }
    return env, tmp_path / "gcs" / "bucket" / "runs" / "exp-1-1" / "outputs"


def test_gcs_to_mount():
    assert ep.gcs_to_mount("gs://b/runs/x/outputs/") == Path("/gcs/b/runs/x/outputs")
    assert ep.gcs_to_mount("gs://b/p", "/mnt") == Path("/mnt/b/p")
    for bad in ["/gcs/b/p", "s3://b/p", "gs://"]:
        with pytest.raises(ValueError):
            ep.gcs_to_mount(bad)


def test_success_writes_cost_and_uploads(job_env):
    env, dest = job_env
    assert ep.main(["fakejob", "a b", "--x"], env) == 0
    assert json.loads((dest / "result.json").read_text()) == {"args": ["a b", "--x"]}
    cost = Cost.model_validate_json((dest / "cost.json").read_text())
    assert cost.run_id == "exp-1-1" and cost.job == "vertex:fakejob"
    (c,) = cost.compute
    assert c.resource == "nvidia-l4" and c.spot and c.rate_per_hour == 0.25
    assert cost.total == pytest.approx(c.cost)


def test_failure_still_uploads_and_keeps_exit_code(job_env):
    env, dest = job_env
    assert ep.main(["fakejob", "--rc", "3"], env) == 3
    assert (dest / "result.json").exists() and (dest / "cost.json").exists()


def test_module_cost_json_is_left_alone(job_env):
    env, dest = job_env
    assert ep.main(["fakejob", "--own-cost"], env) == 0
    assert (dest / "cost.json").read_text() == "module wrote this"


def test_missing_mount_fails_the_job(job_env):
    env, _ = job_env
    env["SNAPCALL_OUTPUT_URI"] = "gs://not-mounted/runs/exp-1-1/outputs"
    assert ep.main(["fakejob"], env) == 1
    assert Path("outputs/result.json").exists()  # still there locally


def test_no_rate_means_no_compute_line(job_env):
    env, dest = job_env
    del env["SNAPCALL_RATE_PER_HOUR"]
    assert ep.main(["fakejob"], env) == 0
    cost = Cost.model_validate_json((dest / "cost.json").read_text())
    assert cost.compute == [] and cost.total == 0


def test_no_module_is_a_usage_error():
    assert ep.main([], {}) == 2


def test_gpu_smoke_command_on_tiny_model(job_env, tiny_qwen3, tmp_path):
    """The GPU smoke test's exact container command, with the tiny model."""
    env, dest = job_env
    model, tok = tiny_qwen3
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    assert ep.main(["snapcall.decide", "smoke", "--model", str(tmp_path / "model")], env) == 0
    assert json.loads((dest / "equivalence.json").read_text())["ok"]
    assert (dest / "cost.json").exists()


def test_dockerfiles():
    base = (ROOT / "docker" / "Dockerfile.base").read_text()
    train = (ROOT / "docker" / "Dockerfile.train").read_text()
    assert "--no-emit-package torch" in base and "--no-emit-package torch" in train
    assert "ARG BASE_IMAGE\nFROM ${BASE_IMAGE}" in train
    assert 'ENTRYPOINT ["python", "/app/docker/entrypoint.py"]' in train
    # The torch pin must match uv.lock, or the base drifts from what CI tests.
    lock = (ROOT / "uv.lock").read_text()
    pinned = next(ln.split("=")[1] for ln in base.splitlines() if "ARG TORCH_VERSION" in ln)
    assert f'name = "torch"\nversion = "{pinned}"' in lock
    ignore = (ROOT / "docker" / "Dockerfile.train.dockerignore").read_text().split()
    assert ignore[ignore.index("*") :][:1] == ["*"] and "!docker/entrypoint.py" in ignore


if sys.platform == "win32":  # pragma: no cover
    pytest.skip("POSIX signals", allow_module_level=True)


def test_gpu_job_without_a_visible_gpu_fails_fast_with_diagnostics(job_env, monkeypatch):
    env, dest = job_env
    env["SNAPCALL_GPU"] = "true"
    monkeypatch.setattr(ep, "GPU_CHECK", "import json; print(json.dumps({'cuda': False}))")
    assert ep.main(["fakejob"], env) == 3
    assert not (dest / "result.json").exists()  # the module never ran
    found = json.loads((dest / "gpu_check.json").read_text())
    assert {"torch_check", "env", "dev_nvidia", "usr_local_nvidia", "nvidia_smi"} <= set(found)
    assert (dest / "cost.json").exists()


def test_gpu_job_with_a_visible_gpu_runs_the_module(job_env, monkeypatch):
    env, dest = job_env
    env["SNAPCALL_GPU"] = "true"
    monkeypatch.setattr(ep, "GPU_CHECK", "import json; print(json.dumps({'cuda': True}))")
    assert ep.main(["fakejob"], env) == 0
    assert (dest / "result.json").exists() and not (dest / "gpu_check.json").exists()


def test_base_image_points_programs_at_the_mounted_driver():
    base = (ROOT / "docker" / "Dockerfile.base").read_text()
    for needle in (
        "NVIDIA_VISIBLE_DEVICES=all",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
        "/usr/local/nvidia/bin",
    ):
        assert needle in base


def test_base_image_has_a_c_compiler_for_triton():
    base = (ROOT / "docker" / "Dockerfile.base").read_text()
    assert "apt-get install -y --no-install-recommends gcc libc6-dev" in base
    assert '"gcc", "--version"' in base  # the build fails if it's missing


def test_a_vertex_restart_adds_to_the_first_attempts_cost(job_env):
    env, dest = job_env
    assert ep.main(["fakejob", "--rc", "1"], env) == 1  # attempt 1 fails...
    first = Cost.model_validate_json((dest / "cost.json").read_text())
    Path("outputs/cost.json").unlink()  # ...and Vertex restarts in a fresh container
    assert ep.main(["fakejob"], env) == 0
    both = Cost.model_validate_json((dest / "cost.json").read_text())
    assert len(both.compute) == 2 and both.compute[0] == first.compute[0]
    assert both.total == pytest.approx(sum(c.cost for c in both.compute))
    assert "1 earlier attempt" in both.notes


def test_a_bad_earlier_cost_json_is_ignored(job_env):
    env, dest = job_env
    dest.mkdir(parents=True)
    (dest / "cost.json").write_text("not json")
    assert ep.main(["fakejob"], env) == 0
    assert len(Cost.model_validate_json((dest / "cost.json").read_text()).compute) == 1

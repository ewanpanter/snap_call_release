"""Container entrypoint for Vertex jobs: ``python -m MODULE ARGS``, then save ``outputs/``.

Contract (vertex-job.yml and scripts/submit_job.py): the job runs with ``SNAPCALL_RUN_ID`` and
``SNAPCALL_OUTPUT_URI`` set, writes to local ``outputs/``, and on exit (success, failure or
SIGTERM) ``outputs/`` is copied to ``SNAPCALL_OUTPUT_URI``. Every job has a ``cost.json``:
if the module didn't write one, this writes the compute part from the container's
wall-clock and the rate ``scripts/submit_job.py`` passed in.

The copy goes through Vertex's Cloud Storage FUSE mount (``gs://B/P`` is ``/gcs/B/P``), the
same mount jobs read weights from, so the image needs no GCS client. Stdlib only, apart
from ``snapcall.schema`` to validate ``cost.json``.

On GPU jobs (``SNAPCALL_GPU=true``) it first checks that torch can see the GPU; if not, it
writes ``outputs/gpu_check.json`` with what it found (driver mount, /dev/nvidia*, env,
nvidia-smi) and fails at once instead of letting the module fall back to the CPU.

Env (set by submit_job.py): SNAPCALL_RUN_ID, SNAPCALL_OUTPUT_URI, SNAPCALL_GPU,
SNAPCALL_COMPUTE_RESOURCE, SNAPCALL_COMPUTE_COUNT, SNAPCALL_SPOT,
SNAPCALL_RATE_PER_HOUR (GBP, per unit);
SNAPCALL_GCS_MOUNT overrides ``/gcs`` (tests).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

OUTPUTS = Path("outputs")


def gcs_to_mount(uri: str, mount: str = "/gcs") -> Path:
    """``gs://bucket/some/path`` -> ``/gcs/bucket/some/path``."""
    if not uri.startswith("gs://") or len(uri) <= len("gs://"):
        raise ValueError(f"not a gs:// URI: {uri!r}")
    return Path(mount) / uri.removeprefix("gs://").strip("/")


def previous_cost(env: dict[str, str]):
    """The cost.json an earlier attempt of this run already saved, if any.

    Vertex can restart the container after a failure (same run id, same output URI); the
    restart's cost.json would otherwise replace the first attempt's.
    """
    from snapcall.schema import Cost

    uri = env.get("SNAPCALL_OUTPUT_URI")
    if not uri:
        return None
    try:
        path = gcs_to_mount(uri, env.get("SNAPCALL_GCS_MOUNT", "/gcs")) / "cost.json"
        return Cost.model_validate_json(path.read_text()) if path.is_file() else None
    except Exception as e:  # a bad earlier file mustn't stop this one being written
        print(f"[entrypoint] ignoring earlier cost.json: {e!r}", flush=True)
        return None


def write_cost(
    outputs: Path, module: str, minutes: float, env: dict[str, str], previous=None
) -> Path:
    """Write ``cost.json`` (compute only) unless the module already wrote one. Items from an
    earlier attempt of the same run (``previous``) are kept and added to the total."""
    from snapcall.schema import ComputeCost, Cost

    path = outputs / "cost.json"
    if path.exists():
        print(f"[entrypoint] {path} written by the module; leaving it", flush=True)
        return path
    compute = []
    if "SNAPCALL_RATE_PER_HOUR" in env:
        count = int(env.get("SNAPCALL_COMPUTE_COUNT", "1"))
        rate = float(env["SNAPCALL_RATE_PER_HOUR"])
        compute.append(
            ComputeCost(
                resource=env.get("SNAPCALL_COMPUTE_RESOURCE", "unknown"),
                count=count,
                minutes=round(minutes, 3),
                spot=env.get("SNAPCALL_SPOT") == "true",
                rate_per_hour=rate,
                cost=count * round(minutes, 3) / 60 * rate,
            )
        )
    notes = (
        "Written by docker/entrypoint.py: container wall-clock x estimated rate. Vertex "
        "also bills VM start-up and image pull; see _vertex_job.json startTime/endTime."
    )
    api = []
    if previous is not None:
        compute = [*previous.compute, *compute]
        api = list(previous.api)
        notes += f" Includes {len(previous.compute)} earlier attempt(s) (Vertex restart)."
    cost = Cost(
        run_id=env.get("SNAPCALL_RUN_ID", "local"),
        job=f"vertex:{module}",
        created_at=datetime.now(UTC),
        compute=compute,
        api=api,
        total=sum(c.cost for c in compute) + sum(a.cost for a in api),
        notes=notes,
    )
    path.write_text(cost.model_dump_json(indent=2) + "\n")
    return path


GPU_CHECK = (
    "import json, torch; print(json.dumps({'cuda': torch.cuda.is_available(), "
    "'torch': torch.__version__, 'built_for': torch.version.cuda, "
    "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
)


def gpu_check(outputs: Path, env: dict[str, str]) -> bool:
    """True if torch sees a GPU. Otherwise record why not in outputs/gpu_check.json."""
    r = subprocess.run([sys.executable, "-c", GPU_CHECK], capture_output=True, text=True, env=env)
    found = {"torch_check": r.stdout.strip() or r.stderr.strip()[-2000:]}
    try:
        ok = r.returncode == 0 and json.loads(r.stdout)["cuda"]
    except (ValueError, KeyError):
        ok = False
    print(f"[entrypoint] GPU check: {found['torch_check']}", flush=True)
    if ok:
        return True

    def sh(cmd: str) -> str:
        p = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True)
        return (p.stdout + p.stderr).strip()[-2000:]

    found.update(
        env={
            k: env.get(k)
            for k in (
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "PATH",
            )
        },
        dev_nvidia=sh("ls -l /dev/nvidia* 2>&1"),
        usr_local_nvidia=sh("ls -R /usr/local/nvidia 2>&1 | head -40"),
        libcuda=sh(
            "ldconfig -p 2>&1 | grep -i libcuda; find / -name 'libcuda.so*' 2>/dev/null | head"
        ),
        nvidia_smi=sh("nvidia-smi 2>&1 | head -20"),
    )
    (outputs / "gpu_check.json").write_text(json.dumps(found, indent=2) + "\n")
    print(
        "[entrypoint] ERROR: GPU job but torch sees no GPU; see outputs/gpu_check.json", flush=True
    )
    return False


def upload(outputs: Path, uri: str, mount: str) -> None:
    dest = gcs_to_mount(uri, mount)
    bucket_root = Path(mount) / dest.relative_to(mount).parts[0]
    if not bucket_root.is_dir():
        raise FileNotFoundError(f"{bucket_root} isn't mounted (no Cloud Storage FUSE?)")
    shutil.copytree(outputs, dest, dirs_exist_ok=True)
    n = sum(1 for p in outputs.rglob("*") if p.is_file())
    print(f"[entrypoint] copied {n} file(s) from {outputs}/ to {uri}", flush=True)


def main(argv: list[str], env: dict[str, str] | None = None) -> int:
    env = dict(os.environ if env is None else env)
    if not argv:
        print("usage: entrypoint.py MODULE [ARGS...]", file=sys.stderr)
        return 2
    module, *args = argv
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    print(f"[entrypoint] run {env.get('SNAPCALL_RUN_ID', 'local')}: python -m {module}", flush=True)

    start = time.monotonic()
    if env.get("SNAPCALL_GPU") == "true" and not gpu_check(OUTPUTS, env):
        return _finish(OUTPUTS, module, time.monotonic() - start, env, 3)
    child = subprocess.Popen([sys.executable, "-m", module, *args], env=env)

    # Vertex sends SIGTERM on cancel/preemption: pass it on, then still save outputs.
    def forward(signum, _frame):
        print(f"[entrypoint] got signal {signum}, passing it to the job", flush=True)
        child.send_signal(signum)

    old = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        rc = child.wait()
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    minutes = (time.monotonic() - start) / 60
    print(f"[entrypoint] python -m {module} exited {rc} after {minutes:.1f} min", flush=True)

    status = 128 - rc if rc < 0 else rc  # killed by a signal -> shell-style 128+N
    return _finish(OUTPUTS, module, minutes * 60, env, status)


def _finish(outputs: Path, module: str, seconds: float, env: dict[str, str], status: int) -> int:
    """Write cost.json if missing and copy outputs/ to SNAPCALL_OUTPUT_URI. Returns the exit
    status: the module's (signal N -> 128+N), or 1 if saving failed after a clean run."""
    try:
        write_cost(outputs, module, seconds / 60, env, previous_cost(env))
    except Exception as e:  # never lose the outputs over cost.json
        print(f"[entrypoint] couldn't write cost.json: {e!r}", flush=True)
        status = status or 1
    uri = env.get("SNAPCALL_OUTPUT_URI")
    if uri:
        try:
            upload(outputs, uri, env.get("SNAPCALL_GCS_MOUNT", "/gcs"))
        except Exception as e:
            print(f"[entrypoint] ERROR: outputs not saved to {uri}: {e!r}", flush=True)
            status = status or 1
    else:
        print("[entrypoint] SNAPCALL_OUTPUT_URI not set; outputs stay local", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

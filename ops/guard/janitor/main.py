"""Janitor (cost-control layer 3, after the GPU quota and the run-time limits).

Cloud Scheduler calls this every 15 minutes (infra/main/janitor.tf). It runs as rt-janitor and
stops anything that has been billing for longer than it should:

- Vertex custom jobs running for longer than JANITOR_MAX_JOB_MINUTES (since they entered
  RUNNING) are cancelled. Jobs that haven't started yet (queued/pending, e.g. flex-start
  waiting for a GPU) get JANITOR_MAX_QUEUED_MINUTES from creation instead: waiting for
  capacity is free.
  Caveat: Vertex also reports "pending" for the few minutes between getting a machine and
  the container starting, so a job stuck in that phase is only cancelled at the queued limit.
- Models deployed to a Vertex endpoint for longer than JANITOR_MAX_ENDPOINT_MINUTES are
  undeployed (an endpoint with nothing deployed costs nothing).
  Both are checked in every region in JANITOR_REGIONS (comma-separated; default GCP_REGION),
  since jobs can run in any of the Vertex regions.
- Compute Engine VMs older than JANITOR_MAX_VM_MINUTES are deleted, in every zone of every
  region. Burst-runner VMs (label snapcall=runner, Decision 9) get
  JANITOR_MAX_RUNNER_VM_MINUTES instead (their 2h max-run-duration + 15 min), and are
  deleted at any age once stopped (TERMINATED or SUSPENDED): a stopped VM still pays for
  its disk, and a runner never restarts.
- Cloud Run services with min-instances above 0 are set back to 0.

Every action (or would-be action, with JANITOR_DRY_RUN=true) is logged to Cloud Logging and
written as JSON to gs://$SCRATCH_BUCKET/janitor/. One failed action doesn't stop the rest.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    import functions_framework

    _http = functions_framework.http
except ImportError:  # unit tests don't need the framework

    def _http(fn):
        return fn


ACTIVE_JOB_STATES = ("JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING")
RUNNER_LABEL = ("snapcall", "runner")  # set by run-gpu.yml on every burst-runner VM
# Not STOPPING: a runner deleting itself passes through it, and a second delete would fail.
STOPPED_VM_STATES = ("TERMINATED", "SUSPENDED")
TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    project: str
    region: str
    scratch_bucket: str
    regions: tuple[str, ...]  # Vertex regions to sweep; GCP_REGION is always first
    max_job: timedelta
    max_queued: timedelta
    max_endpoint: timedelta
    max_vm: timedelta
    max_runner_vm: timedelta
    dry_run: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        env = os.environ if env is None else env
        return cls(
            project=env["GCP_PROJECT_ID"],
            region=env["GCP_REGION"],
            scratch_bucket=env["SCRATCH_BUCKET"],
            regions=tuple(
                dict.fromkeys(
                    [env["GCP_REGION"]]
                    + [r.strip() for r in env.get("JANITOR_REGIONS", "").split(",") if r.strip()]
                )
            ),
            max_job=timedelta(minutes=int(env.get("JANITOR_MAX_JOB_MINUTES", "135"))),
            max_queued=timedelta(minutes=int(env.get("JANITOR_MAX_QUEUED_MINUTES", "135"))),
            max_endpoint=timedelta(minutes=int(env.get("JANITOR_MAX_ENDPOINT_MINUTES", "90"))),
            max_vm=timedelta(minutes=int(env.get("JANITOR_MAX_VM_MINUTES", "60"))),
            max_runner_vm=timedelta(minutes=int(env.get("JANITOR_MAX_RUNNER_VM_MINUTES", "135"))),
            dry_run=env.get("JANITOR_DRY_RUN", "false").strip().lower() in TRUE,
        )


@dataclass
class Clients:
    """The GCP API clients the sweep uses. Tests pass fakes with the same methods."""

    jobs: Any  # aiplatform_v1.JobServiceClient (GCP_REGION)
    endpoints: Any  # aiplatform_v1.EndpointServiceClient (GCP_REGION)
    instances: Any  # compute_v1.InstancesClient
    services: Any  # run_v2.ServicesClient
    log_bucket: Any  # storage.Bucket for the scratch bucket
    # (jobs, endpoints) clients for another Vertex region. Vertex clients are regional.
    vertex_for: Callable[[str], tuple[Any, Any]] | None = None

    def vertex(self, region: str, home: str) -> tuple[Any, Any]:
        if region == home:
            return self.jobs, self.endpoints
        if self.vertex_for is None:
            raise ValueError(f"no Vertex clients for region {region}")
        return self.vertex_for(region)


def _name(value: Any) -> str:
    """Enum (proto-plus) or plain string -> its name."""
    return getattr(value, "name", value)


def _when(value: Any) -> datetime | None:
    """Timestamp (datetime, RFC 3339 string, or unset) -> aware datetime or None."""
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    # proto-plus can hand back the epoch for an unset timestamp.
    return None if value.year < 2000 else value


def _minutes(age: timedelta) -> int:
    return int(age.total_seconds() // 60)


def sweep(clients: Clients, cfg: Config, now: datetime) -> list[dict]:
    """Find everything over its limit and stop it. Returns one record per action."""
    actions: list[dict] = []

    def act(kind: str, target: str, age: timedelta, limit: timedelta, do: Callable[[], Any]):
        record = {
            "kind": kind,
            "target": target,
            "age_minutes": _minutes(age),
            "limit_minutes": _minutes(limit),
            "dry_run": cfg.dry_run,
            "ok": True,
        }
        if not cfg.dry_run:
            try:
                do()
            except Exception as e:  # keep going: one stuck resource mustn't shield the rest
                record.update(ok=False, error=f"{type(e).__name__}: {e}")
        actions.append(record)

    for region in cfg.regions:
        parent = f"projects/{cfg.project}/locations/{region}"
        jobs, endpoints = clients.vertex(region, cfg.region)

        # Vertex custom jobs, by state. Running: age since it entered RUNNING, which is
        # update_time (the last state change); start_time can't be used, because for
        # flex-start jobs Vertex sets it when the job is accepted, while it still waits for
        # a GPU. Queued/pending: age from creation, against the (longer) queued limit.
        # Vertex's own 2h scheduling.timeout is the primary guard on run time.
        job_filter = " OR ".join(f'state="{s}"' for s in ACTIVE_JOB_STATES)
        for job in jobs.list_custom_jobs(request={"parent": parent, "filter": job_filter}):
            if _name(job.state) not in ACTIVE_JOB_STATES:
                continue
            running = _name(job.state) == "JOB_STATE_RUNNING"
            if running:
                since = _when(getattr(job, "update_time", None)) or _when(job.start_time)
            else:
                since = _when(job.create_time)
            limit = cfg.max_job if running else cfg.max_queued
            if since and now - since > limit:
                act(
                    "cancel_vertex_job" if running else "cancel_queued_vertex_job",
                    f"{job.name} ({job.display_name})",
                    now - since,
                    limit,
                    lambda job=job, jobs=jobs: jobs.cancel_custom_job(name=job.name),
                )

        # Vertex endpoints: undeploy old models, keep the (free) endpoint.
        for endpoint in endpoints.list_endpoints(parent=parent):
            for model in endpoint.deployed_models:
                deployed = _when(model.create_time)
                if deployed and now - deployed > cfg.max_endpoint:
                    act(
                        "undeploy_vertex_model",
                        f"{endpoint.name} deployed_model={model.id}",
                        now - deployed,
                        cfg.max_endpoint,
                        lambda e=endpoint, m=model, eps=endpoints: eps.undeploy_model(
                            endpoint=e.name, deployed_model_id=m.id
                        ),
                    )

    # Compute VMs, every zone of every region (one aggregated list), so this covers every
    # region in RUNNER_REGIONS without listing them.
    for _zone_key, scoped in clients.instances.aggregated_list(request={"project": cfg.project}):
        for vm in scoped.instances or []:
            created = _when(vm.creation_timestamp)
            age = now - created if created else timedelta(0)
            runner = (getattr(vm, "labels", None) or {}).get(RUNNER_LABEL[0]) == RUNNER_LABEL[1]
            stopped = runner and _name(vm.status) in STOPPED_VM_STATES
            if stopped:
                kind, limit = "delete_stopped_runner_vm", timedelta(0)
            elif runner:
                kind, limit = "delete_runner_vm", cfg.max_runner_vm
            else:
                kind, limit = "delete_vm", cfg.max_vm
            if stopped or (created and age > limit):
                zone = vm.zone.rsplit("/", 1)[-1]
                act(
                    kind,
                    f"{zone}/{vm.name} ({_name(vm.status)})",
                    age,
                    limit,
                    lambda vm=vm, zone=zone: clients.instances.delete(
                        project=cfg.project, zone=zone, instance=vm.name
                    ),
                )

    # Cloud Run: nothing may keep a warm (GPU) instance around.
    home = f"projects/{cfg.project}/locations/{cfg.region}"
    for svc in clients.services.list_services(parent=home):
        levels = [s for s in (svc.template.scaling, getattr(svc, "scaling", None)) if s]
        if any(s.min_instance_count > 0 for s in levels):

            def to_zero(svc=svc, levels=levels):
                for s in levels:
                    s.min_instance_count = 0
                clients.services.update_service(service=svc)

            act("cloud_run_min_instances_to_0", svc.name, timedelta(0), timedelta(0), to_zero)

    return actions


def write_log(clients: Clients, cfg: Config, now: datetime, actions: list[dict]) -> str | None:
    """Write this run's actions to scratch/janitor/. Quiet runs write nothing."""
    if not actions:
        return None
    path = f"janitor/{now:%Y/%m/%d/%H%M%S}Z.json"
    body = {"time": now.isoformat(), "dry_run": cfg.dry_run, "actions": actions}
    clients.log_bucket.blob(path).upload_from_string(
        json.dumps(body, indent=2), content_type="application/json"
    )
    return path


def _log(severity: str, message: str, **fields: Any) -> None:
    """One JSON line on stdout = one structured Cloud Logging entry."""
    print(json.dumps({"severity": severity, "message": message, **fields}), flush=True)


def real_clients(cfg: Config) -> Clients:
    from google.cloud import aiplatform_v1, compute_v1, run_v2, storage

    def vertex_for(region: str):
        opts = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
        return (
            aiplatform_v1.JobServiceClient(client_options=opts),
            aiplatform_v1.EndpointServiceClient(client_options=opts),
        )

    jobs, endpoints = vertex_for(cfg.region)
    return Clients(
        jobs=jobs,
        endpoints=endpoints,
        vertex_for=vertex_for,
        instances=compute_v1.InstancesClient(),
        services=run_v2.ServicesClient(),
        log_bucket=storage.Client(project=cfg.project).bucket(cfg.scratch_bucket),
    )


def run(clients: Clients, cfg: Config, now: datetime) -> tuple[dict, int]:
    actions = sweep(clients, cfg, now)
    failed = [a for a in actions if not a["ok"]]
    summary = {"dry_run": cfg.dry_run, "actions": actions, "log": None}
    try:
        summary["log"] = write_log(clients, cfg, now, actions)
    except Exception as e:
        failed.append({"kind": "write_log", "error": f"{type(e).__name__}: {e}"})
    for a in actions:
        _log("WARNING" if a["ok"] else "ERROR", f"janitor: {a['kind']} {a['target']}", **a)
    if not actions:
        _log("INFO", "janitor: nothing over its limit", dry_run=cfg.dry_run)
    return summary, 500 if failed else 200


@_http
def run_janitor(request: Any) -> tuple[str, int, dict]:
    """HTTP entry point (Cloud Scheduler, POST, OIDC-authenticated)."""
    cfg = Config.from_env()
    summary, status = run(real_clients(cfg), cfg, datetime.now(UTC))
    return json.dumps(summary), status, {"Content-Type": "application/json"}

"""Janitor (ops/guard/janitor/main.py) against fake GCP clients."""

import importlib.util
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


janitor = _load("guard_janitor", ROOT / "ops/guard/janitor/main.py")

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
ENV = {"GCP_PROJECT_ID": "proj", "GCP_REGION": "us-central1", "SCRATCH_BUCKET": "proj-scratch"}
PARENT = "projects/proj/locations/us-central1"


def ago(minutes: float) -> datetime:
    return NOW - timedelta(minutes=minutes)


def job(name, state="JOB_STATE_RUNNING", started=None, created=None, updated=None):
    return NS(
        name=f"{PARENT}/customJobs/{name}",
        display_name=name,
        state=NS(name=state),
        start_time=started,
        create_time=created or started,
        update_time=updated or started or created,
    )


def endpoint(name, *model_ages):
    models = [NS(id=f"dm{i}", create_time=ago(m)) for i, m in enumerate(model_ages)]
    return NS(name=f"{PARENT}/endpoints/{name}", deployed_models=models)


def vm(name, minutes, zone="us-central1-a", status="RUNNING"):
    return NS(
        name=name,
        zone=f"https://www.googleapis.com/compute/v1/projects/proj/zones/{zone}",
        status=status,
        creation_timestamp=ago(minutes).isoformat(),
    )


def service(name, template_min=0, service_min=0):
    return NS(
        name=f"{PARENT}/services/{name}",
        template=NS(scaling=NS(min_instance_count=template_min)),
        scaling=NS(min_instance_count=service_min),
    )


def make_clients(jobs=(), endpoints=(), vms=(), services=()):
    c = janitor.Clients(
        jobs=MagicMock(),
        endpoints=MagicMock(),
        instances=MagicMock(),
        services=MagicMock(),
        log_bucket=MagicMock(),
    )
    c.jobs.list_custom_jobs.return_value = list(jobs)
    c.endpoints.list_endpoints.return_value = list(endpoints)
    c.instances.aggregated_list.return_value = [
        ("zones/us-central1-a", NS(instances=list(vms))),
        ("zones/europe-west4-a", NS(instances=[])),  # zones with no VMs come back empty
    ]
    c.services.list_services.return_value = list(services)
    return c


def destructive_calls(c):
    return [
        c.jobs.cancel_custom_job,
        c.endpoints.undeploy_model,
        c.instances.delete,
        c.services.update_service,
    ]


def cfg(**env):
    return janitor.Config.from_env({**ENV, **env})


# --- config ------------------------------------------------------------------------------


def test_config_defaults():
    c = cfg()
    assert (c.max_job, c.max_endpoint, c.max_vm) == (
        timedelta(minutes=135),
        timedelta(minutes=90),
        timedelta(minutes=60),
    )
    assert c.dry_run is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("1", True), ("TRUE ", True), ("false", False), ("", False), ("no", False)],
)
def test_config_dry_run_parsing(value, expected):
    assert cfg(JANITOR_DRY_RUN=value).dry_run is expected


# --- nothing to do -----------------------------------------------------------------------


def test_does_nothing_when_nothing_is_over_the_limit():
    c = make_clients(
        jobs=[job("young", started=ago(30)), job("queued", "JOB_STATE_QUEUED", created=ago(5))],
        endpoints=[endpoint("e", 30), endpoint("empty")],
        vms=[vm("fresh", 10)],
        services=[service("snapcall"), service("janitor")],
    )
    summary, status = janitor.run(c, cfg(), NOW)

    assert summary["actions"] == []
    assert status == 200
    for call in destructive_calls(c):
        call.assert_not_called()
    c.log_bucket.blob.assert_not_called()  # quiet runs write no log file


def test_does_nothing_when_the_project_is_empty():
    c = make_clients()
    summary, status = janitor.run(c, cfg(), NOW)
    assert (summary["actions"], status) == ([], 200)
    for call in destructive_calls(c):
        call.assert_not_called()


def test_exactly_at_the_limit_is_not_over_it():
    c = make_clients(jobs=[job("edge", started=ago(135))], vms=[vm("edge", 60)])
    assert janitor.sweep(c, cfg(), NOW) == []


def test_finished_jobs_are_ignored_even_if_old():
    # The real API filters by state; the fake returns everything, so the code must check too.
    c = make_clients(jobs=[job("done", "JOB_STATE_SUCCEEDED", started=ago(600))])
    assert janitor.sweep(c, cfg(), NOW) == []
    c.jobs.cancel_custom_job.assert_not_called()


# --- each kind of action -----------------------------------------------------------------


def test_lists_with_the_right_parent_and_active_state_filter():
    c = make_clients()
    janitor.sweep(c, cfg(), NOW)
    request = c.jobs.list_custom_jobs.call_args.kwargs["request"]
    assert request["parent"] == PARENT
    for state in janitor.ACTIVE_JOB_STATES:
        assert state in request["filter"]
    c.endpoints.list_endpoints.assert_called_once_with(parent=PARENT)
    c.instances.aggregated_list.assert_called_once_with(request={"project": "proj"})
    c.services.list_services.assert_called_once_with(parent=PARENT)


def test_cancels_a_job_running_too_long():
    c = make_clients(jobs=[job("runaway", started=ago(200)), job("young", started=ago(10))])
    actions = janitor.sweep(c, cfg(), NOW)

    c.jobs.cancel_custom_job.assert_called_once_with(name=f"{PARENT}/customJobs/runaway")
    assert [a["kind"] for a in actions] == ["cancel_vertex_job"]
    assert actions[0]["age_minutes"] == 200
    assert actions[0]["limit_minutes"] == 135


def test_a_job_queued_too_long_counts_from_creation():
    c = make_clients(jobs=[job("stuck", "JOB_STATE_PENDING", started=None, created=ago(300))])
    janitor.sweep(c, cfg(), NOW)
    c.jobs.cancel_custom_job.assert_called_once_with(name=f"{PARENT}/customJobs/stuck")


def test_undeploys_only_the_old_model_on_an_endpoint():
    c = make_clients(endpoints=[endpoint("e", 120, 20)])  # dm0 is 120 min old, dm1 20 min
    janitor.sweep(c, cfg(), NOW)
    c.endpoints.undeploy_model.assert_called_once_with(
        endpoint=f"{PARENT}/endpoints/e", deployed_model_id="dm0"
    )


def test_deletes_an_old_vm_in_its_own_zone():
    c = make_clients(vms=[vm("smoke", 61, zone="us-central1-a"), vm("new", 5)])
    janitor.sweep(c, cfg(), NOW)
    c.instances.delete.assert_called_once_with(
        project="proj", zone="us-central1-a", instance="smoke"
    )


@pytest.mark.parametrize(("template_min", "service_min"), [(1, 0), (0, 2), (1, 1)])
def test_forces_cloud_run_min_instances_to_zero(template_min, service_min):
    svc = service("snapcall", template_min, service_min)
    c = make_clients(services=[svc, service("janitor")])
    actions = janitor.sweep(c, cfg(), NOW)

    c.services.update_service.assert_called_once_with(service=svc)
    assert svc.template.scaling.min_instance_count == 0
    assert svc.scaling.min_instance_count == 0
    assert [a["target"] for a in actions] == [f"{PARENT}/services/snapcall"]


def test_custom_limits_from_env():
    c = make_clients(jobs=[job("j", started=ago(40))])
    janitor.sweep(c, cfg(JANITOR_MAX_JOB_MINUTES="30"), NOW)
    c.jobs.cancel_custom_job.assert_called_once()


# --- dry run, logging, failures ----------------------------------------------------------


def everything_over():
    return make_clients(
        jobs=[job("runaway", started=ago(200))],
        endpoints=[endpoint("e", 120)],
        vms=[vm("smoke", 90)],
        services=[service("snapcall", 1)],
    )


def test_dry_run_changes_nothing_but_logs_everything():
    c = everything_over()
    summary, status = janitor.run(c, cfg(JANITOR_DRY_RUN="true"), NOW)

    for call in destructive_calls(c):
        call.assert_not_called()
    assert status == 200
    assert len(summary["actions"]) == 4
    assert all(a["dry_run"] for a in summary["actions"])
    body = json.loads(c.log_bucket.blob.return_value.upload_from_string.call_args.args[0])
    assert body["dry_run"] is True and len(body["actions"]) == 4


def test_every_action_is_logged_to_scratch():
    c = everything_over()
    summary, status = janitor.run(c, cfg(), NOW)

    assert status == 200
    assert summary["log"] == "janitor/2026/09/28/120000Z.json"
    c.log_bucket.blob.assert_called_once_with("janitor/2026/09/28/120000Z.json")
    upload = c.log_bucket.blob.return_value.upload_from_string
    body = json.loads(upload.call_args.args[0])
    assert upload.call_args.kwargs["content_type"] == "application/json"
    assert sorted(a["kind"] for a in body["actions"]) == [
        "cancel_vertex_job",
        "cloud_run_min_instances_to_0",
        "delete_vm",
        "undeploy_vertex_model",
    ]
    assert all(a["ok"] for a in body["actions"])


def test_one_failure_doesnt_stop_the_rest():
    c = everything_over()
    c.jobs.cancel_custom_job.side_effect = RuntimeError("403 denied")
    summary, status = janitor.run(c, cfg(), NOW)

    assert status == 500
    c.endpoints.undeploy_model.assert_called_once()
    c.instances.delete.assert_called_once()
    c.services.update_service.assert_called_once()
    failed = [a for a in summary["actions"] if not a["ok"]]
    assert [a["kind"] for a in failed] == ["cancel_vertex_job"]
    assert "403 denied" in failed[0]["error"]


def test_a_failed_log_write_is_reported_but_actions_still_happen():
    c = everything_over()
    c.log_bucket.blob.return_value.upload_from_string.side_effect = OSError("bucket gone")
    summary, status = janitor.run(c, cfg(), NOW)
    assert status == 500
    c.jobs.cancel_custom_job.assert_called_once()
    assert summary["log"] is None


def test_entry_point_matches_terraform():
    # infra/main/janitor.tf deploys entry point "run_janitor".
    assert callable(janitor.run_janitor)


# --- more than one Vertex region --------------------------------------------------------

EU_PARENT = "projects/proj/locations/europe-west4"


def test_regions_default_to_gcp_region_and_dedupe():
    assert cfg().regions == ("us-central1",)
    c = cfg(JANITOR_REGIONS="us-central1, europe-west4,europe-west4")
    assert c.regions == ("us-central1", "europe-west4")


def test_sweeps_vertex_jobs_and_endpoints_in_every_region():
    c = make_clients(jobs=[job("home-ok", started=ago(10))])
    eu_jobs, eu_endpoints = MagicMock(), MagicMock()
    eu_old = job("eu-old", started=ago(200))
    eu_old.name = f"{EU_PARENT}/customJobs/eu-old"
    eu_jobs.list_custom_jobs.return_value = [eu_old]
    eu_endpoints.list_endpoints.return_value = [endpoint("eu-ep", 100)]
    c.vertex_for = MagicMock(return_value=(eu_jobs, eu_endpoints))

    actions = janitor.sweep(c, cfg(JANITOR_REGIONS="us-central1,europe-west4"), NOW)

    c.vertex_for.assert_called_once_with("europe-west4")
    assert eu_jobs.list_custom_jobs.call_args.kwargs["request"]["parent"] == EU_PARENT
    eu_endpoints.list_endpoints.assert_called_once_with(parent=EU_PARENT)
    eu_jobs.cancel_custom_job.assert_called_once_with(name=eu_old.name)
    eu_endpoints.undeploy_model.assert_called_once()
    c.jobs.cancel_custom_job.assert_not_called()
    assert {a["kind"] for a in actions} == {"cancel_vertex_job", "undeploy_vertex_model"}
    # VMs are listed once (all zones) and Cloud Run only in the home region.
    c.instances.aggregated_list.assert_called_once()
    c.services.list_services.assert_called_once_with(parent=PARENT)


def test_terraform_passes_every_vertex_region():
    tf = (Path(__file__).resolve().parents[1] / "infra" / "main" / "janitor.tf").read_text()
    assert "JANITOR_REGIONS" in tf and "var.vertex_regions" in tf


# --- queued jobs get their own (longer) limit -------------------------------------------


def test_queued_job_is_left_alone_until_the_queued_limit():
    c = make_clients(
        jobs=[
            job("waiting", "JOB_STATE_PENDING", started=None, created=ago(600)),
            job("running", started=ago(100), created=ago(700)),
        ]
    )
    assert janitor.sweep(c, cfg(JANITOR_MAX_QUEUED_MINUTES="1455"), NOW) == []
    c.jobs.cancel_custom_job.assert_not_called()


def test_queued_job_over_the_queued_limit_is_cancelled():
    c = make_clients(jobs=[job("forever", "JOB_STATE_QUEUED", started=None, created=ago(1500))])
    (action,) = janitor.sweep(c, cfg(JANITOR_MAX_QUEUED_MINUTES="1455"), NOW)
    assert action["kind"] == "cancel_queued_vertex_job" and action["limit_minutes"] == 1455
    c.jobs.cancel_custom_job.assert_called_once_with(name=f"{PARENT}/customJobs/forever")


def test_running_job_counts_from_start_not_creation():
    # Queued for 20h, then ran for 2h30: over the 135-min run limit, whatever the queue time.
    c = make_clients(jobs=[job("slow", started=ago(150), created=ago(1350))])
    (action,) = janitor.sweep(c, cfg(JANITOR_MAX_QUEUED_MINUTES="1455"), NOW)
    assert action["kind"] == "cancel_vertex_job" and action["age_minutes"] == 150


def test_terraform_sets_a_long_queued_limit():
    tf = (Path(__file__).resolve().parents[1] / "infra" / "main" / "janitor.tf").read_text()
    assert re.search(r'JANITOR_MAX_QUEUED_MINUTES\s*=\s*"1455"', tf)


def test_terraform_sets_the_runner_vm_limit():
    tf = (Path(__file__).resolve().parents[1] / "infra" / "main" / "janitor.tf").read_text()
    assert re.search(r'JANITOR_MAX_RUNNER_VM_MINUTES\s*=\s*"135"', tf)


# --- flex-start: Vertex sets start_time when the job is accepted, not when it runs -------
# Real record (a Vertex flex-start job): created 23:29:26.10, "started" 23:29:26.30, PENDING for
# 2h16m waiting for an L4, then cancelled by the janitor's run limit. It shouldn't have been.


def test_pending_flex_start_job_with_start_time_uses_the_queued_limit():
    c = make_clients(jobs=[job("flex", "JOB_STATE_PENDING", started=ago(136), created=ago(136))])
    assert janitor.sweep(c, cfg(JANITOR_MAX_QUEUED_MINUTES="1455"), NOW) == []
    c.jobs.cancel_custom_job.assert_not_called()


def test_running_flex_start_job_counts_from_when_it_began_running():
    # Accepted 20h ago, got a GPU 30 min ago: well inside the run limit.
    young = job("flex-young", started=ago(1200), created=ago(1200), updated=ago(30))
    # Accepted 20h ago, running for 150 min: over it.
    old = job("flex-old", started=ago(1200), created=ago(1200), updated=ago(150))
    c = make_clients(jobs=[young, old])
    (action,) = janitor.sweep(c, cfg(JANITOR_MAX_QUEUED_MINUTES="1455"), NOW)
    assert action["kind"] == "cancel_vertex_job" and action["age_minutes"] == 150
    c.jobs.cancel_custom_job.assert_called_once_with(name=f"{PARENT}/customJobs/flex-old")


# --- burst-runner VMs (Decision 9) -------------------------------------------------------


def runner_vm(name, minutes, zone="asia-northeast1-a", status="RUNNING"):
    return NS(**vars(vm(name, minutes, zone, status)), labels={"snapcall": "runner"})


def test_runner_vm_limit_defaults_to_2h15m():
    assert cfg().max_runner_vm == timedelta(minutes=135)
    assert cfg(JANITOR_MAX_RUNNER_VM_MINUTES="30").max_runner_vm == timedelta(minutes=30)


def test_runner_vm_gets_2h15m_not_the_plain_vm_limit():
    c = make_clients(vms=[runner_vm("snapcall-runner-a", 90), runner_vm("snapcall-runner-b", 136)])
    actions = janitor.sweep(c, cfg(), NOW)
    c.instances.delete.assert_called_once_with(
        project="proj", zone="asia-northeast1-a", instance="snapcall-runner-b"
    )
    assert [(a["kind"], a["limit_minutes"]) for a in actions] == [("delete_runner_vm", 135)]


def test_other_vms_keep_the_plain_limit():
    c = make_clients(vms=[vm("adhoc", 61, zone="europe-west1-b")])
    actions = janitor.sweep(c, cfg(), NOW)
    assert [a["kind"] for a in actions] == ["delete_vm"]


@pytest.mark.parametrize("status", ["TERMINATED", "SUSPENDED"])
def test_stopped_runner_vm_is_deleted_at_any_age(status):
    c = make_clients(vms=[runner_vm("snapcall-runner-a", 3, status=status)])
    actions = janitor.sweep(c, cfg(), NOW)
    c.instances.delete.assert_called_once_with(
        project="proj", zone="asia-northeast1-a", instance="snapcall-runner-a"
    )
    assert actions[0]["kind"] == "delete_stopped_runner_vm"
    assert actions[0]["target"] == f"asia-northeast1-a/snapcall-runner-a ({status})"


@pytest.mark.parametrize("status", ["STOPPING", "PROVISIONING", "STAGING", "RUNNING"])
def test_young_runner_vm_that_isnt_stopped_is_left_alone(status):
    # STOPPING: the VM is deleting itself; a second delete would only fail.
    c = make_clients(vms=[runner_vm("snapcall-runner-a", 3, status=status)])
    assert janitor.sweep(c, cfg(), NOW) == []
    c.instances.delete.assert_not_called()


def test_stopped_vm_without_the_runner_label_waits_for_the_plain_limit():
    c = make_clients(vms=[vm("other", 3, status="TERMINATED")])
    assert janitor.sweep(c, cfg(), NOW) == []


def test_runner_vms_are_found_in_every_zone_of_the_try_list():
    c = make_clients()
    c.instances.aggregated_list.return_value = [
        (f"zones/{z}", NS(instances=[runner_vm(f"snapcall-runner-{i}", 200, zone=z)]))
        for i, z in enumerate(["asia-northeast3-b", "europe-west6-c", "northamerica-northeast2-a"])
    ]
    actions = janitor.sweep(c, cfg(), NOW)
    assert [a["target"].split("/")[0] for a in actions] == [
        "asia-northeast3-b",
        "europe-west6-c",
        "northamerica-northeast2-a",
    ]


def test_dry_run_leaves_runner_vms_alone():
    c = make_clients(
        vms=[runner_vm("snapcall-runner-a", 200), runner_vm("r2", 1, status="TERMINATED")]
    )
    actions = janitor.sweep(c, cfg(JANITOR_DRY_RUN="true"), NOW)
    assert len(actions) == 2
    c.instances.delete.assert_not_called()

"""Kill switch (ops/guard/killswitch/main.py) against a fake billing client."""

import base64
import importlib.util
import json
import sys
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


ks = _load("guard_killswitch", ROOT / "ops/guard/killswitch/main.py")


def event(notification) -> NS:
    """What Eventarc hands the function for a Pub/Sub message."""
    raw = notification if isinstance(notification, bytes) else json.dumps(notification).encode()
    return NS(data={"message": {"data": base64.b64encode(raw).decode(), "attributes": {}}})


def notification(cost, budget=30.0, **extra):
    # Shape of a Cloud Billing budget notification (schemaVersion 1.0).
    return {
        "budgetDisplayName": "snapcall",
        "alertThresholdExceeded": 1.0,
        "costAmount": cost,
        "costIntervalStart": "2026-09-01T07:00:00Z",
        "budgetAmount": budget,
        "budgetAmountType": "SPECIFIED_AMOUNT",
        "currencyCode": "GBP",
        **extra,
    }


def billing(enabled=True):
    b = MagicMock()
    b.get_project_billing_info.return_value = NS(
        billing_enabled=enabled, billing_account_name="billingAccounts/000000-000000-000000"
    )
    return b


# --- decide ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("note", "act"),
    [
        (notification(15.0), False),  # 50% threshold
        (notification(29.99), False),  # 90%, just under
        (notification(30.0), True),  # exactly the budget
        (notification(42.5), True),  # over
        # Forecast-only alert: forecast says we'll exceed, but actual spend is low.
        (notification(12.0, forecastThresholdExceeded=1.0), False),
        # Forecast field present, but actual spend has reached the budget: act.
        (notification(31.0, forecastThresholdExceeded=1.0), True),
        ({"budgetDisplayName": "snapcall"}, False),  # no amounts
        (notification("lots"), False),  # not a number
        (notification(5.0, budget=0), False),  # nonsense budget
    ],
)
def test_decide(note, act):
    assert ks.decide(note).act is act


def test_forecast_alert_reason_says_forecast():
    assert "forecast" in ks.decide(notification(12.0, forecastThresholdExceeded=1.0)).reason


# --- handle ------------------------------------------------------------------------------


def test_unlinks_billing_when_actual_cost_reaches_budget(capsys):
    b = billing()
    decision = ks.handle(event(notification(30.5)), b, "proj", dry_run=False)

    assert decision.act
    b.update_project_billing_info.assert_called_once_with(
        name="projects/proj", project_billing_info={"billing_account_name": ""}
    )
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [entry["severity"] for entry in logs] == ["ALERT", "ALERT"]  # loud
    assert "BUDGET EXHAUSTED" in logs[0]["message"]


@pytest.mark.parametrize(
    "note",
    [notification(15.0), notification(12.0, forecastThresholdExceeded=1.0)],
    ids=["below-budget", "forecast-only"],
)
def test_does_nothing_below_budget(note):
    b = billing()
    assert not ks.handle(event(note), b, "proj", dry_run=False).act
    b.get_project_billing_info.assert_not_called()
    b.update_project_billing_info.assert_not_called()


@pytest.mark.parametrize("raw", [b"not json", b'{"costAmount": 99}'], ids=["garbage", "partial"])
def test_bad_messages_do_nothing(raw):
    b = billing()
    ks.handle(event(raw), b, "proj", dry_run=False)
    b.update_project_billing_info.assert_not_called()


def test_missing_payload_does_nothing():
    b = billing()
    ks.handle(NS(data={}), b, "proj", dry_run=False)
    b.update_project_billing_info.assert_not_called()


def test_already_unlinked_is_a_no_op():
    # Budgets keep notifying after the switch fires; later messages must be harmless.
    b = billing(enabled=False)
    ks.handle(event(notification(35.0)), b, "proj", dry_run=False)
    b.update_project_billing_info.assert_not_called()


def test_dry_run_decides_but_doesnt_unlink(capsys):
    b = billing()
    assert ks.handle(event(notification(35.0)), b, "proj", dry_run=True).act
    b.update_project_billing_info.assert_not_called()
    assert "DRY RUN" in capsys.readouterr().out


def test_failure_to_unlink_is_raised_and_logged(capsys):
    b = billing()
    b.update_project_billing_info.side_effect = PermissionError("403")
    with pytest.raises(PermissionError):
        ks.handle(event(notification(35.0)), b, "proj", dry_run=False)
    assert '"severity": "EMERGENCY"' in capsys.readouterr().out


def test_entry_point_matches_terraform():
    # infra/bootstrap/killswitch.tf deploys entry point "handle_budget_alert".
    assert callable(ks.handle_budget_alert)

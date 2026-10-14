"""Kill switch (cost-control layer 4, the last resort).

The project's budget publishes to the budget-alerts Pub/Sub topic; Eventarc delivers each message
here (infra/bootstrap/killswitch.tf). Runs as rt-killswitch (Project Billing Manager on this
project only).

It acts only when ACTUAL spend has reached the budget (costAmount >= budgetAmount). Forecast
alerts and lower thresholds are logged and ignored. Acting means unlinking billing from the
project, which stops everything that bills. To carry on afterwards, re-link billing in the
console (Billing > Account management).

Set KILLSWITCH_DRY_RUN=true to log the decision without unlinking.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

try:
    import functions_framework

    _cloud_event = functions_framework.cloud_event
except ImportError:  # unit tests don't need the framework

    def _cloud_event(fn):
        return fn


TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Decision:
    act: bool
    reason: str
    cost: float | None = None
    budget: float | None = None


def parse(cloud_event: Any) -> dict:
    """CloudEvent (Pub/Sub messagePublished) -> the budget notification JSON."""
    raw = base64.b64decode(cloud_event.data["message"]["data"])
    return json.loads(raw)


def decide(notification: dict) -> Decision:
    """Unlink only if actual cost has reached the budget amount."""
    try:
        cost = float(notification["costAmount"])
        budget = float(notification["budgetAmount"])
    except (KeyError, TypeError, ValueError):
        return Decision(False, "malformed notification: no numeric costAmount/budgetAmount")
    if budget <= 0:
        return Decision(False, "budgetAmount is not positive", cost, budget)
    if cost < budget:
        kind = "forecast" if "forecastThresholdExceeded" in notification else "actual"
        return Decision(
            False, f"{kind} alert below budget: {cost:.2f} of {budget:.2f}", cost, budget
        )
    return Decision(True, f"actual cost {cost:.2f} >= budget {budget:.2f}", cost, budget)


def _log(severity: str, message: str, **fields: Any) -> None:
    """One JSON line on stdout = one structured Cloud Logging entry."""
    print(json.dumps({"severity": severity, "message": message, **fields}), flush=True)


def unlink_billing(billing: Any, project: str, dry_run: bool) -> str:
    """Detach the project from its billing account. Idempotent."""
    name = f"projects/{project}"
    info = billing.get_project_billing_info(name=name)
    if not info.billing_enabled:
        return "billing already disabled"
    if dry_run:
        return f"DRY RUN: would unlink {info.billing_account_name}"
    billing.update_project_billing_info(
        name=name, project_billing_info={"billing_account_name": ""}
    )
    return f"unlinked {info.billing_account_name}"


def handle(cloud_event: Any, billing: Any, project: str, dry_run: bool) -> Decision:
    try:
        notification = parse(cloud_event)
    except Exception as e:
        _log("ERROR", "killswitch: couldn't parse the budget message", error=str(e))
        return Decision(False, "unparseable message")

    decision = decide(notification)
    fields = {
        "budget": notification.get("budgetDisplayName"),
        "cost": decision.cost,
        "budget_amount": decision.budget,
        "currency": notification.get("currencyCode"),
        "dry_run": dry_run,
    }
    if not decision.act:
        _log("INFO", f"killswitch: no action ({decision.reason})", **fields)
        return decision

    _log("ALERT", f"killswitch: BUDGET EXHAUSTED ({decision.reason}), unlinking billing", **fields)
    try:
        result = unlink_billing(billing, project, dry_run)
    except Exception as e:
        _log("EMERGENCY", "killswitch: FAILED to unlink billing", error=repr(e), **fields)
        raise
    _log("ALERT", f"killswitch: {result} for {project}", **fields)
    return decision


@_cloud_event
def handle_budget_alert(cloud_event: Any) -> None:
    """CloudEvent entry point (Eventarc, Pub/Sub topic budget-alerts)."""
    from google.cloud import billing_v1

    handle(
        cloud_event,
        billing_v1.CloudBillingClient(),
        os.environ["GCP_PROJECT_ID"],
        os.environ.get("KILLSWITCH_DRY_RUN", "false").strip().lower() in TRUE,
    )

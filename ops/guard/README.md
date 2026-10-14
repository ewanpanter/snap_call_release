# ops/guard: janitor and kill switch

The two automatic cost guards. Each folder is one Cloud Run function (Python
3.12), zipped and deployed by Terraform. The folder layout and entry-point names are what the
Terraform expects, so don't rename them.

| | Janitor | Kill switch |
|---|---|---|
| Code | `janitor/main.py`, entry `run_janitor` (HTTP) | `killswitch/main.py`, entry `handle_budget_alert` (CloudEvent) |
| Triggered by | Cloud Scheduler, every 15 min | Budget → Pub/Sub `budget-alerts` → Eventarc |
| Runs as | `rt-janitor` | `rt-killswitch` (Project Billing Manager) |
| Deployed by | `infra/main` (CI apply on merge) | `infra/bootstrap` (you, from Cloud Shell) |
| Dry run | `JANITOR_DRY_RUN=true` (Terraform variable `janitor_dry_run`) | `KILLSWITCH_DRY_RUN=true` (not set by Terraform: live by default) |
| Tests | `tests/test_janitor.py` | `tests/test_killswitch.py` |

## Janitor

Every run it:

- cancels Vertex custom jobs active for longer than `JANITOR_MAX_JOB_MINUTES` (135),
  counted from start, or from creation while still queued;
- undeploys models that have sat on a Vertex endpoint longer than
  `JANITOR_MAX_ENDPOINT_MINUTES` (90). The empty endpoint is kept, because it costs nothing;
- deletes Compute Engine VMs older than `JANITOR_MAX_VM_MINUTES` (60), in any zone;
- burst-runner VMs (label `snapcall=runner`, Decision 9) instead get
  `JANITOR_MAX_RUNNER_VM_MINUTES` (135: their 2h `max-run-duration` + 15 min), and are
  deleted at any age once stopped (`TERMINATED` or `SUSPENDED`). The VM list is one
  aggregated call across every zone, so every region in `RUNNER_REGIONS` is covered
  without listing them;
- sets min-instances back to 0 on any Cloud Run service in the region.

Each action goes to Cloud Logging as a WARNING line (ERROR if it failed) and to
`gs://<project>-scratch/janitor/YYYY/MM/DD/HHMMSSZ.json`. Runs with nothing to do write
only an INFO line. If an action fails, the others still run, and the response is HTTP 500,
so the failure shows up in Scheduler.

Known limit: undeploying the only model with traffic is fine, but if an endpoint has several
deployed models, Vertex may refuse to undeploy one that still takes traffic. The failure is
logged, and the next run tries again.

## Kill switch

For each budget message it:

- acts only if **actual** spend has reached the budget (`costAmount >= budgetAmount`);
- ignores forecast alerts, the 50%/90% alerts and malformed messages (logged at INFO or ERROR);
- does nothing if billing is already unlinked (budgets keep sending messages after it fires).

When it acts, it unlinks billing from the project, logging at ALERT, or at EMERGENCY if the
unlink fails. To carry on afterwards, re-link billing in the console
(Billing → Account management).

## Checking locally

```
uv run pytest -q tests/test_janitor.py tests/test_killswitch.py
```

The tests use fake clients, so they need no GCP libraries and no credentials. The
`requirements.txt` in each folder is only installed by Cloud Build at deploy time.

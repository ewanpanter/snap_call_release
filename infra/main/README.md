# infra/main: everything that isn't identity

Managed by CI, never by hand:

- **Push to any branch touching `infra/main/`** → `infra-plan.yml` (Experiment tier) runs
  `terraform plan -lock=false` and posts it as a comment on the branch's open PR.
- **Merge to `main`** → `infra-apply.yml` (Release tier, only when `RELEASE_ACTOR` is the
  actor) runs `terraform apply`.
- **Teardown** → `infra-destroy.yml`, started by `RELEASE_ACTOR` with
  `confirm = "destroy snapcall"`.

Prerequisite: `infra/bootstrap` has been applied (it creates the state bucket, the service
accounts and every permission this folder relies on), the state has been seeded, and the repo
Variables are set.

## What's here

| Resource | Name | Notes |
|---|---|---|
| Scratch bucket | `<project>-scratch` | Data, checkpoints, run outputs, `janitor/` logs. Objects deleted after 30 days, no soft-delete |
| Models bucket | `<project>-models` | Base weights at `<HF repo id>/`, written only by `fetch-weights.yml` (Release tier) |
| Regional models buckets | `<project>-models-<region>` | One per extra region in `vertex_regions` |
| Artifact Registry | `<region>-docker.pkg.dev/<project>/snapcall` | Images for the legacy Vertex path. Keeps the 5 newest of each; deletes others after 14 days |
| Secret slot | `typesafe-api-key` | **No value**: you paste the Jev key in the console. Only `ci-release` can read it |
| Janitor | function `janitor` + Scheduler job `janitor` (every 15 min) | Runs as `rt-janitor`; code in `ops/guard/janitor/` |

The state is in `gs://<project>-tfstate/infra/main/`. `project_id` and `region` come from the
`GCP_PROJECT_ID` / `GCP_REGION` repo variables as `TF_VAR_*`.

The janitor gets these env vars: `GCP_PROJECT_ID`, `GCP_REGION`, `JANITOR_REGIONS`,
`SCRATCH_BUCKET`, the limits `JANITOR_MAX_JOB_MINUTES` (135),
`JANITOR_MAX_QUEUED_MINUTES` (1455), `JANITOR_MAX_ENDPOINT_MINUTES` (90),
`JANITOR_MAX_VM_MINUTES` (60) and `JANITOR_MAX_RUNNER_VM_MINUTES` (135), and
`JANITOR_DRY_RUN` (`true` = only log what it would do; set by the `janitor_dry_run` variable,
default `false`). Its permissions are in `infra/bootstrap/iam.tf` (`rt-janitor`).

Anything new that needs a permission goes in `infra/bootstrap/iam.tf`, and you re-apply
bootstrap from Cloud Shell. CI can't grant permissions.

## Checking locally

```
terraform fmt -check && terraform init -backend=false && terraform validate
```

There's no local `plan` (no GCP credentials needed, by design); read the plan comment on the PR.

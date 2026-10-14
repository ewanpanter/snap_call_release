# infra/bootstrap: the one-off setup you run from Cloud Shell

This creates the pieces that let GitHub Actions talk to GCP **without any keys**, and every
permission in the project. CI can never change these; only you can, from Cloud Shell. You run
it once (about 10 minutes), and again only when a permission needs changing.

Everything below happens in your browser. You don't need anything installed locally.

## Before you start

- You have a new GCP project with billing linked and a budget set (top-level README, step 1).
- Your copy of this repo is on GitHub, and you know:
  - your repo's numeric id: `gh api repos/<owner>/<repo> --jq .id`
  - your GitHub user id: `gh api users/<owner> --jq .id`
  - optionally, the user id of a second account (for example a bot that pushes branches) that
    may run Experiment-tier workflows.

Copy each grey box exactly, paste it into Cloud Shell and press Enter.

## Step by step

### 1. Open Cloud Shell

Open <https://console.cloud.google.com/?cloudshell=true> while logged in as yourself, and pick
your project. If a box says **Authorize Cloud Shell**, click **Authorize**.

### 2. Install Terraform

Cloud Shell may not come with Terraform. Install it from HashiCorp's package repository
([instructions](https://developer.hashicorp.com/terraform/install)):

```
wget -qO - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor --yes -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(grep -oP '(?<=UBUNTU_CODENAME=).*' /etc/os-release || lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt-get update && sudo apt-get install -y terraform
hash -r
terraform version
```

The last line should print `Terraform v1.5` or newer. Cloud Shell keeps only your home folder,
so to keep Terraform between sessions, put the same install in `~/.customize_environment`
(it runs as root when a session starts):

```
cat > ~/.customize_environment <<'SCRIPT'
#!/bin/sh
wget -qO - https://apt.releases.hashicorp.com/gpg | gpg --batch --yes --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(grep -oP '(?<=UBUNTU_CODENAME=).*' /etc/os-release || lsb_release -cs) main" > /etc/apt/sources.list.d/hashicorp.list
apt-get update && apt-get install -y terraform
SCRIPT
```

### 3. Get the repo into Cloud Shell

```
git clone https://github.com/<owner>/<repo>.git ~/snapcall
```

If the repo is private, use a fine-grained personal access token with **Contents: Read-only**
on that one repo and a short expiry as the password, and delete it afterwards.

### 4. Fill in your identifiers

```
cd ~/snapcall/infra/bootstrap
gcloud config set project <your-project-id>
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars
```

Fill in `project_id`, `github_repository_id`, `github_owner_id` and, if you have one,
`github_bot_id`. These are identifiers, not secrets, but they're yours: `terraform.tfvars` is
in `.gitignore`, so don't force-add it.

### 5. Switch on the APIs Terraform needs, then apply

```
gcloud services enable cloudresourcemanager.googleapis.com serviceusage.googleapis.com iam.googleapis.com
terraform init
terraform plan
terraform apply
```

Read the plan: on a first run it should only add things (`0 to change, 0 to destroy`). Type
`yes` at the prompt. If it stops with an error, it's almost always Google still switching
APIs on: wait two minutes and run `terraform apply` again.

### 6. Create the empty state file for `infra/main`

CI's `terraform plan` runs read-only, and Terraform can't start on a brand-new state without
writing an empty one first. Do that write yourself, once:

```
PROJECT=$(gcloud config get-value project)
mkdir -p ~/tfstate-seed && cd ~/tfstate-seed
printf 'terraform {\n  backend "gcs" {\n    bucket = "%s-tfstate"\n    prefix = "infra/main"\n  }\n}\n' "$PROJECT" > main.tf
terraform init
cd ~/snapcall/infra/bootstrap
```

### 7. Connect the budget to the kill-switch topic

In the console: **Billing** → **Budgets & alerts** → your budget → **Edit** → tick **Connect a
Pub/Sub topic to this budget** → your project, topic `budget-alerts` → **Save**.

### 8. Give GitHub the identifiers

```
terraform output github_variables
terraform output runner
```

Add each value as a repository **Variable** (Settings → Secrets and variables → Actions →
Variables), plus `RELEASE_ACTOR`, `RUNNER_REGIONS` and `RUNNER_MAX_VMS`: see the table in the
top-level README.

### 9. Keep a backup of the state file

Terraform's record of what it made is `terraform.tfstate` in this folder. Keep a copy, and
never commit it or paste it anywhere:

```
cp terraform.tfstate ~/snapcall-bootstrap.tfstate.backup
```

## Re-running bootstrap

When a permission needs changing, pull the change and apply again. Read the plan before typing
`yes`: it should only add or change what you expect.

```
cd ~/snapcall && git pull && cd infra/bootstrap
hash -r && terraform apply
```

## What this creates

| Thing | Why |
|---|---|
| The APIs the project uses | Vertex AI, Compute Engine, Cloud Run, Artifact Registry, ... |
| Bucket `<project>-tfstate` | Terraform state for `infra/main` (versioned; CI can't delete or reconfigure it) |
| Bucket `<project>-functions` | Zipped code for the janitor and kill-switch functions |
| Pub/Sub topic `budget-alerts` | Budget → kill switch |
| Workload Identity pool `github`, provider `github-actions` | GitHub's signed run token → a one-hour GCP token, no keys |
| 7 service accounts | `ci-experiment`, `ci-release`, `rt-train` (Vertex jobs), `rt-runner` (burst-runner VMs), `rt-janitor`, `rt-killswitch`, `fn-build` (builds the two functions) |
| 4 custom roles | `snapcallStorageManager`, `snapcallDeployer`: what `ci-release` needs to apply `infra/main`, with no permission to grant permissions. `snapcallRunnerLauncher`, `snapcallRunnerSelfDelete`: create and delete burst-runner VMs |
| VPC `snapcall-runner` | Burst-runner VMs. No firewall rules, so nothing can connect in; outbound is allowed |
| All role bindings | Below |
| Kill-switch function | Unlinks billing when actual spend reaches the budget |

### Who can do what

GitHub's run token is only accepted if it comes from your repo (by numeric id, so a renamed or
recreated repo can't inherit the trust) **and** the run was started by you or the optional
second account (by numeric user id). Nobody else gets a GCP token at all.

| Tier | Token must say | Becomes | Can | Can't |
|---|---|---|---|---|
| Experiment | any branch | `ci-experiment` | read-only view of the project, `terraform plan`, push images, run Vertex jobs as `rt-train`, create and delete burst-runner VMs (named `snapcall-runner-*`) that run as `rt-runner`, call Gemini, read and write `scratch`, read `models` | read any secret value, write `models`, apply, deploy, touch IAM |
| Release | `refs/heads/main` **and** your user id | `ci-release` | read-only view of the project, apply `infra/main`, read and write buckets, read the vendor key, deploy the janitor | touch IAM, service accounts, the WIF pool, the state bucket's settings, or the kill switch |

The workflows check the same thing a second time with the `RELEASE_ACTOR` repo variable.

**Production note:** `ci-experiment` gets the basic `roles/viewer` so that `terraform plan` can
refresh every resource. That's fine for a small project, but it reads far more than it needs.
A production setup would grant a hand-picked set of viewer roles instead.

The **kill switch** (Project Billing Manager on this project) is deployed from here, not by CI,
so no workflow can run code as it.

## Troubleshooting

| You see | Do this |
|---|---|
| `SERVICE_DISABLED`, `has not been used in project`, `API ... is not enabled` | Wait 2 minutes, `terraform apply` again. |
| `Error 403` / `Permission ... denied` on the first run | Permissions take a minute or two to spread. Re-run. |
| `Error acquiring the state lock` | A previous run was cut off. Run the `terraform force-unlock <ID>` it prints, then apply again. |
| `Billing must be enabled` | Link billing to the project. If the kill switch fired, re-link it in **Billing** → **Account management**. |
| `already exists` for pool `github` or role `snapcall...` after a `destroy` | Google keeps deleted ones for up to 30 days. Undelete them (`gcloud iam workload-identity-pools undelete github --location=global`, `gcloud iam roles undelete snapcallDeployer --project=<project>`), then `terraform import` them. |
| `terraform` only prints install instructions | Run `hash -r`, or redo step 2. |

## Teardown

First destroy `infra/main` with the `infra-destroy` workflow. Then, in Cloud Shell:

```
cd ~/snapcall/infra/bootstrap
gcloud storage rm --recursive gs://<project>-tfstate
terraform destroy
```

Google keeps deleted service accounts, custom roles and the WIF pool for up to 30 days, so
re-creating them with the same names within that window needs the `undelete` commands above.
Shutting down the project removes everything.

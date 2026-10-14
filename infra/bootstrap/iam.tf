# Every service account and every role binding lives here, applied only by you from Cloud
# Shell. CI can never grant permissions: neither CI account holds any *.setIamPolicy
# permission.
#
# Buckets, the Artifact Registry repo and the secret slot are created
# later by infra/main, so bindings on them are project-level bindings with an IAM condition
# on the resource name. That scopes them to one resource even though it doesn't exist yet.
# Where a service's resource-name format is ambiguous (project id vs number), the condition
# accepts both.

resource "google_service_account" "sa" {
  for_each = {
    "ci-experiment" = "CI, Experiment tier: any branch, owner or second account"
    "ci-release"    = "CI, Release tier: main, triggered by the owner"
    "rt-train"      = "Runtime: Vertex custom jobs (train, eval, experiments)"
    "rt-runner"     = "Runtime: GCE burst-runner GPU VMs (Decision 9)"
    "rt-janitor"    = "Runtime: janitor (cancels over-long GPU resources)"
    "rt-killswitch" = "Runtime: kill switch (unlinks billing at 100% of budget)"
    "fn-build"      = "Builds the janitor and kill-switch functions (Cloud Build)"
  }

  account_id   = each.key
  display_name = each.key
  description  = each.value

  depends_on = [google_project_service.apis]
}

locals {
  sa = { for k, v in google_service_account.sa : k => "serviceAccount:${v.email}" }

  # Built from known strings (not resource attributes) so for_each keys are known at plan.
  role_deployer        = "projects/${var.project_id}/roles/snapcallDeployer"
  role_storage_manager = "projects/${var.project_id}/roles/snapcallStorageManager"
  role_runner_launcher = "projects/${var.project_id}/roles/snapcallRunnerLauncher"
  role_runner_self     = "projects/${var.project_id}/roles/snapcallRunnerSelfDelete"

  # CEL snippets: "this bucket or anything in it", etc.
  cond_bucket = { for name in concat([local.scratch_bucket, local.models_bucket, local.tfstate_bucket, local.functions_bucket], values(local.regional_models_buckets)) :
    name => "resource.name == 'projects/_/buckets/${name}' || resource.name.startsWith('projects/_/buckets/${name}/')"
  }
  cond_ar_repo = join(" || ", flatten([for p in [var.project_id, local.project_number] : [
    "resource.name == 'projects/${p}/locations/${var.region}/repositories/${local.ar_repository}'",
    "resource.name.startsWith('projects/${p}/locations/${var.region}/repositories/${local.ar_repository}/')",
  ]]))
  cond_secret = join(" || ", flatten([for p in [var.project_id, local.project_number] : [
    "resource.name == 'projects/${p}/secrets/${var.typesafe_secret_id}'",
    "resource.name.startsWith('projects/${p}/secrets/${var.typesafe_secret_id}/')",
  ]]))
  # Runner VMs and their boot disks, by name (runner.tf). Any other resource type the role
  # touches (subnetworks, operations) passes. Labels can't be checked by IAM at create time,
  # so the name prefix is the fence; the snapcall=runner label is for the janitor.
  # IAM Conditions have no regex matches(); extract() returns "" when the template doesn't fit.
  cond_runner = join(" || ", [
    "(resource.type != 'compute.googleapis.com/Instance' && resource.type != 'compute.googleapis.com/Disk')",
    "resource.name.extract('/instances/{name}').startsWith('${local.runner_prefix}-')",
    "resource.name.extract('/disks/{name}').startsWith('${local.runner_prefix}-')",
  ])

  # [service account, role, condition title, condition expression or null]
  project_bindings = [
    # --- ci-experiment: build/push images, submit Vertex jobs, Gemini, scratch rw, models ro,
    #     read-only terraform plan. No secrets, no models writes, no apply, no IAM.
    # roles/viewer is broad (read-only, no secret values) but makes `terraform plan` refresh
    # reliably. Fine for a toy; production would use a hand-picked set of *.viewer roles.
    ["ci-experiment", "roles/viewer", null, null],
    ["ci-experiment", "roles/storage.bucketViewer", null, null],
    ["ci-experiment", "roles/aiplatform.user", null, null],
    ["ci-experiment", "roles/storage.objectUser", "scratch", local.cond_bucket[local.scratch_bucket]],
    ["ci-experiment", "roles/storage.objectViewer", "models", local.cond_bucket[local.models_bucket]],
    ["ci-experiment", "roles/storage.objectViewer", "tfstate", local.cond_bucket[local.tfstate_bucket]],
    ["ci-experiment", "roles/storage.objectViewer", "functions", local.cond_bucket[local.functions_bucket]],
    # No condition: Artifact Registry isn't one of the services whose resource.name IAM
    # Conditions can see (iam/docs/conditions-resource-attributes), so cond_ar_repo never
    # matched and docker push was denied. The project has one repo, so this is the same scope.
    ["ci-experiment", "roles/artifactregistry.writer", null, null],
    # Create (and delete) burst-runner VMs in any zone, only under the runner name prefix.
    ["ci-experiment", local.role_runner_launcher, "runner-vms", local.cond_runner],

    # --- ci-release: everything infra/main needs to apply, the vendor key, models writes.
    #     Custom roles instead of *.admin, which would include setIamPolicy.
    ["ci-release", "roles/viewer", null, null],
    ["ci-release", "roles/storage.bucketViewer", null, null],
    ["ci-release", local.role_deployer, null, null],
    # Buckets and objects anywhere except the state bucket itself (can't delete or reconfigure it)...
    ["ci-release", local.role_storage_manager, "not-tfstate-bucket",
    "!(${local.cond_bucket[local.tfstate_bucket]})"],
    # ...but it can read and write the state objects.
    ["ci-release", "roles/storage.objectUser", "tfstate", local.cond_bucket[local.tfstate_bucket]],
    ["ci-release", "roles/secretmanager.secretAccessor", "vendor-key", local.cond_secret],

    # --- rt-train: what a Vertex job runs as.
    ["rt-train", "roles/storage.objectUser", "scratch", local.cond_bucket[local.scratch_bucket]],
    ["rt-train", "roles/storage.objectViewer", "models", local.cond_bucket[local.models_bucket]],
    ["rt-train", "roles/artifactregistry.reader", "ar-snapcall", local.cond_ar_repo],
    ["rt-train", "roles/logging.logWriter", null, null],

    # --- rt-runner: what a burst-runner VM runs as. Code in and outputs out via scratch,
    #     weights from a models bucket, and delete itself. It can delete any runner VM, not
    #     only its own: IAM can't say "self". Worst case it deletes a sibling run.
    ["rt-runner", "roles/storage.objectUser", "scratch", local.cond_bucket[local.scratch_bucket]],
    ["rt-runner", "roles/storage.objectViewer", "models", local.cond_bucket[local.models_bucket]],
    ["rt-runner", "roles/logging.logWriter", null, null],
    ["rt-runner", local.role_runner_self, "runner-vms", local.cond_runner],

    # --- rt-janitor: cancel/undeploy/delete over-long resources, force Cloud Run min=0, log
    #     to scratch/janitor/. Invoked by Cloud Scheduler as itself.
    ["rt-janitor", "roles/aiplatform.user", null, null],
    ["rt-janitor", "roles/compute.instanceAdmin.v1", null, null],
    ["rt-janitor", "roles/run.developer", null, null],
    ["rt-janitor", "roles/run.invoker", null, null],
    ["rt-janitor", "roles/storage.objectUser", "scratch", local.cond_bucket[local.scratch_bucket]],
    ["rt-janitor", "roles/logging.logWriter", null, null],

    # --- rt-killswitch: unlink billing from this project only. Invoked by Eventarc as itself.
    ["rt-killswitch", "roles/billing.projectManager", null, null],
    ["rt-killswitch", "roles/run.invoker", null, null],
    ["rt-killswitch", "roles/eventarc.eventReceiver", null, null],
    ["rt-killswitch", "roles/logging.logWriter", null, null],

    # --- fn-build: Cloud Build identity for the two functions.
    ["fn-build", "roles/cloudbuild.builds.builder", null, null],
  ]
}

# Read access to the weights copies in the other Vertex regions, same as the home "models"
# bucket: jobs (rt-train) read weights in their own region. ci-release's storage role
# already covers writing them (fetch-weights.yml).
locals {
  # Two plain for-expressions, not flatten(): flatten() would also flatten each
  # [account, role, title, condition] tuple into loose strings.
  regional_models_bindings = concat(
    [for r, bucket in local.regional_models_buckets :
    ["ci-experiment", "roles/storage.objectViewer", "models-${r}", local.cond_bucket[bucket]]],
    [for r, bucket in local.regional_models_buckets :
    ["rt-train", "roles/storage.objectViewer", "models-${r}", local.cond_bucket[bucket]]],
    [for r, bucket in local.regional_models_buckets :
    ["rt-runner", "roles/storage.objectViewer", "models-${r}", local.cond_bucket[bucket]]],
  )
}

resource "google_project_iam_member" "bindings" {
  for_each = {
    for b in concat(local.project_bindings, local.regional_models_bindings) :
    "${b[0]} ${b[1]} ${coalesce(b[2], "-")}" => b
  }

  depends_on = [
    google_project_iam_custom_role.deployer,
    google_project_iam_custom_role.storage_manager,
    google_project_iam_custom_role.runner_launcher,
    google_project_iam_custom_role.runner_self,
  ]

  project = var.project_id
  member  = local.sa[each.value[0]]
  role    = each.value[1]

  dynamic "condition" {
    for_each = each.value[3] == null ? [] : [each.value]
    content {
      title      = condition.value[2]
      expression = condition.value[3]
    }
  }
}

# --- "Act as" grants: who may launch something that runs as which runtime SA.
locals {
  act_as = [
    ["rt-train", "ci-experiment"],  # submit Vertex jobs as rt-train
    ["rt-runner", "ci-experiment"], # create burst-runner VMs that run as rt-runner
    ["rt-janitor", "ci-release"],   # deploy the janitor (infra/main)
    ["fn-build", "ci-release"],     # ...and build it
  ]
}

resource "google_service_account_iam_member" "act_as" {
  for_each = { for a in local.act_as : "${a[0]} ${a[1]}" => a }

  service_account_id = google_service_account.sa[each.value[0]].name
  role               = "roles/iam.serviceAccountUser"
  member             = local.sa[each.value[1]]
}

# --- Custom roles: the admin-level permissions ci-release needs, minus every setIamPolicy.

resource "google_project_iam_custom_role" "storage_manager" {
  role_id     = "snapcallStorageManager"
  title       = "SnapCall storage manager (no IAM)"
  description = "Create/update/delete buckets and objects. Like storage.admin without setIamPolicy."
  permissions = [
    "storage.buckets.create",
    "storage.buckets.delete",
    "storage.buckets.get",
    "storage.buckets.list",
    "storage.buckets.update",
    "storage.objects.create",
    "storage.objects.delete",
    "storage.objects.get",
    "storage.objects.list",
    "storage.objects.update",
  ]
}

resource "google_project_iam_custom_role" "deployer" {
  role_id     = "snapcallDeployer"
  title       = "SnapCall deployer (no IAM)"
  description = "Manage the non-identity resources in infra/main: AR repo, secret slots (not values), Scheduler, functions (which run on Cloud Run)."
  permissions = [
    "artifactregistry.repositories.create",
    "artifactregistry.repositories.delete",
    "artifactregistry.repositories.get",
    "artifactregistry.repositories.list",
    "artifactregistry.repositories.update",
    "cloudfunctions.functions.create",
    "cloudfunctions.functions.delete",
    "cloudfunctions.functions.get",
    "cloudfunctions.functions.list",
    "cloudfunctions.functions.sourceCodeSet",
    "cloudfunctions.functions.update",
    "cloudfunctions.operations.get",
    "cloudscheduler.jobs.create",
    "cloudscheduler.jobs.delete",
    "cloudscheduler.jobs.enable",
    "cloudscheduler.jobs.get",
    "cloudscheduler.jobs.list",
    "cloudscheduler.jobs.pause",
    "cloudscheduler.jobs.update",
    "run.operations.get",
    "run.revisions.get",
    "run.revisions.list",
    "run.services.create",
    "run.services.delete",
    "run.services.get",
    "run.services.list",
    "run.services.update",
    "secretmanager.secrets.create",
    "secretmanager.secrets.delete",
    "secretmanager.secrets.get",
    "secretmanager.secrets.list",
    "secretmanager.secrets.update",
  ]

  depends_on = [google_project_service.apis]
}

# --- Burst runner (Decision 9). No setIamPolicy here either.

resource "google_project_iam_custom_role" "runner_launcher" {
  role_id     = "snapcallRunnerLauncher"
  title       = "SnapCall runner launcher"
  description = "Create and delete burst-runner GPU VMs. Bound with a condition on the VM/disk name prefix."
  permissions = [
    "compute.disks.create",
    "compute.disks.delete",
    "compute.instances.create",
    "compute.instances.delete",
    "compute.instances.get",
    "compute.instances.setLabels",
    "compute.instances.setMetadata",
    "compute.instances.setScheduling",
    "compute.instances.setServiceAccount",
    "compute.networks.use",
    "compute.networks.useExternalIp",
    "compute.subnetworks.use",
    "compute.subnetworks.useExternalIp",
    "compute.zoneOperations.get",
  ]

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_custom_role" "runner_self" {
  role_id     = "snapcallRunnerSelfDelete"
  title       = "SnapCall runner self-delete"
  description = "Lets a burst-runner VM delete itself (and its boot disk). Bound with a condition on the name prefix."
  permissions = [
    "compute.disks.delete",
    "compute.instances.delete",
    "compute.instances.get",
    "compute.zoneOperations.get",
  ]

  depends_on = [google_project_service.apis]
}

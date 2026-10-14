locals {
  # Shared names (also used by .github/workflows/). The service accounts and the tfstate and
  # functions buckets come from bootstrap; IAM there is scoped to exactly these names.
  scratch_bucket   = "${var.project_id}-scratch"
  models_bucket    = "${var.project_id}-models"
  functions_bucket = "${var.project_id}-functions"
  ar_repository    = "snapcall"
  # Extra Vertex regions get "<project>-models-<region>" and their own "snapcall" repo; the
  # home region keeps the original names.
  extra_regions = toset([for r in var.vertex_regions : r if r != var.region])
  sa            = { for n in ["rt-janitor", "fn-build"] : n => "${n}@${var.project_id}.iam.gserviceaccount.com" }
}

# ---------------------------------------------------------------------------------------
# Buckets. Both are emptied on destroy (infra-destroy.yml): data is re-creatable from
# data.yml and weights from fetch-weights.yml.
# ---------------------------------------------------------------------------------------
resource "google_storage_bucket" "scratch" {
  name                        = local.scratch_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  # Data, checkpoints, run outputs, janitor logs: nothing here is precious.
  lifecycle_rule {
    condition {
      age = var.scratch_retention_days
    }
    action {
      type = "Delete"
    }
  }

  # Don't pay to keep deleted checkpoints for a week.
  soft_delete_policy {
    retention_duration_seconds = 0
  }
}

resource "google_storage_bucket" "models" {
  name                        = local.models_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  soft_delete_policy {
    retention_duration_seconds = 0
  }
}

# Weights copies for the other Vertex regions (fetch-weights.yml uploads to every one).
resource "google_storage_bucket" "models_regional" {
  for_each = local.extra_regions

  name                        = "${local.models_bucket}-${each.key}"
  location                    = each.key
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  soft_delete_policy {
    retention_duration_seconds = 0
  }
}

# ---------------------------------------------------------------------------------------
# Artifact Registry: images base and snapcall (thin train layer), for the legacy Vertex path.
# ---------------------------------------------------------------------------------------
resource "google_artifact_registry_repository" "snapcall" {
  repository_id = local.ar_repository
  location      = var.region
  format        = "DOCKER"
  description   = "SnapCall train images"

  # CUDA images are several GB each; keep storage from creeping up.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }
  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "1209600s" # 14 days
    }
  }
}

# The same repo in each other Vertex region; base-image.yml pushes to all of them.
resource "google_artifact_registry_repository" "snapcall_regional" {
  for_each = local.extra_regions

  repository_id = local.ar_repository
  location      = each.key
  format        = "DOCKER"
  description   = "SnapCall images for Vertex jobs in ${each.key}"

  # CUDA images are several GB each; keep storage from creeping up.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }
  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "1209600s" # 14 days
    }
  }
}

# ---------------------------------------------------------------------------------------
# Secret slot, no value. You paste the value in the console; anything
# Terraform set would sit in plain text in the state file.
# ---------------------------------------------------------------------------------------
resource "google_secret_manager_secret" "typesafe" {
  secret_id = var.typesafe_secret_id

  replication {
    auto {}
  }
}

data "google_project" "this" {
  project_id = var.project_id
}

locals {
  project_number = data.google_project.this.number

  # Names shared with infra/main and the workflows (.github/workflows/).
  scratch_bucket = "${var.project_id}-scratch"
  models_bucket  = "${var.project_id}-models"
  # Weights copies for the other Vertex regions (created by infra/main, same naming).
  regional_models_buckets = { for r in var.vertex_regions : r => "${var.project_id}-models-${r}" if r != var.region }
  tfstate_bucket          = "${var.project_id}-tfstate"
  functions_bucket        = "${var.project_id}-functions"
  ar_repository           = "snapcall"
  # Burst runner (Decision 9): VM names start with this; the VPC has the same name.
  runner_prefix  = "snapcall-runner"
  runner_network = "snapcall-runner"
}

# ---------------------------------------------------------------------------------------
# APIs. Never disabled on destroy: turning an API off can break things outside this config.
# ---------------------------------------------------------------------------------------
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbilling.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "compute.googleapis.com",
    "eventarc.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "logging.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "serviceusage.googleapis.com",
    "storage.googleapis.com",
    "sts.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------------------
# Terraform state for infra/main. Versioned, so a bad apply can be rolled back by hand.
# ---------------------------------------------------------------------------------------
resource "google_storage_bucket" "tfstate" {
  name                        = local.tfstate_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 20
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.apis]
}

# Zipped source for the Cloud Run functions (kill switch here, janitor in infra/main).
resource "google_storage_bucket" "functions" {
  name                        = local.functions_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------------------
# Budget alerts -> kill switch. You connect your budget to this topic
# in the console (see this folder's README); budgets need billing-account rights, so no Terraform.
# ---------------------------------------------------------------------------------------
resource "google_pubsub_topic" "budget_alerts" {
  name       = "budget-alerts"
  depends_on = [google_project_service.apis]
}

# Workload Identity Federation: GitHub Actions swaps its signed OIDC token for a 1-hour GCP
# token. No keys anywhere.
#
# Two gates:
#   1. The provider's attribute_condition rejects every token that isn't from this repo
#      (by numeric id) AND run by the owner or the optional second account (by numeric user
#      id, which survives renames). A stranger, a fork, or a renamed/recreated repo never
#      gets a GCP token.
#   2. Which service account a token may impersonate depends on its tier:
#        experiment = any branch, owner or second account -> ci-experiment
#        release    = refs/heads/main AND owner            -> ci-release
#      A second-account push to main is still only "experiment".

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  description               = "SnapCall GitHub Actions workflows"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions"
  display_name                       = "GitHub Actions OIDC"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  attribute_mapping = {
    "google.subject"          = "assertion.sub"
    "attribute.repository_id" = "assertion.repository_id"
    "attribute.actor"         = "assertion.actor"
    "attribute.actor_id"      = "assertion.actor_id"
    "attribute.ref"           = "assertion.ref"
    "attribute.tier"          = "(assertion.ref == 'refs/heads/main' && assertion.actor_id == '${var.github_owner_id}') ? 'release' : 'experiment'"
  }

  attribute_condition = join(" && ", [
    "assertion.repository_id == '${var.github_repository_id}'",
    "assertion.repository_owner_id == '${var.github_owner_id}'",
    "assertion.actor_id in [${join(", ", [for id in compact([var.github_owner_id, var.github_bot_id]) : "'${id}'"])}]",
  ])
}

locals {
  wif_principal_set = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}"
}

resource "google_service_account_iam_member" "wif_experiment" {
  service_account_id = google_service_account.sa["ci-experiment"].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "${local.wif_principal_set}/attribute.repository_id/${var.github_repository_id}"
}

resource "google_service_account_iam_member" "wif_release" {
  service_account_id = google_service_account.sa["ci-release"].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "${local.wif_principal_set}/attribute.tier/release"
}

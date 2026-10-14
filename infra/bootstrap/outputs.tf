output "github_variables" {
  description = "Paste these into GitHub: Settings > Secrets and variables > Actions > Variables. Identifiers, not secrets."
  value = {
    GCP_PROJECT_ID = var.project_id
    GCP_REGION     = var.region
    WIF_PROVIDER   = google_iam_workload_identity_pool_provider.github.name
    SA_EXPERIMENT  = google_service_account.sa["ci-experiment"].email
    SA_RELEASE     = google_service_account.sa["ci-release"].email
  }
}

output "tfstate_bucket" {
  description = "State bucket for infra/main."
  value       = google_storage_bucket.tfstate.name
}

output "budget_alerts_topic" {
  description = "Connect the budget to this topic (README step 10)."
  value       = google_pubsub_topic.budget_alerts.id
}

output "killswitch_deployed" {
  description = "false until ops/guard/killswitch/main.py exists and bootstrap is re-applied."
  value       = local.killswitch_ready
}

output "runner" {
  description = "Burst runner (Decision 9): the VM identity, network and name prefix run-gpu.yml uses."
  value = {
    service_account = google_service_account.sa["rt-runner"].email
    network         = google_compute_network.runner.name
    vm_name_prefix  = local.runner_prefix
  }
}

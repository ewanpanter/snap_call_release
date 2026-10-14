output "scratch_bucket" {
  value = google_storage_bucket.scratch.url
}

output "models_bucket" {
  value = google_storage_bucket.models.url
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.snapcall.repository_id}"
}

output "typesafe_secret" {
  description = "Paste the value in the console: Secret Manager > this secret > New version."
  value       = google_secret_manager_secret.typesafe.secret_id
}

output "janitor_deployed" {
  value = local.janitor_ready
}

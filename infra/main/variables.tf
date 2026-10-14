# CI sets project_id and region as TF_VAR_* from the repo variables GCP_PROJECT_ID/GCP_REGION.

variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "Region for everything regional."
  type        = string
  default     = "us-central1"
}

variable "typesafe_secret_id" {
  description = "Secret Manager slot for the vendor API key (Decision 3). Must match bootstrap's variable of the same name, which scopes who can read it."
  type        = string
  default     = "typesafe-api-key"
}

variable "scratch_retention_days" {
  description = "Objects in the scratch bucket are deleted after this many days."
  type        = number
  default     = 30
}

variable "janitor_source_dir" {
  description = "Janitor function source, relative to this directory. Deployed only once <dir>/main.py exists."
  type        = string
  default     = "../../ops/guard/janitor"
}

variable "janitor_entry_point" {
  description = "HTTP handler in <janitor_source_dir>/main.py, called by Cloud Scheduler."
  type        = string
  default     = "run_janitor"
}

variable "janitor_dry_run" {
  description = "If true, the janitor only logs what it would cancel/undeploy/delete."
  type        = bool
  default     = false
}

variable "janitor_schedule" {
  description = "How often the janitor runs (cron, UTC)."
  type        = string
  default     = "*/15 * * * *"
}

variable "vertex_regions" {
  description = "Regions Vertex jobs may run in. Each gets its own models bucket and Artifact Registry repo, so jobs read weights and pull images locally; the janitor watches all of them. Keep in sync with infra/bootstrap and the VERTEX_REGIONS repo variable."
  type        = list(string)
  default     = ["us-central1", "europe-west4"]
}

# Identifiers, not secrets, but they're yours: set them in terraform.tfvars (copy
# terraform.tfvars.example). Variables without a default must be set.

variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "Region for everything regional (the one that had L4 capacity)."
  type        = string
  default     = "us-central1"
}

variable "github_repository_id" {
  description = "Numeric id of your GitHub repo (gh api repos/<owner>/<repo> --jq .id). The id, not the name, so a renamed or recreated repo can't inherit the trust."
  type        = string
}

variable "github_owner_id" {
  description = "Numeric GitHub user id of the repo owner (gh api users/<owner> --jq .id). Owns the repo and is the only Release-tier actor. Set the RELEASE_ACTOR repo variable to the same account's username."
  type        = string
}

variable "github_bot_id" {
  description = "Optional: numeric GitHub user id of a second account (e.g. a bot that pushes branches). Experiment tier only. Leave empty for none."
  type        = string
  default     = ""
}

variable "typesafe_secret_id" {
  description = "Secret Manager secret id for the vendor API key (Decision 3). The slot itself is created by infra/main; this only scopes who may read it. Keep in sync with infra/main."
  type        = string
  default     = "typesafe-api-key"
}

variable "killswitch_source_dir" {
  description = "Kill-switch function source, relative to this directory. Deployed only once <dir>/main.py exists; until then bootstrap creates everything else and you re-run it later."
  type        = string
  default     = "../../ops/guard/killswitch"
}

variable "killswitch_entry_point" {
  description = "Python function in <killswitch_source_dir>/main.py that handles a budget Pub/Sub message (CloudEvent)."
  type        = string
  default     = "handle_budget_alert"
}

variable "vertex_regions" {
  description = "Regions Vertex jobs may run in. Keep in sync with infra/main (which creates a models bucket per extra region) and the VERTEX_REGIONS repo variable."
  type        = list(string)
  default     = ["us-central1", "europe-west4"]
}

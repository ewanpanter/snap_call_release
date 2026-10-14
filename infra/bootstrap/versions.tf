# One-off bootstrap, applied by the project owner from Cloud Shell (see README.md).
#
# No backend block on purpose: the state file stays next to this file in Cloud Shell's home
# directory, which persists between sessions. Never commit it (.gitignore covers *.tfstate*).
terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.8"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

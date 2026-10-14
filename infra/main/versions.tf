# Everything that isn't identity. CI plans it on every push to a branch that
# touches this folder (infra-plan.yml, Experiment tier, -lock=false) and applies it on merge
# to main (infra-apply.yml, Release tier). All IAM is in ../bootstrap, applied by hand.
terraform {
  required_version = ">= 1.5"

  # bucket is passed at init: -backend-config="bucket=<project>-tfstate" (created by bootstrap).
  backend "gcs" {
    prefix = "infra/main"
  }

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

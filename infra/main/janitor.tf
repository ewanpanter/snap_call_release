# Janitor: Cloud Scheduler every 15 min -> function running as rt-janitor,
# which cancels/undeploys/deletes over-long GPU resources and logs to scratch/janitor/.
#
# The code is in ops/guard/janitor/. If main.py is missing there, this file creates nothing.

locals {
  janitor_dir   = abspath("${path.module}/${var.janitor_source_dir}")
  janitor_ready = fileexists("${local.janitor_dir}/main.py")
}

data "archive_file" "janitor" {
  count = local.janitor_ready ? 1 : 0

  type        = "zip"
  source_dir  = local.janitor_dir
  output_path = "${path.module}/.build/janitor.zip"
  excludes    = ["__pycache__", "tests"]
}

resource "google_storage_bucket_object" "janitor" {
  count = local.janitor_ready ? 1 : 0

  bucket = local.functions_bucket
  name   = "janitor/${data.archive_file.janitor[0].output_md5}.zip"
  source = data.archive_file.janitor[0].output_path
}

resource "google_cloudfunctions2_function" "janitor" {
  count = local.janitor_ready ? 1 : 0

  name        = "janitor"
  location    = var.region
  description = "Stops GPU resources that outlive their limits."

  build_config {
    runtime         = "python312"
    entry_point     = var.janitor_entry_point
    service_account = "projects/${var.project_id}/serviceAccounts/${local.sa["fn-build"]}"
    source {
      storage_source {
        bucket = local.functions_bucket
        object = google_storage_bucket_object.janitor[0].name
      }
    }
  }

  service_config {
    service_account_email          = local.sa["rt-janitor"]
    max_instance_count             = 1
    available_memory               = "256M"
    timeout_seconds                = 300
    ingress_settings               = "ALLOW_ALL" # Scheduler calls it over HTTPS; still needs an OIDC token
    all_traffic_on_latest_revision = true
    environment_variables = {
      GCP_PROJECT_ID                = var.project_id
      GCP_REGION                    = var.region
      JANITOR_REGIONS               = join(",", distinct(concat([var.region], var.vertex_regions)))
      SCRATCH_BUCKET                = local.scratch_bucket
      JANITOR_MAX_JOB_MINUTES       = "135"  # Vertex jobs: 2h timeout + 15 min grace
      JANITOR_MAX_QUEUED_MINUTES    = "1455" # not started yet (flex-start queue, free): 24h + 15 min
      JANITOR_MAX_ENDPOINT_MINUTES  = "90"
      JANITOR_MAX_VM_MINUTES        = "60"
      JANITOR_MAX_RUNNER_VM_MINUTES = "135" # burst runner (label snapcall=runner): 2h max-run-duration + 15 min
      JANITOR_DRY_RUN               = tostring(var.janitor_dry_run)
    }
  }
}

resource "google_cloud_scheduler_job" "janitor" {
  count = local.janitor_ready ? 1 : 0

  name             = "janitor"
  region           = var.region
  schedule         = var.janitor_schedule
  time_zone        = "Etc/UTC"
  attempt_deadline = "320s"

  retry_config {
    retry_count = 0 # it runs again in 15 minutes anyway
  }

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions2_function.janitor[0].service_config[0].uri

    oidc_token {
      service_account_email = local.sa["rt-janitor"]
      audience              = google_cloudfunctions2_function.janitor[0].service_config[0].uri
    }
  }
}

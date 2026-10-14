# Kill switch: budget -> Pub/Sub budget-alerts -> this function, which
# unlinks billing at 100% of actual spend. It lives in bootstrap, not infra/main, because it
# runs as rt-killswitch (Project Billing Manager): CI must not be able to deploy code as it.
#
# The code is in ops/guard/killswitch/. If main.py is missing there, this file creates
# nothing.

locals {
  killswitch_dir   = abspath("${path.module}/${var.killswitch_source_dir}")
  killswitch_ready = fileexists("${local.killswitch_dir}/main.py")
}

data "archive_file" "killswitch" {
  count = local.killswitch_ready ? 1 : 0

  type        = "zip"
  source_dir  = local.killswitch_dir
  output_path = "${path.module}/.build/killswitch.zip"
  excludes    = ["__pycache__", "tests"]
}

resource "google_storage_bucket_object" "killswitch" {
  count = local.killswitch_ready ? 1 : 0

  bucket = google_storage_bucket.functions.name
  name   = "killswitch/${data.archive_file.killswitch[0].output_md5}.zip"
  source = data.archive_file.killswitch[0].output_path
}

resource "google_cloudfunctions2_function" "killswitch" {
  count = local.killswitch_ready ? 1 : 0

  name        = "killswitch"
  location    = var.region
  description = "Unlinks billing from ${var.project_id} when the budget reports 100% spend."

  build_config {
    runtime         = "python312"
    entry_point     = var.killswitch_entry_point
    service_account = google_service_account.sa["fn-build"].id
    source {
      storage_source {
        bucket = google_storage_bucket.functions.name
        object = google_storage_bucket_object.killswitch[0].name
      }
    }
  }

  service_config {
    service_account_email          = google_service_account.sa["rt-killswitch"].email
    max_instance_count             = 1
    available_memory               = "256M"
    timeout_seconds                = 60
    ingress_settings               = "ALLOW_INTERNAL_ONLY"
    all_traffic_on_latest_revision = true
    environment_variables = {
      GCP_PROJECT_ID = var.project_id
    }
  }

  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic          = google_pubsub_topic.budget_alerts.id
    retry_policy          = "RETRY_POLICY_DO_NOT_RETRY"
    service_account_email = google_service_account.sa["rt-killswitch"].email
  }

  depends_on = [google_project_iam_member.bindings]
}

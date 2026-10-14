# Network for the burst-runner VMs (Decision 9). The runner's IAM is in iam.tf.
#
# Its own VPC rather than "default": the default network comes with firewall rules that
# allow SSH, RDP and ICMP from anywhere. This one has no firewall rules at all, so every
# inbound connection is dropped and outbound (Hugging Face, PyPI, Google APIs) is allowed.
# The VMs get an ephemeral external IP for that outbound traffic. Auto mode makes a subnet
# in every region, so every region in RUNNER_REGIONS works without listing them here.
# A VPC and its subnets cost nothing.

resource "google_compute_network" "runner" {
  name                    = local.runner_network
  description             = "Burst-runner GPU VMs: no firewall rules, so no inbound traffic."
  auto_create_subnetworks = true

  depends_on = [google_project_service.apis]
}

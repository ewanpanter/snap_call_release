# ops/runner: the GCE burst runner (Decision 9)

GPU jobs run on short-lived Compute Engine VMs instead of Vertex. A launch tries every L4
zone of each region in `RUNNER_REGIONS`, in order, and runs wherever the first create
succeeds. A stockout comes back in seconds, so the search usually takes a minute or two.

```
gh workflow run run-gpu.yml -f script=snapcall.decide -f args="smoke --model {models}/Qwen/Qwen3-0.6B"
# summary prints the run id (gce-<run>-<attempt>), the zone and the search time
gh workflow run fetch-results.yml -f run_id=gce-<run>-<attempt> [-f wait_minutes=30]
gh run download <fetch-results run id>      # -> outputs-<run_id>/
```

## Files

| File | Runs on | Does |
|---|---|---|
| `launch.py` | GitHub runner (`run-gpu.yml`), as `ci-experiment` | Max-VMs check, picks the newest Deep Learning VM image, orders the L4 zones, creates the VM zone by zone, writes `launch.json` |
| `startup.sh` | the VM, as root, at boot (instance metadata) | Code from scratch → uv install → weights → GPU check → `python -m MODULE ARGS` → on exit: save and self-delete |
| `vm.py` | the VM (image's `python3`, stdlib only) | Fetch `{models}/...` weights; write `cost.json`, timings, `done.json` |

## Guardrails

| # | Guard | Where |
|---|---|---|
| 1 | `--max-run-duration=2h` + `--instance-termination-action=DELETE`: Google deletes the VM at 2h whatever it's doing | `launch.py` `create_cmd` (tested) |
| 2 | On exit (success, failure, SIGTERM): outputs + log to scratch, `done.json`, then delete the VM. If the delete fails, power off | `startup.sh` `trap finish EXIT` (tested end to end with stubs) |
| 3 | `GPUS_ALL_REGIONS` = 1: one GPU project-wide. A create that hits it stops the search | quota; `launch.py` |
| 4 | Janitor, every 15 min, every zone: runner VMs older than 2h15m, or stopped, are deleted | `ops/guard/janitor` |
| 5 | Refuse to create if `RUNNER_MAX_VMS` (default 1) VMs labelled `snapcall=runner` exist anywhere. Launches are serialised (concurrency group) so two can't both pass | `launch.py`, `run-gpu.yml` |
| - | IAM: CI can only create VMs named `snapcall-runner-*`; they run as `rt-runner` on the `snapcall-runner` VPC, which has no firewall rules (nothing can connect in) | `infra/bootstrap` |

A create error the launcher doesn't recognise (permissions, ...) stops the search
at once rather than failing the same way in 40 zones.

**Image choice:** the newest Deep Learning VM `common-cu*` image whose status is `READY` and
which isn't deprecated, with the next two newest as fallbacks. If a create says the image
can't be found (it happened on 2026-10-01 with a just-published image), the same zone is
retried with the next fallback; `launch.json` records the image actually used.
`-f image_family=...` still pins a family. If that call somehow left a VM, it is
deleted.

## What the VM does

1. Waits for the NVIDIA driver (the image ships it).
2. Downloads `runs/<run_id>/code.tar.gz` (the workflow's `git archive` of the commit) from
   scratch. Tracked files only.
3. Installs uv, Python 3.12, **the same pinned CUDA torch, peft and accelerate as the Vertex
   image** (read from `docker/Dockerfile.base`), and the rest of `uv.lock`. The image's own
   PyTorch isn't used, so every run gets the same versions.
4. Weights: each `{models}/<org>/<name>` in the args is fetched to local disk, from the models
   bucket in the VM's own region if it has one (`us-central1`, `europe-west4`), otherwise
   from Hugging Face (ungated, no token). `_runner_weights.json` records which.
5. Runs the module in the code directory (outputs go to `outputs/`), with `SNAPCALL_RUN_ID`,
   `SNAPCALL_OUTPUT_URI` and `SNAPCALL_GPU=true` set, as on Vertex.
6. On exit: `cost.json` (unless the module wrote one), `_runner_timings.json`, `_nvidia_smi.txt`
   and `_runner_log.txt` go into `outputs/`, which is copied to `runs/<run_id>/outputs/`;
   then `runs/<run_id>/done.json`; then the VM deletes itself.

Every 5 minutes the log so far goes to `runs/<run_id>/_runner_log_partial.txt`, so a VM
that is hard-killed still leaves something to read.

## Timing data

- `launch.json` (run-gpu artifact and `runs/<run_id>/launch.json`): each zone tried, its
  outcome (`capacity`, `region_quota`, `created`, ...) and seconds, plus `search_seconds`.
- `_runner_timings.json` in the outputs: seconds for boot → script, driver wait, install,
  weights, module, and boot → finish (what's billed, roughly).

## Known gaps

- `cost.json` uses the **us-central1** on-demand rate. Tokyo, Seoul or Zurich cost up to
  ~25% more, and the boot disk (~£0.02/h) isn't counted. Billing is the truth.
- IAM can't say "only yourself", so a runner VM could delete another runner VM (never
  anything else). The `snapcall=runner` label isn't enforced by IAM, only the name prefix.
  An unlabelled `snapcall-runner-*` VM still falls under the janitor's plain 60-minute limit.
- `experiment.yml` still submits to Vertex (the legacy path, kept for reference);
  `train.yml` calls `run-gpu.yml`.

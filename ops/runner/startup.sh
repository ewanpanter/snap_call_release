#!/bin/bash
# Burst-runner VM startup script (Decision 9). ops/runner/launch.py puts it in the VM's
# metadata; it runs once, as root, when the VM boots.
#
#   1. get this commit's code from scratch, install it with uv (pinned CUDA torch, as the
#      Vertex image), fetch weights (regional bucket if one exists, else Hugging Face);
#   2. check the GPU, then run `python -m MODULE ARGS` in the code dir (outputs/ is local);
#   3. ON EXIT, whatever happened: write cost.json and timings, copy outputs/ + this log to
#      scratch, write done.json last, then DELETE THIS VM. If the delete fails, power off:
#      the janitor deletes stopped runner VMs, and max-run-duration deletes it anyway.
#
# Metadata (attributes/snapcall-*): run-id, module, args, code-uri, status-uri,
# output-uri, models-buckets, machine-type, rate-gbp-per-hour, git-sha; snapcall-vm-py is
# ops/runner/vm.py.
set -uo pipefail

MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl -sf -H 'Metadata-Flavor: Google' "$MD/$1"; }
attr() { md "attributes/$1"; }

# SNAPCALL_ROOT is only set by tests/test_runner.py, which runs this script with stub tools.
ROOT="${SNAPCALL_ROOT:-}"
WORK=$ROOT/opt/snapcall
APP=$WORK/app
OUT=$APP/outputs
LOG=$ROOT/var/log/snapcall-runner.log
VENV=$ROOT/opt/venv
MODELS_DIR=$ROOT/opt/models
GPU_WAIT_TRIES="${SNAPCALL_GPU_WAIT_TRIES:-60}"
TIMINGS=$WORK/timings
VM_PY=$WORK/vm.py
mkdir -p "$WORK" "$OUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
stamp() { echo "$1=$(date +%s.%N)" >> "$TIMINGS"; }
say() { echo "[runner $(date -u +%T)] $*"; }

NAME="$(md name)"
ZONE="$(md zone)"; ZONE="${ZONE##*/}"

delete_self() {
  local i
  for i in 1 2 3; do
    say "deleting this VM ($NAME in $ZONE), attempt $i"
    gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet && return 0
    sleep 10
  done
  say "self-delete failed; powering off (the janitor deletes stopped runner VMs)"
  poweroff
}

# Startup scripts run on every boot. A runner never reruns its job.
if [ -e "$WORK/started" ]; then
  say "booted again after a run: not rerunning, deleting"
  delete_self
  exit 0
fi
touch "$WORK/started"
echo "boot=$(date -d "$(uptime -s)" +%s)" >> "$TIMINGS"
stamp script_start

RUN_ID="$(attr snapcall-run-id)"
MODULE="$(attr snapcall-module)"
ARGS="$(attr snapcall-args)"
CODE_URI="$(attr snapcall-code-uri)"
STATUS_URI="$(attr snapcall-status-uri)"
OUTPUT_URI="$(attr snapcall-output-uri)"
BUCKETS="$(attr snapcall-models-buckets)"
MACHINE="$(attr snapcall-machine-type)"
RATE="$(attr snapcall-rate-gbp-per-hour)"
attr snapcall-vm-py > "$VM_PY"

RC=1       # the job's exit status; stays 1 if setup fails
CHILD=""
UPLOADER=""

finish() {
  trap - EXIT TERM INT
  stamp finish
  [ -n "$UPLOADER" ] && kill "$UPLOADER" 2>/dev/null
  say "finishing: exit status $RC"
  if ! python3 "$VM_PY" finish --outputs "$OUT" --timings "$TIMINGS" --exit-code "$RC" \
      --run-id "$RUN_ID" --module "$MODULE" --machine-type "$MACHINE" --rate "$RATE" \
      --zone "$ZONE" --vm "$NAME" > "$WORK/done.json"; then
    printf '{"run_id": "%s", "vm": "%s", "zone": "%s", "exit_code": %d}\n' \
      "$RUN_ID" "$NAME" "$ZONE" "$RC" > "$WORK/done.json"
  fi
  cp "$LOG" "$OUT/_runner_log.txt"
  gcloud storage rsync -r "$OUT" "$OUTPUT_URI" || say "ERROR: couldn't copy outputs to $OUTPUT_URI"
  # Last, so fetch-results only sees "done" once the outputs are there.
  gcloud storage cp "$WORK/done.json" "$STATUS_URI/done.json" || say "ERROR: couldn't write done.json"
  delete_self
}
on_term() {
  say "got SIGTERM (max-run-duration or someone deleting the VM)"
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null && wait "$CHILD"
  RC=143
  exit
}
trap finish EXIT
trap on_term TERM INT

fail() { say "ERROR: $*"; exit; }  # RC stays 1; the EXIT trap does the rest

say "run $RUN_ID: python -m $MODULE $ARGS  (VM $NAME, zone $ZONE, $MACHINE)"

# The log so far, every 5 minutes: if the VM is hard-killed, some of it survives.
( while sleep 300; do
    gcloud storage cp -q "$LOG" "$STATUS_URI/_runner_log_partial.txt" 2>/dev/null
  done ) &
UPLOADER=$!

# --- GPU driver (Deep Learning VM images ship it; give it a moment on first boot) -------
for _ in $(seq "$GPU_WAIT_TRIES"); do nvidia-smi > /dev/null 2>&1 && break; sleep 10; done
nvidia-smi > "$OUT/_nvidia_smi.txt" 2>&1 || fail "no working NVIDIA driver after 10 minutes"
cat "$OUT/_nvidia_smi.txt"
stamp gpu_ready

# --- code + environment -------------------------------------------------------------------
gcloud storage cp "$CODE_URI" "$WORK/code.tar.gz" || fail "couldn't download $CODE_URI"
mkdir -p "$APP" && tar -xzf "$WORK/code.tar.gz" -C "$APP" || fail "bad code tarball"
cd "$APP" || fail "no $APP"

# Same pins as the Vertex base image, read from it so they can't drift apart.
pin() { sed -n "s/^ARG $1=//p" docker/Dockerfile.base; }
UV_VERSION="$(sed -n 's#^COPY --from=ghcr.io/astral-sh/uv:\([0-9.]*\) .*#\1#p' docker/Dockerfile.base)"
[ -n "$UV_VERSION" ] && [ -n "$(pin TORCH_VERSION)" ] || fail "can't read pins from docker/Dockerfile.base"

export HOME="${HOME:-/root}" UV_CACHE_DIR=$ROOT/opt/uv-cache UV_PYTHON_INSTALL_DIR=$ROOT/opt/uv-python \
  HF_HOME=$ROOT/opt/hf
curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" \
  | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh || fail "couldn't install uv $UV_VERSION"
# uv's own Python, never the image's: Triton compiles a C helper against Python.h the
# first time it touches the GPU, and the image's python3.12 has no headers (first proof run,
# 2026-09-30: "fatal error: Python.h: No such file or directory").
UV_PYTHON_PREFERENCE=only-managed uv venv --python 3.12 "$VENV" || fail "uv venv"
"$VENV/bin/python" -c 'import os, sysconfig; p = os.path.join(sysconfig.get_paths()["include"], "Python.h"); assert os.path.isfile(p), p' \
  || fail "the venv's Python has no Python.h (Triton needs it)"
PY=$VENV/bin/python
uv pip install --python "$PY" --index-url "$(pin TORCH_INDEX)" "torch==$(pin TORCH_VERSION)" \
  || fail "couldn't install torch"
uv export --frozen --no-dev --no-emit-project --no-emit-package torch \
  --no-hashes --no-header --no-annotate -o "$WORK/requirements.txt" || fail "uv export"
uv pip install --python "$PY" -r "$WORK/requirements.txt" \
  "peft==$(pin PEFT_VERSION)" "accelerate==$(pin ACCELERATE_VERSION)" || fail "couldn't install deps"
stamp installed

"$PY" -c 'import sys, torch; ok = torch.cuda.is_available(); print("torch", torch.__version__, "cuda", ok, torch.cuda.get_device_name(0) if ok else ""); sys.exit(0 if ok else 1)' \
  || fail "torch can't see the GPU (see _nvidia_smi.txt)"

# --- weights: {models}/<org>/<name> in the args -> $MODELS_DIR/<org>/<name> -------------
python3 "$VM_PY" models --args "$ARGS" --zone "$ZONE" --buckets "$BUCKETS" --dest "$MODELS_DIR" \
  --hf "$VENV/bin/hf" --args-out "$WORK/args" --record "$OUT/_runner_weights.json" \
  || fail "couldn't fetch weights"
mapfile -d '' ARGV < <(python3 -c 'import shlex, sys; sys.stdout.write("\0".join(shlex.split(open(sys.argv[1]).read())))' "$WORK/args")
stamp weights_ready

# --- the job ------------------------------------------------------------------------------
export PYTHONPATH="$APP/src:$APP" SNAPCALL_RUN_ID="$RUN_ID" SNAPCALL_OUTPUT_URI="$OUTPUT_URI" \
  SNAPCALL_GPU=true PYTHONUNBUFFERED=1
stamp module_start
"$PY" -m "$MODULE" "${ARGV[@]}" &
CHILD=$!
wait "$CHILD"
RC=$?
CHILD=""
stamp module_end
say "python -m $MODULE exited $RC"
exit

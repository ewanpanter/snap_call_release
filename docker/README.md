# Containers (legacy Vertex path)

**Superseded by the burst runner** (`ops/runner/`, Decision 9): GPU jobs now run on
short-lived Compute Engine VMs that install the code with uv, so these images are only
used by the Vertex workflows (`experiment.yml`, `vertex-job.yml`, `base-image.yml`).
They're kept for reference.

Two layers:

| Image | Built by | Contents |
|---|---|---|
| `$AR/base:latest` (+ `:<date>-<sha>`), in every Vertex region's repo | `base-image.yml` (dispatch, or a push to main touching `Dockerfile.base`) | Python 3.12 slim, torch 2.14 **cu126**, and the rest of `uv.lock`'s runtime deps, plus peft and accelerate. ~10 min. |
| `$AR/snapcall:<sha>` | `vertex-job.yml`, every experiment/train run | This repo's code (`src/`, `eval/`, `baselines/`, `data/`, `scripts/`) on the base. About a minute. |

`$AR` = `<region>-docker.pkg.dev/<project>/snapcall`, one per Vertex region
(`VERTEX_REGIONS`); a job builds and pulls its image in its own region.

No `nvidia/cuda` base image: the torch wheel ships its CUDA libraries and Vertex provides
the driver. The torch pin in `Dockerfile.base` must match `uv.lock` (a test checks). If
`uv.lock` changes, the thin layer installs the difference, so the base only needs a rebuild
when torch, CUDA or peft change.

## What runs in the job

`docker/entrypoint.py MODULE ARGS...`:

1. runs `python -m MODULE ARGS` in `/app` (outputs go to `/app/outputs/`);
2. passes SIGTERM (cancel, Spot preemption) on to the module;
3. writes `outputs/cost.json` if the module didn't (container wall-clock x the estimated
   rate from `submit_job.py`; Vertex also bills start-up, see `_vertex_job.json`);
4. copies `outputs/` to `$SNAPCALL_OUTPUT_URI` through the `/gcs/` Cloud Storage FUSE
   mount, whether the module succeeded or not. A failed copy fails the job.

Weights are read from the same mount, from the models bucket in the job's own region.
Write `--model {models}/Qwen/Qwen3-4B-Instruct-2507` in the job args and `submit_job.py`
fills in that bucket (`/gcs/<project>-models` in us-central1,
`/gcs/<project>-models-europe-west4` in europe-west4).

## Submitting

`scripts/submit_job.py` (see its docstring, and `.github/workflows/vertex-job.yml`). Normally
via `gh workflow run experiment.yml ...`. Try a spec without submitting:

```
uv run python scripts/submit_job.py --project P --region us-central1 --image IMG \
  --module snapcall.decide --args "smoke --model /gcs/..." --run-id exp-1-1 \
  --output-uri gs://P-scratch/runs/exp-1-1/outputs --service-account SA --gpu --spot --dry-run
```

There's no serving image: the Cloud Run `/v1/decide` service was never built.

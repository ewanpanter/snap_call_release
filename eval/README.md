# Eval & report

| File | Runs on | Does |
|---|---|---|
| `run_eval.py` | burst-runner GPU (`run-gpu.yml`) | One model per job: equivalence check, zero-shot test-set run, latency sweep, label-prior probe |
| `equivalence.py` | GPU | packed / cached vs separate, per dtype, plus the bf16 noise floor |
| `multi.py` | GPU | Several models or adapters on one data file in one job |
| `vllm_score.py` | GPU | Zero-shot scoring on vLLM, for models the transformers path can't run well on one L4 |
| `metrics.py` | anywhere | accuracy, macro-F1, ECE + reliability, Brier, log loss, MAE, AUROC / AP, order sensitivity |
| `report.py` | anywhere (`--group eval`) | every system's metrics, `report.json`, `summary.md`, PNG charts |
| `bench.py` | burst-runner GPU | latency mode vs throughput mode (many states per pass), bf16 vs FP8 (`torch._scaled_mm`), base + LoRA |

## Running the sweep

Only one GPU exists project-wide (`GPUS_ALL_REGIONS` = 1), so run one job at a time; the
launcher refuses anyway if a runner VM exists. Check with `gh run list --workflow run-gpu.yml`.

```
# Qwen3-0.6B, plus the fp32/bf16 equivalence check on 0.6B and 4B
gh workflow run run-gpu.yml -f script=eval.run_eval -f args="\
  --model {models}/Qwen/Qwen3-0.6B --data gs://<project>-scratch/data/test.jsonl \
  --equivalence-models {models}/Qwen/Qwen3-0.6B {models}/Qwen/Qwen3-4B-Instruct-2507 \
  --equivalence-dtypes float32 bfloat16"
# Qwen3-8B: bf16 only (fp32 doesn't fit an L4)
gh workflow run run-gpu.yml -f machine_type=g2-standard-8 -f script=eval.run_eval \
  -f args="--model {models}/Qwen/Qwen3-8B --data gs://<project>-scratch/data/test.jsonl"
# A model with linear-attention layers: cache-copy smoke test first, eval only if it passes.
# Gemma 4 E4B (sliding window) is too big for fp32 on an L4: add --gate-dtype bfloat16 and
# --equivalence-dtypes bfloat16.
gh workflow run run-gpu.yml -f script=eval.run_eval -f args="\
  --model {models}/Qwen/Qwen3.5-4B --data gs://<project>-scratch/data/test.jsonl \
  --forward cached --gate-cached --equivalence-dtypes float32 bfloat16 \
  --equivalence-datasets customer-support-tickets --latency-forwards cached separate"
```

Then `gh workflow run fetch-results.yml -f run_id=<run id> -f wait_minutes=15` and
`gh run download <fetch run id>`.

## Report

`eval.report` scores any set of runs against a gold file and writes `report.json`,
`summary.md` and (with `--group eval`) PNG charts:

```
D=~/snapcall-eval-data
uv run python data/prepare.py --out $D/data                 # gold, deterministic
uv run python data/prepare.py --ood-only --out $D/data      # out-of-domain gold
uv run --group eval python -m eval.report --gold $D/data/test.jsonl --out $D/rep/<name> \
  --run $D/runs/outputs-<run id> --run $D/runs/outputs-<another run id> ...
```

## Metric notes

- Accuracy uses the system's `answer`; calibration metrics use the probability it gave that
  answer. Probabilities are renormalised first (Gemini's stated ones don't always sum to 1).
- Log loss clips at 1e-3: Jev rounds to two decimals and often says 0.00.
- Yes/no questions with few positives make always-"no" look good on accuracy. Read AUROC and
  average precision (chance = the positive rate) for them.
- SnapCall predictions keep each permutation's distribution (`meta.per_permutation`), so one
  debiased run also gives the debias-off answers (the first permutation is the original
  order) and the order-sensitivity numbers.

## Equivalence: masking is exact; bf16 is noisy

`equivalence.py` compares the packed and cached paths with one forward pass per branch. In
float32 the gaps are at the level of float rounding, so **the mask and the cache reuse are
exact**. In bf16 the reference itself moves by as much as packed and cached do, because
near-ties swing on bf16 rounding, so a maximum-gap tolerance can't work there.
`snapcall.decide` therefore checks exactness in float32 with a tight tolerance, and in bf16
only checks that almost every branch keeps its answer.

# The shared contract

The Python source of truth is `src/snapcall/schema.py`. This folder holds the generated JSON
Schemas and one example file per format. Tests check that the schemas are current and that
every example validates, so these can't drift.

| Format | Model | Schema | Example |
|---|---|---|---|
| `POST /v1/decide` request | `DecideRequest` | `schemas/decide_request.schema.json` | `examples/decide_request.example.json` |
| `POST /v1/decide` response | `DecideResponse` | `schemas/decide_response.schema.json` | `examples/decide_response.example.json` |
| `data/*.jsonl` (one line per state) | `DataRecord` | `schemas/data_record.schema.json` | `examples/data.example.jsonl` |
| `predictions.jsonl` | `Prediction` | `schemas/prediction.schema.json` | `examples/predictions.example.jsonl` |
| `metrics.json` | `Metrics` | `schemas/metrics.schema.json` | `examples/metrics.example.json` |
| `cost.json` | `Cost` | `schemas/cost.schema.json` | `examples/cost.example.json` |
| TypeSafe Decisions API | adapter functions | (theirs) | `examples/typesafe_{request,response}.example.json` |

After changing `schema.py`: `uv run python -m snapcall.schema contract/schemas`.

## Conventions

- **Question ids are stable task names** (`queue`, `priority`, `is_billing`), shared by every
  state in a dataset, so metrics can be grouped by question.
- **Questions:** `choice` has `options` (key → description, 2–26); `score` has `levels`
  (ordered low → high, 2–10); `yesno` has a statement plus optional `yes`/`no` descriptions.
- **Gold / answers:** choice = option key (str), score = level index (int, 0-based), yesno = bool.
- **Probabilities:** choice keyed by option key; score keyed `"0"…"n-1"` (as in Jev);
  yesno keyed `"yes"`/`"no"`. Each sums to 1 (±0.02, for Jev's rounding).
- **Score `value`** is the expected level index, so it can fall between levels.
- **`confidence`** is optional: SnapCall fills it with permutation agreement; Jev omits it
  for yes/no.
- **`latency_ms`** on a prediction is for the whole request it was in. Put `n_questions` and
  `debias` in `meta` for the latency sweep.
- **Errors:** a prediction with `error` set has null `answer`/`probabilities`. Eval counts
  them in `n_errors` rather than dropping them silently.
- **Money** is in `Cost.currency` (GBP by default). Keep the as-billed amount in
  `raw_cost`/`raw_currency` when converting (Jev bills in USD).
- **Component-specific extras** go in `meta` (predictions, data) or `extra` (metrics), not new fields.

## TypeSafe mapping

Taken from the published Decisions API reference (OpenRouter, checked 2026-09-27):

| SnapCall | TypeSafe |
|---|---|
| `question` | `instructions` |
| choice `options` | choice `criteria` (object) |
| score `levels` | score `criteria` (array) |
| `yesno`, `yes`/`no` | `noul`, `criteria.true`/`criteria.false` |
| `ChoiceAnswer.choice` | `choice` |
| `ScoreAnswer.value` | `score` |
| `YesNoAnswer.p_yes` | `noul` |

Check this against whichever endpoint your key is for (TypeSafe direct or OpenRouter)
before the first real run.

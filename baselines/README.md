# Baselines

Two reference systems run over the same test split. Both read
`DataRecord` JSONL and write to `--out`:

| File | What |
|---|---|
| `predictions.jsonl` | contract `Prediction` lines, one per (state, question); failures have `error` set |
| `cost.json` | contract `Cost` (GBP; the raw USD figure is kept in `api[0].raw_cost`) |
| `summary.json` | errors, retries, models seen, tokens, latency, and **anomaly counts** with examples |
| `raw_responses.jsonl` | each provider response body (never request headers), for digging into oddities |

## `typesafe_client.py`: Jev (Release tier)

`POST https://api.typesafe.ai/v1/systemone` (TypeSafe direct), model pinned to `jev-1.13.0`,
one request per state with every question (the contract adapter in `schema.py` builds the
body). Standard library only. The key comes from `TYPESAFE_API_KEY`, set by
`baseline-typesafe.yml` from Secret Manager, and only goes into the Authorization header.

- 429/5xx/timeouts retry with backoff (honours `Retry-After`); 401/402/403 stop the run
  after one request instead of burning through the test set.
- Each answer is converted on its own, so one malformed answer fails one question.
- `meta.anomalies` flags anything odd: probabilities not summing to 1 (beyond Jev's
  two-decimal rounding), a choice that isn't the most probable option, a score that isn't
  the expected level, missing confidence, wrong legend, type mismatches. `meta.raw_*` keeps
  the numbers before renormalisation.
- Cost: `usage.cost` (USD) where the response has one, else input tokens × $0.042/M.
- `--repeat-same-order N` (Decision 19, determinism check): N full passes over the file,
  every request identical (original option order; pass 2 starts after pass 1 ends). Pass 1
  is scored as usual; later passes go in each prediction's `meta.repeats`, and
  `summary.json → repeat_same_order` counts answer changes per question and gives latency
  per pass. Can't be combined with `--permutations`. System name gets `-repN`.

## `gemini_client.py`: Gemini on Vertex (Experiment tier)

The "just ask a general LLM" approach: one structured-output call per state. The JSON
schema forces an `answer` from the allowed keys and **stated** `probabilities` per
question (`meta.prob_source = verbalised`). If the model exposes logprobs, the chosen
answer token's probability and top alternatives go in `meta.answer_token_*`
(`verbalised+logprobs`). Temperature 0, thinking off (`thinking_budget=0`). If the model
rejects logprobs or thinking-off, the client adapts once and records it in
`summary.json → config_fallbacks`.

Needs the optional `baselines` dependency group (`google-genai`), kept out of the core
install so the GPU images and local installs stay slim:

```
uv run --group baselines python -m baselines.gemini_client --project P --region R \
    --data test.jsonl --run-id ID --out outputs/ [--model M] [--location global] [--limit 20]
```

Cost is estimated from token counts × list price. Only checked prices are built in
(`PRICES_USD_PER_M`); for other models pass `--price-in/--price-out` (USD per 1M tokens),
otherwise `cost.json` shows 0 and says so.

## Useful flags (both)

`--limit N` takes N states round-robin across datasets (cheap probe), `--concurrency`,
`--timeout`, `--max-attempts`, `--usd-to-gbp` (default 0.75, approximate), `--system`.

# Design decisions

Design decisions made while building SnapCall, lightly edited for publication.

The numbers are the ones used in the original log, so comments in the code such as
"Decision 9" still point to the right entry. Each entry gives what was decided, why, and the
alternative that was turned down. The story of how the project was run is in the
[second blog post](https://ewanpanter.github.io/2026-10-05/building-jev).

## 1. Machine type is a parameter

**Decided:** GPU jobs take the machine type as an input: `g2-standard-4` by default (enough
for 4B models) and `g2-standard-8` for 8B models. Both have one L4.

**Why:** loading 16 GB of 8B weights needs more host RAM than `g2-standard-4` has.

**Alternative:** always use the bigger machine. It costs more for every run, and most runs
don't need it.

## 2. GPU workflows submit and exit

**Decided:** a GPU workflow starts the job and finishes in about a minute. A separate, short
`fetch-results.yml` downloads the outputs once the job has finished. Actions artifacts are kept
for 7 days.

**Why:** a GitHub runner that waits for a GPU job burns Actions minutes doing nothing.

**Alternative:** one workflow that submits, waits and downloads. Simpler, but it pays for the
wait.

## 3. Jev endpoint and the secret's name

**Decided:** call Jev through TypeSafe's own API. The Secret Manager secret that holds the key
is named by a Terraform variable, `typesafe_secret_id` (default `typesafe-api-key`).

**Why:** the schema is published in both places, and direct is one less intermediary. A
variable keeps the name in one place.

**Alternative:** OpenRouter, which also lists Jev's schema.

## 4. No workflow can be started by an outsider

**Decided:** workflows trigger only on `push`, `workflow_dispatch` or `workflow_call`, which
all need write access to the repo. `pull_request_target` and every other trigger a stranger can
cause are never used, not even for checks that need no credentials. A test
(`tests/test_workflow_triggers.py`) enforces this. Repo settings: while the repo is private,
leave "Run workflows from fork pull requests" off; once it's public, set "Approval for running
fork pull request workflows from contributors" to "Require approval for all external
contributors".

**Why:** a classic way to steal secrets is to open a pull request that runs a workflow.

**Alternative:** run credential-free checks on `pull_request`. Safe in theory, but one later
edit could make it unsafe.

## 5. Bootstrap first

**Decided:** `infra/bootstrap/` (identity, permissions, state bucket, kill switch) is built and
applied by hand before `infra/main/`, and changed separately from it.

**Why:** everything else depends on it, and CI must never be able to change permissions.

**Alternative:** one Terraform root. CI would then need the power to change IAM.

## 6. Labels are single tokens

**Decided:** use `yes`/`no`, `A`–`Z` and `0`–`9` as answer labels. CI confirms that each is a
single token in Qwen3's tokenizer.

**Why:** the restricted softmax reads one logit per option, so each label must be one token.

**Alternative:** multi-token labels, scored by summing log-probabilities. More complex, and
not needed.

## 7. A zero-shot model sweep before any fine-tuning

**Decided:** score the same test set zero-shot on several open-weight models before choosing
one to fine-tune: Qwen3-0.6B, Qwen3-4B-Instruct-2507 and Qwen3-8B, plus newer models
with other architectures (Gemma 4 E4B with sliding-window layers; Qwen3.5 with hybrid
linear attention). Models whose layers can't use the packed mask run through the cached path
instead, after a smoke test that `transformers` can copy their cache.

**Why:** to test "size against training recency" for gut-call decisions, and to fine-tune the
best starting point. The sweep showed little benefit beyond 4B parameters, so
Qwen3-4B-Instruct-2507 became the base model.

**Alternative:** pick a model up front. Cheaper, but it leaves the question unanswered.

## 8. Transfer test for LoRA

**Decided:** besides the main LoRA run (trained on all the training data), train one adapter
on the support-ticket data only. Score both, and the zero-shot model, on the tickets test set
(in domain) and the telco test set (never seen by the tickets-only adapter).

**Why:** to separate "learned this dataset's conventions" from "learned the general decision
format".

**Outcome:** the adapters didn't transfer. Fine-tuning helps on the data it was trained on,
and not elsewhere.

## 9. GPU jobs move from Vertex to a Compute Engine "burst runner"

**Decided:** all GPU work runs on short-lived Compute Engine VMs (`ops/runner/`,
`run-gpu.yml`). The Vertex path stays in the repo but is no longer used.

**Why:** Vertex custom jobs often waited for hours for an L4 and then timed out, and Vertex
gives no quick signal about capacity. A Compute Engine create call says within seconds
whether a zone has a GPU.

- **The capacity search is the create call.** Try every L4 zone of each region in an ordered
  list (the `RUNNER_REGIONS` repo variable), move on when a zone is out of stock, and run where
  the first create succeeds. Nothing is created just to probe.
- **Guardrails, all in place before the first run:**
  1. `--max-run-duration=2h` with `--instance-termination-action=DELETE`: Google deletes the VM
     at the deadline even if our code hangs.
  2. The startup script deletes its own VM on success or failure.
  3. The project's `GPUS_ALL_REGIONS` quota stays at 1: one GPU in the whole project, however
     many regions have quota.
  4. The janitor covers every region: it deletes any VM labelled `snapcall=runner` that is older
     than 2h15m, and any stopped runner VM.
  5. The launcher refuses to create a VM if `RUNNER_MAX_VMS` (default 1) runner VMs already
     exist anywhere.
  6. Known gap: Compute Engine spend isn't stopped by anything faster than the budget kill
     switch, which lags. The worst case is one GPU VM for two hours.
- **No cross-region data copying.** Use Google's Deep Learning VM image, install the code with
  `uv`, and download weights from Hugging Face at start-up (ungated, free ingress), or from a
  regional bucket copy where one exists.
- **Identity:** a runtime service account, `rt-runner`, can read models, write scratch and
  delete runner VMs. CI may create VMs only under the `snapcall-runner-` name prefix. All of this
  is in `infra/bootstrap/`, applied by hand.
- **Network:** a VPC of its own with no firewall rules, so nothing can connect in; outbound
  traffic (Hugging Face, PyPI) is allowed.
- **Try-list:** ordered so that regions where it's night come first and the busy US afternoon
  comes last, for example: `asia-northeast1 asia-northeast3 asia-east1 asia-southeast1
  asia-south1 europe-west4 europe-west1 europe-west2 europe-west3 europe-west6 us-east4 us-east1
  us-central1 us-west1 us-west4 northamerica-northeast1 northamerica-northeast2` (17 regions).
- **Parallelism:** `RUNNER_MAX_VMS` can be raised with the GPU quota. A request to raise the
  quota was refused for a new project, so jobs ran one after another. That was fine: the search
  usually found a GPU in seconds (a median of 13 s; the worst was 87 s).

**Alternative:** stay on Vertex with spot, on-demand, a second region or the flex-start queue.
All of them were tried, and none reliably produced a GPU.

## 10. LoRA scope

**Decided:**

- **Runs:** (a) the main LoRA on Qwen3-4B-Instruct-2507; (b) the tickets-only transfer run on
  the same model (Decision 8); (c) the main LoRA on Gemma 4 E4B, which was level with Jev
  zero-shot. Gemma trains through the cached path, because its sliding-window layers rule out
  the single-pass mask; that's about twice as slow.
- **Calibration:** a temperature per question type, plus a bias term for yes/no questions
  (Qwen leaned towards "yes"), fitted on the validation split from the stored per-order
  probabilities.
- **Forgetting check:** a separate out-of-domain test set (AG News and BoolQ). The zero-shot
  model and every adapter are scored on it, to show whether LoRA hurt general ability.
- **Gate and model registry: dropped.** They needed a serving image, which was never built.
  Each report instead has a plain pass/fail line: "the LoRA beat zero-shot on test without
  hurting the out-of-domain set".
- **Runner image:** no pin. The launcher falls back to the next-newest image if the newest
  can't be found.

**Why:** Qwen3-4B shows the clean single-pass mechanism, and its weaknesses (the "yes" lean,
poor calibration) are what LoRA and calibration should fix. Gemma asks whether a newer model
fine-tuned does better.

**Alternative:** full fine-tuning (needs far more GPU memory than an L4 has) or a classifier
head (can't change the question).

## 11. Phi-4 models: training recipe against size

**Decided:** add two Phi-4 models to test whether zero-shot "gut-check" quality comes more from
the training recipe than from size:

- **Phi-4-mini-instruct** (3.8B, MIT licence). Its configured sliding window is longer than
  any input, so it's effectively full attention and may use the packed path: a window at least
  as long as the sequence counts as full attention, confirmed with the float32 agreement check.
  It gets the full treatment: zero-shot, then LoRA run (d) with calibration.
- **Phi-4** (14B, MIT licence) doesn't fit an L4 in bf16, so it runs in 8-bit on
  `g2-standard-8`, labelled "8-bit" everywhere because size and quantisation are confounded.
  Zero-shot only.

**Why:** Phi-4-mini against Qwen3-4B compares two recipes at the same size; Phi-4 against
Phi-4-mini compares two sizes from the same family.

**Outcome:** fine-tuning brought the four models that were fine-tuned to essentially the same
level, so a better prior mattered little once fine-tuned.

**Alternative:** LoRA on Phi-4 14B. Only worth it if its zero-shot score was clearly ahead of
every 4B model, which it wasn't.

## 12. Extra experiments

**Decided:**

- **(i) Unseen question types.** LoRA run (f) on Qwen3-4B-Instruct-2507 holds whole questions
  out of training: tickets `priority` (score) and tickets `type` (choice). Score them for
  zero-shot, for run (a), which saw them, and for run (f), which didn't. Questions whose labels
  are derived from other questions (`is_billing`, `wants_to_leave`) are not held out, because a
  model could infer them. The question: does fine-tuning teach the general decision skill, or
  only the questions it saw?
- **(ii) Latency mode against throughput mode.** One state per forward pass is fair for latency
  against Jev's API, but it overstates the cost per decision, because a GPU doing one request
  at a time is mostly idle. `eval/bench.py` also runs many states per batch and reports
  decisions per second and £ per 1,000 decisions. Both numbers are reported, saying which
  compares with which.
- **(iii) FP8 on the L4.** The L4 supports FP8. Run the best LoRA model and its base in FP8
  in both modes, and report the speed-up and the change in accuracy and calibration against
  bf16.
- **(iv)** Internal documentation tooling. Not relevant to this release.

**Why:** (i) tests the main claim of a product like Jev, that it answers questions nobody
trained it on; (ii) and (iii) make the cost comparison fair.

**Alternative:** report only latency-mode costs, which would make self-hosting look worse
than it is.

## 13. Running GPU jobs unattended

**Decided:** a single-GPU queue that can run unattended. One job runs at a time; the launcher's
refusal when a runner VM already exists is the hard lock, and a claim posted before each launch
is the advisory one. Failures get a bounded number of retries: wait and retry when no zone has
capacity, retry once after an infrastructure error, and at most two fix-and-retry attempts for
a bug in our own code before the job is marked failed and the queue moves on. A spend ceiling
stops new launches for the session. Changing quotas, IAM or infrastructure is never done
unattended.

**Why:** most GPU runs happened while nobody was watching, and a stuck queue or a retry loop
would waste either time or money.

**Alternative:** start each run by hand. Safe, but slow.

The [second blog post](https://ewanpanter.github.io/2026-10-05/building-jev) tells the full
story of how the project was run.

## 14. Yes/no calibration bias per dataset

**Decided:**

- fit the yes/no bias per dataset (or per question, where the validation data has at least
  100 items and at least 10 of each answer), using only that dataset's validation split;
- apply no bias, only the temperature, to datasets that have no validation split (the
  out-of-domain sets), since calibrating them with in-domain data is the wrong thing;
- re-score the earlier LoRA runs from their stored per-order outputs, with no GPU, and keep
  both the old and the new numbers.

**Why:** one bias fitted on the support-ticket data and applied everywhere hurt Qwen3-8B on
AG News, and caused most of run (e)'s forgetting failure.

**Outcome:** test accuracy barely changed. Two forgetting verdicts flipped: run (e)'s failure
became a pass (its old failure came from the shared bias, not the adapter), and run (b)'s pass
became a failure (the shared bias had been helping it by accident, which strengthens "transfer
fails").

**Alternative:** keep one global bias. Simpler, but it calibrates one dataset with another's
data.

## 15. Documentation corrections

**Decided:** record the following corrections. `infra-plan` runs on pushes to branches, not on
pull requests. `train.yml` uses the burst runner. The runner image pin isn't needed. The
gate-and-register job in `fetch-results.yml` was dead code (Vertex only, and the gate was
dropped in Decision 10); it's removed from this release. `fetch-weights.yml` lists only the
Qwen3 models; Gemma and Phi weights come from Hugging Face on the VM.

**Why:** the documents had drifted from the code.

## 16. CFPB complaints: the fair "does it beat Jev?" test

**Decided:** add a realistic domain that no system trained on: consumer complaints from the
US Consumer Financial Protection Bureau (CFPB). They are real, recent, messy text, released
under CC0.

- **Data:** complaints received in June and July 2026, after the models' training cutoffs, to
  limit contamination. English only, deduplicated (exact and near-duplicate), with credit
  reporting capped at 25% of the pool and a stratified sample with a fixed seed. Splits:
  3,000 train, 300 validation, 1,000 test.
- **Gate:** before anyone scores, a TF-IDF nearest-neighbour baseline checks that the test set
  isn't mostly near-copies of the training set (that check had exposed the telco set as
  template-like).
- **Questions:** `product` (choice, 11), `issue` (choice, 26) and `monetary_relief` (yes/no,
  about 9% yes, so judged on AUC rather than accuracy). There's no `outcome` question, because
  `monetary_relief` would be derivable from it.
- **Scoring:** Jev and Gemini, zero-shot Qwen3-4B and Gemma 4 E4B, the adapters from other
  domains, and a LoRA trained on the CFPB training split.
- **Data source change:** the CFPB removed complaint narratives from its database and API in
  September 2026. They survive only in the CFPB's FOIA reading-room archives, which are used
  instead, pinned by sha256.

**Why:** the in-domain comparison favours SnapCall: the adapters train on the test domains,
and some of those datasets are synthetic and formulaic. A held-out, real domain is the fair
test, and it gives the headline.

**Outcome:** the CFPB-trained LoRA beat Jev by about 12 percentage points. Zero-shot, Jev,
Gemini and Gemma were close. Adapters from other domains didn't transfer. One lesson: scoring
must wait until a data set is final, because scoring a draft wastes money.

**Alternative:** rely on the in-domain results only. They flatter fine-tuning.

## 17. Two telecom-flavoured domains: TeleLogs and CVE advisories

**Decided:** the same pattern as the CFPB set (build the data and run the nearest-neighbour
gate, then score zero-shot, transfer and the baselines, then train and score a LoRA), one GPU
job at a time. A third candidate, LoGHub, was dropped.

- **TeleLogs (5G root-cause analysis):** the `telelogs` set in `GSMA/ot-full` on Hugging Face
  (MIT, ungated). One `choice` question, `root_cause`. Lettered options are stripped from the
  text so that the options come only from our prompt. The cases are grouped into drive-test
  segments, each edited once per cause, so the split is by segment, not by case. Every case is
  longer than the packed path's token limit, so scoring falls back to the cached path. It's a
  published benchmark that may be in the models' training data, so it's presented as a
  telecom-knowledge test, not a fully held-out one.
- **CVE advisories:** Red Hat's security data (CC BY 4.0), for CVEs made public in April to
  September 2026. Every question must be answerable from the text, and the text must not contain
  the CVSS vector or score. After a blind hand-check, two questions were kept: `severity`
  (score: low, medium, high, critical) and `attack_vector` (choice of four). The others
  (non-default configuration, "is this device affected?", exploited in the wild) were dropped as
  unanswerable from the text or too rare to score. Cisco's advisories were not used, because
  their terms allow internal use only.

**Why:** to test the system-one claim on harder, more technical decisions.

**Outcome:** TeleLogs failed, as expected: it needs reasoning over tables and numbers. The
fine-tuned model learned to say chance for everything. The CVE fine-tune improved the scores
but narrowly failed the forgetting check, and the TeleLogs fine-tune damaged out-of-domain
performance.

**Alternative:** stop at CFPB. Cheaper, but it wouldn't show where this kind of model fails.

## 18. Gemini baseline: Gemini 3.5 Flash-Lite, one option order

**Decided:** from Decision 17 on, the Gemini baseline uses Gemini 3.5 Flash-Lite on Vertex AI,
asked in one option order only, with the lowest thinking setting the model allows. Each call
logs input, output and thinking tokens, and the cost per 1,000 decisions. Flip rates for
Gemini already existed from the earlier in-domain and CFPB runs on Gemini 2.5 Flash.

**Why:** Gemini scoring was the largest single cost in the project. Asking in one order (four
times fewer calls) with minimal thinking cuts it sharply; the per-token price is the same as
Gemini 2.5 Flash.

**Alternative:** keep four orders on Gemini 2.5 Flash. Comparable flip rates, at four times
the cost.

## 19. Evaluation extras

**Decided:**

- **`monetary_relief` threshold:** pick the yes/no threshold on the CFPB validation split (to
  maximise balanced accuracy) and apply it to test, from the stored predictions. Report it next
  to the 0.5 threshold.
- **Jev determinism check:** send Jev the same 1,000 CFPB test complaints twice, in the same
  option order (`--repeat-same-order` in `baselines/typesafe_client.py`). Any changed answer is
  Jev's own randomness, which separates it from order sensitivity in its flip rate.
- **Not done:** a measured H100 latency run (the project's quota was L4 only). The estimate
  in the post is labelled as an estimate.

## 20. Public release

**Decided:** publish through an allowlist export into a new repository with fresh history.
Internal planning documents, agent files, generated documentation, data and run outputs are
left out, project-specific identifiers become configuration, and this log is published as
`docs/DECISIONS.md`. Licence: Apache-2.0.

**Why:** the working repository's history contains internal notes and identifiers; a fresh
copy is easier to check than a history rewrite.

**Alternative:** make the working repository public. Its history can't be checked as
thoroughly.

## 21. A talk version of the posts

Not relevant to this release.

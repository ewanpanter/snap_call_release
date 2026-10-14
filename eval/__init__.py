"""Evaluation and report.

GPU side (run on the burst runner with ``run-gpu.yml``):

- ``eval.run_eval``: one model per job. Equivalence check (packed / cached vs separate, in
  any dtypes), the zero-shot test-set run, the latency sweep and the label-prior probe.

CPU side (on any machine or a GitHub runner, from downloaded outputs):

- ``eval.metrics``: accuracy, macro-F1, ECE, Brier, log loss, AUROC, order sensitivity.
- ``eval.report``: metrics for every system, charts and a markdown summary.
"""

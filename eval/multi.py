"""Score several models (or one base with different adapters) on one data file in ONE GPU job.

Decision 16 (CFPB held-out domain) needs zero-shot Qwen3-4B, zero-shot Gemma 4 E4B and the
Qwen3-4B adapters (a) and (f) on ``cfpb_test.jsonl``. The GPU slot is shared, so they run
one after another on one VM. Each variant goes through ``run_eval``'s eval stage (debias
auto, one ticket per pass, as in the sweep) into ``<out>/<name>/``. Adapters are merged
and scored at T = 1 (uncalibrated). Calibration is applied afterwards from the stored
per-permutation distributions (``snapcall.calibrate apply``), so one run serves both the
old and the Decision 14 calibration. Outputs are copied to scratch after each variant,
since the 2 h cap deletes the VM before its exit hook runs.

A variant is ``name=<dir name>[,adapter=<dir or gs://>][,forward=<packed|cached>]
[,system=<Prediction.system>],model=<path>``. On the runner, put ``model=`` **last**: its
weight prefetch reads everything after ``{models}/`` in a token as the repo id.

    gh workflow run run-gpu.yml --ref <branch> -f script=eval.multi -f args="\\
      --data gs://<scratch>/data/cfpb_test.jsonl \\
      --variant name=qwen4b,model={models}/Qwen/Qwen3-4B-Instruct-2507 \\
      --variant name=gemma,forward=cached,model={models}/google/gemma-4-E4B-it \\
      --variant name=lora-a,adapter=gs://<scratch>/runs/<run_id>/outputs/adapter,\\
model={models}/Qwen/Qwen3-4B-Instruct-2507"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from eval.bench import load_variant, local_dir, save_progress
from eval.common import device_info, free, local_path, model_label, write_json
from eval.run_eval import stage_eval
from snapcall.masking import FORWARDS
from snapcall.schema import DataRecord, read_jsonl

KEYS = {"name", "model", "adapter", "forward", "system"}


def parse_variant(spec: str) -> dict:
    """``name=x,model=y,...`` -> dict. Commas separate fields; values may hold ``=``/``:``."""
    v = {}
    for part in spec.split(","):
        key, sep, value = part.partition("=")
        key = key.strip()
        if not sep or key not in KEYS:
            raise argparse.ArgumentTypeError(f"bad variant field {part!r} (keys: {sorted(KEYS)})")
        v[key] = value.strip()
    if "name" not in v or "model" not in v:
        raise argparse.ArgumentTypeError(f"variant needs name= and model=: {spec!r}")
    v.setdefault("forward", "packed")
    if v["forward"] not in FORWARDS:
        raise argparse.ArgumentTypeError(f"forward must be one of {sorted(FORWARDS)}")
    if "system" not in v:
        kind = f"lora-{v['name']}-uncal" if v.get("adapter") else "zeroshot"
        v["system"] = f"snapcall-{model_label(v['model'])}-{kind}"
    return v


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p = argparse.ArgumentParser(prog="python -m eval.multi")
    p.add_argument("--data", required=True, help="DataRecord jsonl, local or gs://")
    p.add_argument("--out", default="outputs")
    p.add_argument("--variant", action="append", type=parse_variant, required=True)
    p.add_argument("--limit", type=int, default=None, help="first N states only")
    a = p.parse_args(argv)
    names = [v["name"] for v in a.variant]
    if len(set(names)) != len(names):
        p.error(f"variant names must be unique: {names}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(local_path(a.data), DataRecord))[: a.limit]
    summary: dict = {"device": device_info(), "data": a.data, "n_states": len(records)}
    summary["variants"] = {}
    failed = False
    for v in a.variant:
        entry: dict = {k: v.get(k) for k in sorted(KEYS)}
        summary["variants"][v["name"]] = entry
        vdir = out / v["name"]
        vdir.mkdir(parents=True, exist_ok=True)
        adapter = model = tok = None
        try:
            t0 = time.perf_counter()
            adapter = local_dir(v["adapter"]) if v.get("adapter") else None
            model, tok, _ = load_variant(v["model"], adapter, "bf16")
            entry["load_seconds"] = time.perf_counter() - t0
            write_json(out / "multi.json", summary)
            args = SimpleNamespace(
                system=v["system"],
                run_id=os.environ.get("SNAPCALL_RUN_ID"),
                forward=v["forward"],
                dtype="bfloat16",
                quant=None,
            )
            entry["eval"] = stage_eval(args, model, tok, records, vdir)
            entry["ok"] = True
        except Exception as e:  # one variant's failure must not lose the others
            traceback.print_exc()
            entry.update(ok=False, error=f"{type(e).__name__}: {e}"[:1000])
            failed = True
        finally:
            model = tok = None  # noqa: F841 - drop references before freeing
            free()
            if adapter is not None and adapter.parent.name.startswith("snapcall-adapter-"):
                shutil.rmtree(adapter, ignore_errors=True)
            write_json(out / "multi.json", summary)
            save_progress(out)
    print(json.dumps({k: e.get("ok") for k, e in summary["variants"].items()}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

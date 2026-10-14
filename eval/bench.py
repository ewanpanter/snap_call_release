"""Latency mode vs throughput mode, bf16 vs FP8, for one base model and its LoRA (Decision 12).

The sweep ran one ticket per forward pass. That is the right comparison for *latency*
against Jev's API (one request at a time), but it overstates *cost per decision*: a GPU
doing one ticket at a time is mostly idle, and Jev's price reflects batched serving. This
job measures both, for each variant:

- ``eval``: the 1,000 test states through ``decide`` one at a time (latency mode), as in
  the sweep -> ``<variant>/predictions.jsonl``, ``metrics*.json``, ``eval_timings.json``.
  LoRA variants are scored at T = 1 (uncalibrated) in both precisions, so bf16 vs FP8 is
  like for like; the published LoRA numbers are calibrated on top of that.
- ``throughput``: many tickets per forward pass. Each ticket is still packed (its prefix +
  question branches with the branch mask); tickets are stacked as a batch, padded to the
  longest, sorted by length first as a serving queue would bucket them. Sweeps
  ``--batch-sizes`` until memory runs out -> ``<variant>/throughput.json`` (tickets/s,
  decisions/s, tokens/s, batch latency, peak memory), plus a check that batched answers
  match one-at-a-time answers.

Variants: ``bf16``, ``fp8`` (per-tensor FP8 weights, dynamic per-tensor FP8 activations,
``torch._scaled_mm`` on the L4's Ada tensor cores; lm_head stays bf16), each for the base
model and for the base with ``--adapter`` merged in. GBP per 1,000 decisions is computed in
the report from the VM's hourly rate (``cost.json``).

    gh workflow run run-gpu.yml --ref <branch> -f script=eval.bench -f args="\\
      --model {models}/Qwen/Qwen3-4B-Instruct-2507 \\
      --adapter gs://<scratch>/runs/<run_id>/outputs/adapter \\
      --data gs://<scratch>/data/test.jsonl"

One option order vs the sweep's four (production runs one): add ``--debias auto off`` (and
``--skip-base --skip-eval`` for a latency-only run on the merged LoRA).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from eval.common import device_info, free, load_model, local_path, model_label, sync, write_json
from eval.run_eval import stage_eval
from snapcall.masking import FORWARDS, pack
from snapcall.prompt import DEFAULT_LABELS, build_prompt, encode_prompt, profile_for
from snapcall.readout import read_branches, restricted_softmax
from snapcall.schema import DataRecord, read_jsonl

FP8_MAX = 448.0  # largest float8_e4m3fn value


# --------------------------------------------------------------------------------------
# FP8 linear layers (no extra dependency: PyTorch's own FP8 matmul)
# --------------------------------------------------------------------------------------


class FP8Linear(torch.nn.Module):
    """``nn.Linear`` with FP8 (e4m3) weights and per-tensor scales.

    Weights are quantised once (one scale for the whole matrix). Activations are quantised
    on the fly, one scale per call. The matmul runs in FP8 on the tensor cores and writes
    bf16. Per-tensor scaling is the simplest scheme the L4 (compute capability 8.9) runs
    natively; finer (row-wise) scales need newer GPUs.
    """

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        w = linear.weight.detach()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        w_scale = (w.abs().max().float() / FP8_MAX).clamp(min=1e-12)
        self.register_buffer("weight_fp8", (w.float() / w_scale).to(torch.float8_e4m3fn))
        self.register_buffer("w_scale", w_scale.reshape(()))
        self.bias = linear.bias
        self.out_dtype = w.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        m = x2.shape[0]
        pad = (-m) % 16  # _scaled_mm wants dimensions that are multiples of 16
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        x_scale = (x2.abs().amax().float() / FP8_MAX).clamp(min=1e-12)
        # Scale in the activations' own dtype (no float32 copy), clamp against rounding up.
        x_fp8 = (x2 * (1.0 / x_scale).to(x2.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),  # column-major view, as _scaled_mm expects
            scale_a=x_scale,
            scale_b=self.w_scale,
            out_dtype=self.out_dtype,
        )
        out = out[:m]
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


def fp8_matmul_microbench(m: int = 1280, k: int = 2560, n: int = 9728, iters: int = 50) -> dict:
    """Is the FP8 matmul itself faster than bf16 on this GPU? One MLP-sized matmul (a ticket's
    worth of tokens x Qwen3-4B's hidden x intermediate size), timed both ways, plus the
    per-call activation quantisation FP8Linear adds."""
    if not torch.cuda.is_available():
        return {}
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    lin = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
    lin.weight.data.copy_(w)
    fp8 = FP8Linear(lin)
    w8, ws = fp8.weight_fp8, fp8.w_scale
    x8 = (x * (1.0 / (x.abs().amax().float() / FP8_MAX)).to(x.dtype)).to(torch.float8_e4m3fn)
    xs = (x.abs().amax().float() / FP8_MAX).reshape(())

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    bf16_ms = timed(lambda: x @ w.t())
    fp8_ms = timed(
        lambda: torch._scaled_mm(x8, w8.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
    )
    layer_ms = timed(lambda: fp8(x))
    flops = 2 * m * k * n
    return {
        "shape": [m, k, n],
        "bf16_ms": bf16_ms,
        "fp8_matmul_ms": fp8_ms,
        "fp8_layer_ms": layer_ms,
        "bf16_tflops": flops / bf16_ms / 1e9,
        "fp8_tflops": flops / fp8_ms / 1e9,
    }


def to_fp8(model) -> dict:
    """Swap every ``nn.Linear`` inside the transformer body for ``FP8Linear`` (lm_head and
    embeddings stay bf16). Returns how many were swapped and the worst relative error of
    an FP8 layer against bf16 on a random input (a self-check, recorded in the outputs)."""
    swapped, worst = 0, 0.0
    body = model.model
    for _, module in list(body.named_modules()):
        for child_name, child in list(module.named_children()):
            if (
                isinstance(child, torch.nn.Linear)
                and child.in_features % 16 == 0
                and (child.out_features % 16 == 0)
            ):
                fp8 = FP8Linear(child)
                if swapped < 4:  # spot-check the first few
                    x = torch.randn(
                        32, child.in_features, device=child.weight.device, dtype=child.weight.dtype
                    )
                    ref = child(x).float()
                    err = ((fp8(x).float() - ref).norm() / ref.norm()).item()
                    worst = max(worst, err)
                setattr(module, child_name, fp8)
                swapped += 1
    free()
    return {"fp8_linears": swapped, "fp8_selfcheck_rel_err": worst}


# --------------------------------------------------------------------------------------
# Many tickets per forward pass
# --------------------------------------------------------------------------------------


def label_logits(model, hidden_rows: torch.Tensor) -> torch.Tensor:
    """lm_head on just the rows we read (not the whole batch x sequence x vocab), plus the
    model's final logit soft-capping if it has one (Gemma), so logits match ``model()``."""
    logits = model.lm_head(hidden_rows)
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return logits


def batched_packed_logits(model, encoded: list) -> list[torch.Tensor]:
    """One forward pass for several tickets: each ticket packed with its own branch mask,
    the tickets stacked as a batch and right-padded to the longest. Padding rows attend to
    themselves only (so their softmax is defined) and are never read. Returns each
    ticket's last-token logits per branch, as ``packed_forward`` would."""
    device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype
    packs = [pack(e.prefix_ids, e.branch_ids) for e in encoded]
    b, t = len(packs), max(p.input_ids.shape[1] for p in packs)
    ids = torch.zeros(b, t, dtype=torch.long)
    pos = torch.zeros(b, t, dtype=torch.long)
    allowed = torch.zeros(b, t, t, dtype=torch.bool)
    for i, p in enumerate(packs):
        n = p.input_ids.shape[1]
        ids[i, :n], pos[i, :n] = p.input_ids[0], p.position_ids[0]
        allowed[i, :n, :n] = p.allowed
        allowed[i, torch.arange(n, t), torch.arange(n, t)] = True
    mask = torch.zeros(b, 1, t, t, dtype=dtype)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    hidden = model.model(
        input_ids=ids.to(device),
        position_ids=pos.to(device),
        attention_mask=mask.to(device),
        use_cache=False,
    ).last_hidden_state
    rows = [hidden[i, p.last_index] for i, p in enumerate(packs)]
    flat = label_logits(model, torch.cat(rows))
    out, k = [], 0
    for r in rows:
        out.append(flat[k : k + len(r)])
        k += len(r)
    return out


def prepare(tok, records: list[DataRecord], debias: str = "auto"):
    """(request, prompt, encoded) per record. ``debias="auto"`` packs every option order the
    sweep used (4 for choice questions); ``"off"`` packs one order, as a production server
    would. Tokenising is CPU work a server overlaps with the GPU, so it isn't timed."""
    profile = profile_for(tok)
    items = []
    for r in records:
        req = r.to_request(debias=debias)
        prompt = build_prompt(req, DEFAULT_LABELS, profile)
        items.append((req, prompt, encode_prompt(prompt, tok)))
    return items


def throughput(
    model, tok, records, batch_sizes, check_states=32, n_tickets=None, debias="auto"
) -> dict:
    items = prepare(tok, records, debias)
    items.sort(key=lambda x: x[2].n_tokens)  # length buckets, as a serving queue would
    if n_tickets and n_tickets < len(items):  # every k-th by length: same length mix, less time
        step = len(items) / n_tickets
        items = [items[int(i * step)] for i in range(n_tickets)]
    n_decisions = sum(len(req.questions) for req, _, _ in items)
    n_tokens = sum(e.n_tokens for _, _, e in items)

    # Do batched answers match one-at-a-time packed answers? (bf16 noise expected)
    sub = items[:: max(1, len(items) // check_states)][:check_states]
    with torch.inference_mode():
        batched = batched_packed_logits(model, [e for _, _, e in sub])
        gaps, same = [], 0
        for (_req, prompt, e), bl in zip(sub, batched, strict=True):
            one = FORWARDS["packed"](model, e.prefix_ids, e.branch_ids)
            for row_b, row_1, br in zip(bl, one, prompt.branches, strict=True):
                ids = e.label_ids[: len(br.keys)]
                pb, p1 = restricted_softmax(row_b, ids), restricted_softmax(row_1, ids)
                gaps.append((pb - p1).abs().max().item())
                same += int(pb.argmax() == p1.argmax())
    check = {
        "n_branches": len(gaps),
        "max_prob_gap": max(gaps),
        "p95_prob_gap": sorted(gaps)[int(0.95 * (len(gaps) - 1))],
        "same_answer_rate": same / len(gaps),
    }

    rows = []
    for bs in batch_sizes:
        batches = [items[i : i + bs] for i in range(0, len(items), bs)]
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                batched_packed_logits(model, [e for _, _, e in batches[-1]])  # warm-up, longest
                sync()
                lat = []
                t0 = time.perf_counter()
                for batch in batches:
                    t1 = time.perf_counter()
                    logits = batched_packed_logits(model, [e for _, _, e in batch])
                    for (req, prompt, e), lg in zip(batch, logits, strict=True):
                        read_branches(req, prompt.branches, lg, e.label_ids)
                    sync()
                    lat.append((time.perf_counter() - t1) * 1000)
                seconds = time.perf_counter() - t0
        except torch.cuda.OutOfMemoryError:
            free()
            rows.append({"batch_size": bs, "oom": True})
            print(f"throughput bs={bs}: out of memory, stopping the sweep", flush=True)
            break
        lat.sort()
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
        rows.append(
            {
                "batch_size": bs,
                "seconds": seconds,
                "tickets_per_s": len(items) / seconds,
                "decisions_per_s": n_decisions / seconds,
                "tokens_per_s": n_tokens / seconds,
                "batch_p50_ms": lat[len(lat) // 2],
                "batch_p95_ms": lat[int(0.95 * (len(lat) - 1))],
                "peak_gib": peak,
            }
        )
        print(f"throughput bs={bs}: {rows[-1]['decisions_per_s']:.1f} decisions/s", flush=True)
    return {
        "debias": debias,
        "n_tickets": len(items),
        "n_decisions": n_decisions,
        "tokens_per_ticket_p50": sorted(e.n_tokens for _, _, e in items)[len(items) // 2],
        "n_tokens": n_tokens,
        "batched_vs_single": check,
        "rows": rows,
    }


# --------------------------------------------------------------------------------------


def save_progress(out: Path) -> None:
    """Copy outputs/ to the run's scratch folder now, not only when the VM exits: a run
    that hits the 2 h cap is deleted before its exit hook can copy anything (attempt 1)."""
    import os

    uri = os.environ.get("SNAPCALL_OUTPUT_URI")
    if not uri or shutil.which("gcloud") is None:
        return
    r = subprocess.run(["gcloud", "storage", "rsync", "-r", str(out), uri], capture_output=True)
    print(f"progress copied to {uri} (exit {r.returncode})", flush=True)


def local_dir(uri: str) -> Path:
    """A local copy of a directory (``gs://`` copied with gcloud)."""
    if not uri.startswith("gs://"):
        return Path(uri)
    dest = Path(tempfile.mkdtemp(prefix="snapcall-adapter-"))
    subprocess.run(["gcloud", "storage", "cp", "-r", uri.rstrip("/") + "/*", str(dest)], check=True)
    return dest


def load_variant(model_path: str, adapter: Path | None, precision: str):
    model, tok = load_model(model_path, "bfloat16")
    info: dict = {"precision": precision, "adapter": str(adapter) if adapter else None}
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().eval()
    if precision == "fp8":
        info.update(to_fp8(model))
    return model, tok, info


def main(argv: list[str] | None = None) -> int:
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p = argparse.ArgumentParser(prog="python -m eval.bench")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (local or gs://)")
    p.add_argument("--data", required=True, help="test.jsonl")
    p.add_argument("--out", default="outputs")
    p.add_argument("--precisions", nargs="+", default=["bf16", "fp8"], choices=["bf16", "fp8"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument(
        "--throughput-tickets",
        type=int,
        default=256,
        help="length-stratified sample for the throughput sweep (all 1,000 took ~30 min/variant)",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-eval", action="store_true", help="throughput only")
    p.add_argument(
        "--debias",
        nargs="+",
        default=["auto"],
        choices=["auto", "off"],
        help="option orders packed per ticket in the throughput sweep: auto = the sweep's 4, "
        "off = one order (production); each gets its own throughput[-off].json",
    )
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="with --adapter, bench only the merged LoRA (same latency as its base)",
    )
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(local_path(a.data), DataRecord))[: a.limit]
    adapter = local_dir(a.adapter) if a.adapter else None
    label = model_label(a.model)
    variants = [(None, prec) for prec in a.precisions]
    if adapter is not None:
        variants = [] if a.skip_base else variants
        variants += [(adapter, prec) for prec in a.precisions]

    summary: dict = {"device": device_info(), "variants": {}}
    if "fp8" in a.precisions:
        try:
            summary["fp8_microbench"] = fp8_matmul_microbench()
            print(f"fp8 microbench: {summary['fp8_microbench']}", flush=True)
        except Exception as e:  # noqa: BLE001 - a diagnostic, not the job
            summary["fp8_microbench"] = {"error": f"{type(e).__name__}: {e}"[:500]}
    failed = False
    for adp, prec in variants:
        name = f"{'lora' if adp else 'base'}-{prec}"
        vdir = out / name
        vdir.mkdir(parents=True, exist_ok=True)
        entry: dict = {}
        summary["variants"][name] = entry
        try:
            t0 = time.perf_counter()
            model, tok, info = load_variant(a.model, adp, prec)
            entry.update(info, load_seconds=time.perf_counter() - t0)
            write_json(out / "bench.json", summary)
            if not a.skip_eval:
                kind = "lora-uncal" if adp else "zeroshot"
                args = SimpleNamespace(
                    system=f"snapcall-{label}-{kind}-{prec}",
                    run_id=os.environ.get("SNAPCALL_RUN_ID"),
                    forward="packed",
                    dtype="bfloat16",
                    quant=prec if prec != "bf16" else None,
                )
                entry["eval"] = stage_eval(args, model, tok, records, vdir)
                write_json(out / "bench.json", summary)
            for debias in a.debias:
                tp = throughput(
                    model, tok, records, a.batch_sizes, n_tickets=a.throughput_tickets,
                    debias=debias,
                )  # fmt: skip
                suffix = "" if debias == "auto" else f"-{debias}"
                write_json(vdir / f"throughput{suffix}.json", tp)
                summ = {k: v for k, v in tp.items() if k != "rows"}
                summ["best_decisions_per_s"] = max(
                    (r["decisions_per_s"] for r in tp["rows"] if not r.get("oom")), default=None
                )
                entry[f"throughput{suffix}"] = summ
                write_json(out / "bench.json", summary)
            entry["ok"] = True
        except Exception as e:  # one variant's failure must not lose the others
            traceback.print_exc()
            entry.update(ok=False, error=f"{type(e).__name__}: {e}"[:1000])
            failed = True
        finally:
            model = tok = None  # noqa: F841 - drop references before freeing
            free()
            write_json(out / "bench.json", summary)
            save_progress(out)
    if adapter is not None and adapter.parent.name.startswith("snapcall-adapter-"):
        shutil.rmtree(adapter, ignore_errors=True)
    print(json.dumps({k: v.get("ok") for k, v in summary["variants"].items()}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

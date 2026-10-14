"""LoRA fine-tuning for SnapCall decisions, then calibration and eval.

**What is trained.** Small LoRA matrices (PEFT) on every attention and MLP projection; the
base weights stay frozen. The forward pass is the one inference uses: ``packed_forward``,
the state once plus one branch per question, each branch masked from the others. So a
training step sees a state and *all* its questions together, exactly as ``decide`` does.
Models with sliding-window or linear-attention layers (Gemma 4) can't use the packed mask,
so they train (and are evaluated) through ``cached_forward``: the state's cache is copied
once per question, and the gradient flows back through the copies into the state's pass.

**The loss** is cross-entropy over the *restricted* option logits: at each branch's last
token, keep only the logits of the labels that branch offers (``A``, ``B``, ...), softmax
over those, and take ``-log p(gold label)``. That is ``readout.restricted_softmax`` with a
gradient. The other ~150k vocabulary entries play no part, so the model learns to rank the
offered options, not to generate text.

**Option shuffling.** Every time a question is seen, its options get a fresh order, so the
gold answer sits behind a random label. The model can't learn "the answer is usually A";
it has to read the options. ``choice``: a random permutation. ``score``: low-to-high or
high-to-low at random (never shuffled: the order is the meaning). ``yesno``: Yes first or
No first at random. These are the same kinds of order ``decide`` uses for debiasing.

**Pipeline** (one GPU job on the burst runner, via ``train.yml`` or ``run-gpu.yml``)::

    [--eval-base: base model on the test split -> outputs/base/]
    LoRA train on train.jsonl                   -> outputs/adapter/, train_log.jsonl
    eval on val (T = 1)                         -> outputs/val/
    fit a temperature per type (+ yes/no bias)  -> outputs/temperatures.json
    eval on test (T = 1)                        -> outputs/uncalibrated/
    apply the calibration                       -> outputs/predictions.jsonl, metrics*.json
    [--ood: same on an out-of-domain set        -> outputs/ood/ (forgetting check;
     with --eval-base, the base model too          base: outputs/base/ood/)]

Eval is the zero-shot harness (``eval.run_eval.stage_eval`` and ``eval.metrics``), so the numbers
line up with the zero-shot sweep. ``--datasets`` restricts training and calibration data,
e.g. to tickets only for the transfer test (Decision 8); test always covers every dataset.
``--hold-out DATASET:QUESTION ...`` removes whole questions from training and calibration
(Decision 12 run f: can the adapter answer questions it never saw?); test still scores them.

    gh workflow run train.yml -f args="--model {models}/Qwen/Qwen3-4B-Instruct-2507 \\
        --data gs://<project>-scratch/data"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from snapcall import calibrate
from snapcall.decide import PACKED_MAX_TOKENS
from snapcall.masking import FORWARDS, full_attention_only
from snapcall.prompt import (
    DEFAULT_LABELS,
    PromptProfile,
    branch_text,
    label_token_ids,
    prefix_text,
    profile_for,
)
from snapcall.schema import (
    ChoiceQuestion,
    DataRecord,
    Prediction,
    answer_keys,
    read_jsonl,
    write_jsonl,
)

# Optimizer steps timed before fitting the plan to ``max_minutes``.
PACE_STEPS = 10
# Every linear layer in a decoder block: attention (q, k, v, o) and MLP (gate, up, down).
# Phi-3/Phi-4 fuse them (qkv_proj, gate_up_proj): listed too, or a Phi LoRA would only get o/down.
LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj",
    "gate_proj", "up_proj", "gate_up_proj", "down_proj",
)  # fmt: skip
# ...of the language model only: Gemma 4 loads with vision and audio towers whose layers
# have the same names. PEFT full-matches this against each module's dotted name.
LORA_TARGET_RE = rf"^(?!.*(?:vision|audio)).*\.(?:{'|'.join(LORA_TARGETS)})$"


# --------------------------------------------------------------------------------------
# One training example: a state and all its questions, options shuffled
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    state_id: str
    prefix_ids: list[int]
    branch_ids: list[list[int]]  # one branch per question
    n_options: list[int]  # per branch: labels offered are the first n of ``label_ids``
    gold: list[int]  # per branch: index of the gold answer's label

    @property
    def n_tokens(self) -> int:
        return len(self.prefix_ids) + sum(len(b) for b in self.branch_ids)


def shuffled_keys(q, rng: random.Random) -> tuple[str, ...]:
    """A fresh option order for one training view of ``q`` (see the module docstring)."""
    keys = answer_keys(q)
    if isinstance(q, ChoiceQuestion):
        return tuple(rng.sample(keys, len(keys)))
    return tuple(keys if rng.random() < 0.5 else keys[::-1])


def make_example(
    record: DataRecord,
    tokenizer,
    rng: random.Random,
    profile: PromptProfile,
    labels: tuple[str, ...] = DEFAULT_LABELS,
) -> Example:
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    branch_ids, n_options, gold = [], [], []
    for qid, q in record.questions.items():
        keys = shuffled_keys(q, rng)
        branch_ids.append(encode(branch_text(q, keys, labels, profile)))
        n_options.append(len(keys))
        gold.append(keys.index(calibrate.gold_key(q, record.gold[qid])))
    prefix = encode(prefix_text(record.state, profile))
    return Example(record.state_id, prefix, branch_ids, n_options, gold)


def restricted_loss(
    branch_logits: torch.Tensor, label_ids: list[int], example: Example
) -> torch.Tensor:
    """Mean over branches of ``-log softmax(logits[offered labels])[gold]``."""
    losses = []
    for logits, n, gold in zip(branch_logits, example.n_options, example.gold, strict=True):
        restricted = logits[label_ids[:n]].float()
        losses.append(-torch.log_softmax(restricted, dim=-1)[gold])
    return torch.stack(losses).mean()


def example_loss(
    model, example: Example, label_ids: list[int], forward: str = "packed"
) -> torch.Tensor:
    logits = FORWARDS[forward](model, example.prefix_ids, example.branch_ids)
    return restricted_loss(logits, label_ids, example)


# --------------------------------------------------------------------------------------
# LoRA and the training loop
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 2e-4
    epochs: int = 1
    grad_accum: int = 8  # states per optimizer step
    warmup: float = 0.05  # fraction of optimizer steps
    max_grad_norm: float = 1.0
    max_steps: int | None = None  # optimizer steps; None = the whole of ``epochs``
    # Time budget for training, keeping time for eval. After PACE_STEPS optimizer steps the
    # plan shrinks to what fits, so the learning-rate schedule still decays to zero; a hard
    # stop at the limit is the backstop.
    max_minutes: float | None = None
    # Skip states whose packed prompt is longer: decide() runs those with the cached forward,
    # not the packed one trained here, and it bounds activation memory on the L4.
    max_tokens: int = PACKED_MAX_TOKENS
    grad_checkpointing: bool = False
    forward: str = "packed"  # "cached" for sliding-window / linear-attention models
    seed: int = 0
    log_every: int = 10  # optimizer steps


def add_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    if cfg.grad_checkpointing and cfg.forward == "cached":
        # HF turns the cache off under gradient checkpointing; cached_forward needs it.
        raise ValueError("gradient checkpointing doesn't work with the cached forward")
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=LORA_TARGET_RE,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


class CpuLookup(torch.nn.Module):
    """An embedding table kept in CPU memory: looks rows up there and returns them on the
    input's device. For big, frozen tables that a forward pass only reads a few rows of."""

    def __init__(self, table: torch.nn.Embedding):
        super().__init__()
        self.table = table.to("cpu")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table(input_ids.to("cpu")).to(input_ids.device)


# Gemma 4 E-models' per-layer embeddings: 2.8B parameters (5.25 GiB in bf16) of the E4B's 7.9B,
# but each token reads one ~20 KB row. On the GPU they left too little room to train (the first
# attempt at run (c) ran out of memory at step 1).
CPU_TABLES = ("embed_tokens_per_layer",)


def offload_lookup_tables(model) -> list[str]:
    """Move ``CPU_TABLES`` embeddings to CPU memory (same results). Returns what moved."""
    moved = []
    for name, module in list(model.named_modules()):
        parent, _, leaf = name.rpartition(".")
        if leaf in CPU_TABLES and isinstance(module, torch.nn.Embedding):
            setattr(model.get_submodule(parent), leaf, CpuLookup(module))
            moved.append(name)
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def pick_forward(model, requested: str) -> str:
    """``auto``: packed when the mask reaches every layer, else cached (Gemma 4, hybrids)."""
    if requested == "auto":
        return "packed" if full_attention_only(model) else "cached"
    if requested == "packed" and not full_attention_only(model):
        raise ValueError("this model has sliding-window or linear-attention layers: use cached")
    return requested


def lr_at(step: int, total: int, warmup_steps: int) -> float:
    """Multiplier for the learning rate: linear warm-up, then cosine down to zero."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train(model, tokenizer, records: list[DataRecord], cfg: TrainConfig, log_path: Path) -> dict:
    """Train the LoRA parameters of ``model`` (a PeftModel) in place. Returns a summary."""
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    profile = profile_for(tokenizer)
    label_ids = label_token_ids(tokenizer, DEFAULT_LABELS)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    total = math.ceil(len(records) * cfg.epochs / cfg.grad_accum)
    if cfg.max_steps is not None:
        total = min(total, cfg.max_steps)
    warmup_steps = max(1, round(cfg.warmup * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_at(s, total, warmup_steps)
    )

    model.train()
    log = log_path.open("w")
    start = time.perf_counter()
    step = seen = skipped = tokens = 0
    window: list[float] = []
    stop_reason = "done"
    first_plan = total
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(math.ceil(cfg.epochs)):
        order = list(records)
        rng.shuffle(order)
        if epoch + 1 > cfg.epochs:  # fractional last epoch
            order = order[: round(len(order) * (cfg.epochs - epoch))]
        for record in order:
            example = make_example(record, tokenizer, rng, profile)
            if example.n_tokens > cfg.max_tokens:
                skipped += 1
                continue
            loss = example_loss(model, example, label_ids, cfg.forward)
            (loss / cfg.grad_accum).backward()
            window.append(loss.item())
            seen += 1
            tokens += example.n_tokens
            if seen % cfg.grad_accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - start
            if step == PACE_STEPS and cfg.max_minutes is not None:
                affordable = int(cfg.max_minutes * 60 / (elapsed / step))
                if affordable < total:
                    # The scheduler's lambda reads ``total`` when called, so it follows.
                    total = max(step + 1, affordable)
                    print(f"time budget: plan cut to {total} steps", flush=True)
            if step % cfg.log_every == 0 or step == total:
                row = {
                    "step": step,
                    "states": seen,
                    "loss": sum(window) / len(window),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "seconds": elapsed,
                    "tokens": tokens,
                }
                if torch.cuda.is_available():
                    row["max_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    f"step {step}/{total} loss {row['loss']:.4f} lr {row['lr']:.2e} {elapsed:.0f}s",
                    flush=True,
                )
                window = []
            if step >= total:
                break
            if cfg.max_minutes is not None and elapsed > cfg.max_minutes * 60:
                stop_reason = f"time limit ({cfg.max_minutes} min)"
                break
        if step >= total or stop_reason != "done":
            break
    log.close()
    model.eval()
    return {
        "optimizer_steps": step,
        "planned_steps": total,
        "first_planned_steps": first_plan,
        "states_seen": seen,
        "states_skipped_too_long": skipped,
        "tokens": tokens,
        "train_seconds": time.perf_counter() - start,
        "stop_reason": stop_reason,
        "trainable_params": sum(p.numel() for p in params),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


# --------------------------------------------------------------------------------------
# Eval (the zero-shot harness) and calibration
# --------------------------------------------------------------------------------------


def evaluate(
    model, tokenizer, records, out: Path, *, system, run_id, dtype, split="test", forward="packed"
):
    """E's eval stage over ``records`` -> ``out/predictions.jsonl``, ``out/metrics*.json``.
    Returns the predictions."""
    from eval.run_eval import stage_eval

    out.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(system=system, run_id=run_id, forward=forward, dtype=dtype)
    stage_eval(args, model, tokenizer, records, out)
    preds = list(read_jsonl(out / "predictions.jsonl", Prediction))
    if split != "test":  # stage_eval labels its metrics as test
        write_metrics(preds, records, out, system=system, run_id=run_id, split=split)
    return preds


def write_metrics(
    preds, records, out: Path, *, system, run_id, split="test", temperature=None, bias=None
):
    from eval.metrics import compute_all

    out.mkdir(parents=True, exist_ok=True)
    for name, m in compute_all(preds, records, system=system, run_id=run_id, split=split).items():
        if temperature is not None:
            extra = {**m.extra, **{f"{k}_bias": v for k, v in (bias or {}).items()}}
            m = m.model_copy(update={"temperature": temperature, "extra": extra})
        suffix = "" if name == "all" else f"_{name}"
        (out / f"metrics{suffix}.json").write_text(m.model_dump_json(indent=2) + "\n")


def parse_held_out(items: list[str] | None) -> set[tuple[str, str]]:
    """``["customer-support-tickets:priority", ...]`` -> ``{(dataset, question_id), ...}``."""
    out = set()
    for item in items or []:
        dataset, sep, qid = item.partition(":")
        if not (sep and dataset and qid):
            raise ValueError(f"--hold-out wants DATASET:QUESTION_ID, got {item!r}")
        out.add((dataset, qid))
    return out


def drop_questions(records: list[DataRecord], held_out: set[tuple[str, str]]) -> list[DataRecord]:
    """``records`` without the ``held_out`` questions (and their gold answers). A record left
    with no questions is dropped."""
    out = []
    for r in records:
        keep = {q: v for q, v in r.questions.items() if (r.dataset, q) not in held_out}
        if keep:
            out.append(
                r if len(keep) == len(r.questions)
                else r.model_copy(update={"questions": keep, "gold": {q: r.gold[q] for q in keep}})
            )  # fmt: skip
    return out


def load_split(data: str, split: str, prefix: str = "") -> list[DataRecord]:
    """``<data>/<prefix><split>.jsonl``, e.g. ``cfpb_train.jsonl`` with ``prefix="cfpb_"``."""
    return load_records(f"{data.rstrip('/')}/{prefix}{split}.jsonl")


def load_records(path: str) -> list[DataRecord]:
    from eval.common import local_path

    return list(read_jsonl(local_path(path), DataRecord))


def parse_args(argv):
    d = TrainConfig()
    p = argparse.ArgumentParser(prog="python -m snapcall.train_lora")
    p.add_argument("--model", required=True, help="local path (runner: {models}/<org>/<name>)")
    p.add_argument("--data", required=True, help="dir with train/val/test.jsonl, local or gs://")
    p.add_argument(
        "--split-prefix",
        default="",
        help="read <prefix>train/val/test.jsonl instead, e.g. cfpb_ (Decision 16)",
    )
    p.add_argument("--out", default="outputs")
    p.add_argument("--datasets", nargs="+", default=None, help="train+val on these only")
    p.add_argument(
        "--hold-out",
        nargs="+",
        default=None,
        metavar="DATASET:QUESTION",
        help="questions never trained or calibrated on (still scored on test), e.g. "
        "customer-support-tickets:priority (Decision 12 run f)",
    )
    p.add_argument("--tag", default=None, help="system name suffix, e.g. tickets")
    p.add_argument("--dtype", default=None, help="default: bfloat16 on GPU, float32 on CPU")
    p.add_argument("--run-id", default=None, help="default: $SNAPCALL_RUN_ID")
    p.add_argument("--train-limit", type=int, default=None, help="first N train states")
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--test-limit", type=int, default=None)
    p.add_argument("--eval-base", action="store_true", help="also eval the base model first")
    p.add_argument(
        "--ood",
        default=None,
        help="out-of-domain test jsonl (forgetting check), local or gs://; never trained on",
    )
    p.add_argument("--r", type=int, default=d.r)
    p.add_argument("--alpha", type=int, default=d.alpha)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--epochs", type=float, default=d.epochs)
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--warmup", type=float, default=d.warmup)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=d.max_tokens)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument(
        "--forward",
        default="auto",
        choices=["auto", "packed", "cached"],
        help="train and eval through this forward; auto = packed unless the model has "
        "sliding-window or linear-attention layers",
    )
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--log-every", type=int, default=d.log_every)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from eval.common import load_model, model_label, write_json
    from eval.run_eval import round_robin

    # Fragmentation matters when training fills the L4 (Gemma 4); must be set before CUDA starts.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args(argv)
    run_id = args.run_id or os.environ.get("SNAPCALL_RUN_ID")
    dtype = args.dtype or ("bfloat16" if torch.cuda.is_available() else "float32")
    cfg = TrainConfig(
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        warmup=args.warmup,
        max_steps=args.max_steps,
        max_minutes=args.max_minutes,
        max_tokens=args.max_tokens,
        grad_checkpointing=args.grad_checkpointing,
        seed=args.seed,
        log_every=args.log_every,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = model_label(args.model)
    system = f"snapcall-{label}-lora" + (f"-{args.tag}" if args.tag else "")

    held_out = parse_held_out(args.hold_out)

    def pick(records, limit):
        if args.datasets:
            records = [r for r in records if r.dataset in args.datasets]
        records = drop_questions(records, held_out)
        return round_robin(records, limit) if limit else records

    all_train = load_split(args.data, "train", args.split_prefix)
    missing = held_out - {(r.dataset, q) for r in all_train for q in r.questions}
    if missing:  # a typo would otherwise hold out nothing, silently
        raise SystemExit(f"--hold-out names questions not in the training data: {sorted(missing)}")
    train_records = pick(all_train, args.train_limit)
    val_records = pick(load_split(args.data, "val", args.split_prefix), args.val_limit)
    test_records = load_split(args.data, "test", args.split_prefix)
    if args.test_limit:
        test_records = round_robin(test_records, args.test_limit)
    if not train_records or not val_records:
        raise SystemExit(f"no train/val records for datasets {args.datasets}")
    summary: dict = {
        "system": system,
        "run_id": run_id,
        "model": args.model,
        "dtype": dtype,
        "datasets": args.datasets,
        "held_out": sorted(f"{d}:{q}" for d, q in held_out),
        "n_train": len(train_records),
        "n_val": len(val_records),
        "n_test": len(test_records),
        "config": asdict(cfg),
        "seconds": {},
    }
    ood_records = []
    if args.ood:
        # Loaded up front so a bad path shows at once, but never fatal: the forgetting
        # check mustn't cost the training run.
        try:
            ood_records = load_records(args.ood)
            summary["n_ood"] = len(ood_records)
        except Exception as e:
            summary["ood_error"] = f"{type(e).__name__}: {e}"[:500]
            print(f"WARNING: skipping the OOD check: {summary['ood_error']}", flush=True)

    def phase(name, fn):
        t0 = time.perf_counter()
        result = fn()
        summary["seconds"][name] = time.perf_counter() - t0
        write_json(out / "train_summary.json", summary)
        print(f"{name}: {summary['seconds'][name]:.0f}s", flush=True)
        return result

    model, tokenizer = phase("load", lambda: load_model(args.model, dtype))
    summary["cpu_tables"] = offload_lookup_tables(model)
    forward = pick_forward(model, args.forward)
    cfg = replace(cfg, forward=forward)
    summary["config"] = asdict(cfg)

    def run_eval(records, where, name, split="test"):
        return evaluate(
            model, tokenizer, records, where, system=name, run_id=run_id, dtype=dtype,
            split=split, forward=forward,
        )  # fmt: skip

    if args.eval_base:
        base = f"snapcall-{label}-zeroshot"
        phase("eval_base", lambda: run_eval(test_records, out / "base", base))
        if ood_records:
            phase("eval_base_ood", lambda: run_eval(ood_records, out / "base" / "ood", base))

    model = add_lora(model, cfg)
    summary["train"] = phase(
        "train", lambda: train(model, tokenizer, train_records, cfg, out / "train_log.jsonl")
    )
    phase("save", lambda: model.save_pretrained(out / "adapter"))
    # Fold the adapter into the base weights for eval: same answers, no LoRA overhead.
    model = model.merge_and_unload().eval()

    uncal = f"{system}-uncal"
    val_preds = phase("eval_val", lambda: run_eval(val_records, out / "val", uncal, "val"))
    temperatures, bias, report = calibrate.fit(val_preds, val_records)
    write_json(
        out / "temperatures.json", {"temperatures": temperatures, "bias": bias, "report": report}
    )
    summary["temperatures"], summary["bias"] = temperatures, bias

    def calibrated_eval(records, uncal_dir: Path, cal_dir: Path):
        preds = run_eval(records, uncal_dir, uncal)
        calibrated = calibrate.apply(preds, records, temperatures, system, bias)
        cal_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(cal_dir / "predictions.jsonl", calibrated)
        write_metrics(
            calibrated, records, cal_dir, system=system, run_id=run_id,
            temperature=temperatures, bias=bias,
        )  # fmt: skip

    phase("eval_test", lambda: calibrated_eval(test_records, out / "uncalibrated", out))
    if ood_records:
        ood = out / "ood"
        phase("eval_ood", lambda: calibrated_eval(ood_records, ood / "uncalibrated", ood))
    write_json(out / "train_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

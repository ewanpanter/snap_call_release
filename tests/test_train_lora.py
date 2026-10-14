import json
import random

import pytest
import torch

from snapcall import train_lora
from snapcall.masking import separate_forward
from snapcall.prompt import DEFAULT_LABELS, PromptProfile, label_token_ids, option_texts
from snapcall.schema import ChoiceQuestion, DataRecord, ScoreQuestion, YesNoQuestion, read_jsonl
from tests.fixtures.tiny_qwen3 import build_tiny_qwen3
from tests.test_schema import EXAMPLES

PROFILE = PromptProfile()


@pytest.fixture
def records():
    return list(read_jsonl(EXAMPLES / "data.example.jsonl", DataRecord))


@pytest.fixture
def tiny():
    """A fresh tiny model per test: these tests train it."""
    return build_tiny_qwen3()


def test_choice_options_get_a_fresh_random_order():
    q = ChoiceQuestion(question="q", options={k: k for k in "abcdef"})
    rng = random.Random(0)
    orders = {train_lora.shuffled_keys(q, rng) for _ in range(50)}
    assert all(sorted(o) == list("abcdef") for o in orders)
    assert len(orders) > 20


def test_score_levels_are_only_ever_flipped_and_yesno_swapped():
    rng = random.Random(0)
    score = ScoreQuestion(question="q", levels=["low", "medium", "high"])
    assert {train_lora.shuffled_keys(score, rng) for _ in range(30)} == {
        ("0", "1", "2"),
        ("2", "1", "0"),
    }
    yesno = YesNoQuestion(question="q")
    assert {train_lora.shuffled_keys(yesno, rng) for _ in range(30)} == {
        ("yes", "no"),
        ("no", "yes"),
    }


def test_gold_label_points_at_the_gold_option(records, tiny):
    _, tok = tiny
    rng = random.Random(1)
    for record in records:
        for _ in range(5):
            ex = train_lora.make_example(record, tok, rng, PROFILE)
            for (qid, q), branch, gold in zip(
                record.questions.items(), ex.branch_ids, ex.gold, strict=True
            ):
                text = tok.decode(branch)
                want = option_texts(q)[train_lora.calibrate.gold_key(q, record.gold[qid])]
                assert f"\n{DEFAULT_LABELS[gold]}. {want}\n" in text


def test_restricted_loss_is_cross_entropy_over_offered_labels(records, tiny):
    model, tok = tiny
    label_ids = label_token_ids(tok, DEFAULT_LABELS)
    ex = train_lora.make_example(records[0], tok, random.Random(0), PROFILE)
    with torch.no_grad():
        logits = separate_forward(model, ex.prefix_ids, ex.branch_ids)
        got = train_lora.restricted_loss(logits, label_ids, ex)
    want = torch.stack(
        [
            torch.nn.functional.cross_entropy(row[label_ids[:n]][None], torch.tensor([g]))
            for row, n, g in zip(logits, ex.n_options, ex.gold, strict=True)
        ]
    ).mean()
    assert got.item() == pytest.approx(want.item(), rel=1e-5)


def test_packed_loss_and_gradients_match_one_forward_per_question(records):
    """Training through the masked single pass is the same as training on each question
    alone: same loss, same gradients."""
    model, tok = build_tiny_qwen3()
    label_ids = label_token_ids(tok, DEFAULT_LABELS)
    ex = train_lora.make_example(records[0], tok, random.Random(0), PROFILE)

    def loss_and_grad(forward):
        model.zero_grad()
        loss = train_lora.restricted_loss(
            forward(model, ex.prefix_ids, ex.branch_ids), label_ids, ex
        )
        loss.backward()
        return loss.item(), model.model.layers[0].self_attn.q_proj.weight.grad.clone()

    packed_loss, packed_grad = loss_and_grad(train_lora.FORWARDS["packed"])
    sep_loss, sep_grad = loss_and_grad(separate_forward)
    assert packed_loss == pytest.approx(sep_loss, rel=1e-5)
    torch.testing.assert_close(packed_grad, sep_grad, rtol=1e-4, atol=1e-6)


def test_lr_schedule_warms_up_then_decays_to_zero():
    lrs = [train_lora.lr_at(s, 100, 10) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:], strict=False))
    assert lrs[-1] < 0.01


def test_lora_can_fit_a_fixed_example(records, tiny):
    """The gradient reaches the adapter through packed_forward: one fixed view of a state,
    trained on repeatedly, is learned."""
    model, tok = tiny
    model = train_lora.add_lora(model, train_lora.TrainConfig(dropout=0.0))
    label_ids = label_token_ids(tok, DEFAULT_LABELS)
    ex = train_lora.make_example(records[0], tok, random.Random(0), PROFILE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(25):
        loss = train_lora.example_loss(model, ex, label_ids)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_training_moves_only_lora(records, tiny, tmp_path):
    model, tok = tiny
    cfg = train_lora.TrainConfig(lr=5e-3, epochs=2, grad_accum=1, log_every=1)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    model = train_lora.add_lora(model, cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)

    summary = train_lora.train(model, tok, records, cfg, tmp_path / "log.jsonl")
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert summary["optimizer_steps"] == len(records) * 2 == len(log)
    assert summary["stop_reason"] == "done" and not model.training
    for n, p in model.get_base_model().named_parameters():
        if "lora_" not in n:
            torch.testing.assert_close(p, base[n.replace(".base_layer", "")])
    assert any(
        not torch.equal(p, torch.zeros_like(p))
        for n, p in model.named_parameters()
        if "lora_B" in n
    )


def test_training_stops_at_max_steps(records, tiny, tmp_path):
    model, tok = tiny
    cfg = train_lora.TrainConfig(epochs=10, grad_accum=2, max_steps=3)
    summary = train_lora.train(
        train_lora.add_lora(model, cfg), tok, records, cfg, tmp_path / "log.jsonl"
    )
    assert summary["optimizer_steps"] == summary["planned_steps"] == 3
    assert summary["states_seen"] == 6


def test_too_long_states_are_skipped(records, tiny, tmp_path):
    model, tok = tiny
    cfg = train_lora.TrainConfig(max_tokens=10, grad_accum=1)
    summary = train_lora.train(
        train_lora.add_lora(model, cfg), tok, records, cfg, tmp_path / "log.jsonl"
    )
    assert summary["states_skipped_too_long"] == len(records)
    assert summary["optimizer_steps"] == 0


# --------------------------------------------------------------------------------------
# End to end with the eval harness (skipped if eval/ isn't importable)
# --------------------------------------------------------------------------------------


def test_main_end_to_end(records, tmp_path):
    pytest.importorskip("eval.run_eval")
    from snapcall.schema import Metrics, Prediction, write_jsonl

    model, tok = build_tiny_qwen3()
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val", "test"):
        rows = [r.model_copy(update={"split": split}) for r in records]
        write_jsonl(data / f"{split}.jsonl", rows)
    out = tmp_path / "out"
    rc = train_lora.main(
        [
            "--model", str(tmp_path / "model"), "--data", str(data), "--out", str(out),
            "--max-steps", "2", "--grad-accum", "1", "--eval-base", "--tag", "t",
            "--ood", str(data / "test.jsonl"),
        ]
    )  # fmt: skip
    assert rc == 0
    preds = list(read_jsonl(out / "predictions.jsonl", Prediction))
    assert {p.system for p in preds} == {"snapcall-model-lora-t"}
    assert len(preds) == sum(len(r.questions) for r in records)
    metrics = Metrics.model_validate_json((out / "metrics.json").read_text())
    assert metrics.temperature and metrics.split == "test"
    val = Metrics.model_validate_json((out / "val" / "metrics.json").read_text())
    assert val.split == "val"
    for f in ("adapter/adapter_config.json", "temperatures.json", "train_summary.json"):
        assert (out / f).exists(), f
    assert (out / "base" / "metrics.json").exists()
    assert (out / "base" / "ood" / "metrics.json").exists()
    ood = Metrics.model_validate_json((out / "ood" / "metrics.json").read_text())
    assert ood.system == "snapcall-model-lora-t" and (out / "ood" / "uncalibrated").is_dir()
    summary = json.loads((out / "train_summary.json").read_text())
    assert summary["train"]["optimizer_steps"] == 2
    assert set(summary["bias"]) <= {"yesno"}


def test_missing_ood_file_does_not_stop_training(records, tmp_path):
    from snapcall.schema import write_jsonl

    model, tok = build_tiny_qwen3()
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val", "test"):
        write_jsonl(
            data / f"{split}.jsonl", [r.model_copy(update={"split": split}) for r in records]
        )
    out = tmp_path / "out"
    args = ["--model", str(tmp_path / "model"), "--data", str(data), "--out", str(out)]
    assert train_lora.main([*args, "--max-steps", "1", "--ood", str(data / "nope.jsonl")]) == 0
    summary = json.loads((out / "train_summary.json").read_text())
    assert "ood_error" in summary and not (out / "ood").exists()


def test_time_budget_shrinks_the_plan_so_the_schedule_finishes(
    records, tiny, tmp_path, monkeypatch
):
    """Each optimizer step 'takes' 1 s; a 30 s budget cuts the plan to 30 steps after the
    pacing steps, and training ends normally with the learning rate decayed."""
    clock = iter(range(10_000))
    monkeypatch.setattr(train_lora.time, "perf_counter", lambda: float(next(clock)))
    model, tok = tiny
    cfg = train_lora.TrainConfig(epochs=100, grad_accum=1, max_minutes=0.5, log_every=1)
    summary = train_lora.train(
        train_lora.add_lora(model, cfg), tok, records, cfg, tmp_path / "log.jsonl"
    )
    assert summary["first_planned_steps"] == 100 * len(records)
    assert summary["planned_steps"] == summary["optimizer_steps"] == 30
    assert summary["stop_reason"] == "done"
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert log[-1]["lr"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# Sliding-window models (Gemma 4) train through the cached forward
# --------------------------------------------------------------------------------------


def tiny_gemma4(vocab_size: int):
    """2 sliding + 2 full-attention layers, the last two sharing earlier layers' KV (as in
    Gemma 4 E4B), with a window shorter than the test prompts. Random weights, built in code."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"] * 2,
        vocab_size_per_layer_input=vocab_size,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=2,
        pad_token_id=0,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Gemma4ForCausalLM(cfg).float()


@pytest.fixture
def gemma(tiny):
    _, tok = tiny
    return tiny_gemma4(len(tok)), tok


def test_cached_loss_and_gradients_match_one_forward_per_question(records, gemma):
    """Every parameter's gradient through the copied caches matches one forward per question
    (to float32 noise: ~1e-4 of the gradient's norm)."""
    model, tok = gemma
    label_ids = label_token_ids(tok, DEFAULT_LABELS)
    ex = train_lora.make_example(records[0], tok, random.Random(0), PROFILE)
    assert len(ex.prefix_ids) > model.config.sliding_window  # the window really bites

    def loss_and_grads(forward):
        model.zero_grad()
        loss = train_lora.restricted_loss(
            forward(model, ex.prefix_ids, ex.branch_ids), label_ids, ex
        )
        loss.backward()
        return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()}

    cached_loss, cached = loss_and_grads(train_lora.FORWARDS["cached"])
    sep_loss, sep = loss_and_grads(separate_forward)
    assert cached_loss == pytest.approx(sep_loss, rel=1e-5)
    for name, g in sep.items():
        if g.norm() > 0:
            assert (cached[name] - g).norm() / g.norm() < 1e-3, name


def test_forward_is_picked_from_the_architecture(tiny, gemma):
    assert train_lora.pick_forward(tiny[0], "auto") == "packed"
    assert train_lora.pick_forward(gemma[0], "auto") == "cached"
    with pytest.raises(ValueError, match="cached"):
        train_lora.pick_forward(gemma[0], "packed")


def test_lora_targets_skip_vision_and_audio_towers():
    import re

    hit = re.compile(train_lora.LORA_TARGET_RE).fullmatch
    assert hit("model.language_model.layers.3.self_attn.q_proj")
    assert hit("model.layers.0.mlp.down_proj")
    assert not hit("model.vision_tower.encoder.layers.0.self_attn.q_proj")
    assert not hit("model.audio_tower.layers.1.mlp.up_proj")
    assert not hit("model.layers.0.self_attn.q_norm")


def test_gemma_trains_through_the_cached_forward(records, gemma, tmp_path):
    model, tok = gemma
    cfg = train_lora.TrainConfig(forward="cached", grad_accum=1, epochs=2, log_every=1)
    model = train_lora.add_lora(model, cfg)
    summary = train_lora.train(model, tok, records, cfg, tmp_path / "log.jsonl")
    assert summary["optimizer_steps"] == 2 * len(records)
    assert any(
        not torch.equal(p, torch.zeros_like(p))
        for n, p in model.named_parameters()
        if "lora_B" in n
    )


def test_gradient_checkpointing_is_refused_with_the_cached_forward(gemma):
    cfg = train_lora.TrainConfig(forward="cached", grad_checkpointing=True)
    with pytest.raises(ValueError, match="checkpointing"):
        train_lora.add_lora(gemma[0], cfg)


def test_main_end_to_end_on_a_sliding_window_model(records, gemma, tmp_path):
    from snapcall.schema import Metrics, write_jsonl

    model, tok = gemma
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val", "test"):
        write_jsonl(
            data / f"{split}.jsonl", [r.model_copy(update={"split": split}) for r in records]
        )
    out = tmp_path / "out"
    args = ["--model", str(tmp_path / "model"), "--data", str(data), "--out", str(out)]
    assert train_lora.main([*args, "--max-steps", "2", "--grad-accum", "1"]) == 0
    summary = json.loads((out / "train_summary.json").read_text())
    assert summary["config"]["forward"] == "cached"
    assert summary["cpu_tables"] == ["model.embed_tokens_per_layer"]
    timings = json.loads((out / "uncalibrated" / "eval_timings.json").read_text())
    assert timings["forwards"] == {"cached": len(records)}
    assert Metrics.model_validate_json((out / "metrics.json").read_text()).temperature


def test_offloading_the_per_layer_table_changes_nothing(records, gemma):
    model, tok = gemma
    ex = train_lora.make_example(records[0], tok, random.Random(0), PROFILE)
    forward = train_lora.FORWARDS["cached"]
    with torch.no_grad():
        before = forward(model, ex.prefix_ids, ex.branch_ids)
        assert train_lora.offload_lookup_tables(model) == ["model.embed_tokens_per_layer"]
        assert isinstance(model.model.embed_tokens_per_layer, train_lora.CpuLookup)
        after = forward(model, ex.prefix_ids, ex.branch_ids)
    torch.testing.assert_close(after, before)
    assert train_lora.offload_lookup_tables(build_tiny_qwen3()[0]) == []


# --------------------------------------------------------------------------------------
# Phi-4-mini (Decision 11): fused projections, a sliding window longer than any input
# --------------------------------------------------------------------------------------


def tiny_phi3(vocab_size: int):
    """Phi3ForCausalLM as Phi-4-mini configures it (fused qkv / gate_up, partial rotary,
    sliding_window=262144), 2 layers, random weights."""
    from transformers import Phi3Config, Phi3ForCausalLM

    cfg = Phi3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        partial_rotary_factor=0.75,
        sliding_window=262144,
        tie_word_embeddings=True,
        pad_token_id=0,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Phi3ForCausalLM(cfg).float()


def test_phi_lora_adapts_the_fused_projections(tiny):
    model = train_lora.add_lora(tiny_phi3(len(tiny[1])), train_lora.TrainConfig())
    adapted = {n.split(".")[-1] for n, m in model.named_modules() if hasattr(m, "lora_A")}
    assert adapted == {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}


def test_phi_uses_the_packed_forward_and_it_is_exact(records, tiny):
    """A window longer than any input is full attention: packed is picked, and its loss and
    gradients match one forward per question."""
    _, tok = tiny
    model = tiny_phi3(len(tok))
    assert train_lora.pick_forward(model, "auto") == "packed"
    label_ids = label_token_ids(tok, DEFAULT_LABELS)
    ex = train_lora.make_example(records[0], tok, random.Random(0), PROFILE)

    def loss_and_grads(forward):
        model.zero_grad()
        loss = train_lora.restricted_loss(
            forward(model, ex.prefix_ids, ex.branch_ids), label_ids, ex
        )
        loss.backward()
        return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()}

    packed_loss, packed = loss_and_grads(train_lora.FORWARDS["packed"])
    sep_loss, sep = loss_and_grads(separate_forward)
    assert packed_loss == pytest.approx(sep_loss, rel=1e-5)
    for name, g in sep.items():
        if g.norm() > 0:
            assert (packed[name] - g).norm() / g.norm() < 1e-3, name


# --------------------------------------------------------------------------------------
# Held-out questions (Decision 12 run f)
# --------------------------------------------------------------------------------------


def test_held_out_questions_are_removed_with_their_gold(records):
    ds = records[0].dataset
    qids = list(records[0].questions)
    held = train_lora.parse_held_out([f"{ds}:{qids[0]}"])
    out = train_lora.drop_questions(records, held)
    for r in out:
        if r.dataset == ds:
            assert qids[0] not in r.questions and qids[0] not in r.gold
            assert set(r.questions) == set(r.gold)
    everything = train_lora.parse_held_out([f"{ds}:{q}" for q in qids])
    assert all(r.dataset != ds for r in train_lora.drop_questions(records, everything))


def test_held_out_names_must_be_dataset_colon_question():
    with pytest.raises(ValueError, match="DATASET:QUESTION_ID"):
        train_lora.parse_held_out(["priority"])


def test_main_holds_questions_out_of_training_and_calibration_only(records, tmp_path):
    from snapcall.schema import Prediction, write_jsonl

    model, tok = build_tiny_qwen3()
    model.save_pretrained(tmp_path / "model")
    tok.save_pretrained(tmp_path / "model")
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val", "test"):
        write_jsonl(
            data / f"{split}.jsonl", [r.model_copy(update={"split": split}) for r in records]
        )
    ds, qid = records[0].dataset, next(iter(records[0].questions))
    out = tmp_path / "out"
    args = ["--model", str(tmp_path / "model"), "--data", str(data), "--out", str(out)]
    assert train_lora.main([*args, "--max-steps", "1", "--hold-out", f"{ds}:{qid}"]) == 0
    val = list(read_jsonl(out / "val" / "predictions.jsonl", Prediction))
    test = list(read_jsonl(out / "predictions.jsonl", Prediction))
    assert val and all(p.question_id != qid for p in val)
    assert any(p.question_id == qid for p in test)
    assert json.loads((out / "train_summary.json").read_text())["held_out"] == [f"{ds}:{qid}"]
    with pytest.raises(SystemExit, match="not in the training data"):
        train_lora.main([*args, "--max-steps", "1", "--hold-out", f"{ds}:nope"])


def test_split_prefix_reads_other_files(records, tmp_path):
    from snapcall.schema import write_jsonl

    write_jsonl(tmp_path / "cfpb_val.jsonl", records[:1])
    got = train_lora.load_split(str(tmp_path), "val", "cfpb_")
    assert [r.state_id for r in got] == [records[0].state_id]

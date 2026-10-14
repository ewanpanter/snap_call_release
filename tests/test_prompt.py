import copy
import json
from pathlib import Path

import pytest

from snapcall.prompt import (
    DEFAULT_LABELS,
    DEFAULT_PROFILE,
    EMPTY_THINK,
    SYSTEM_PROMPT,
    ChatFormat,
    PromptProfile,
    build_prompt,
    encode_prompt,
    label_token_ids,
    n_permutations,
    permutations,
    profile_for,
)
from snapcall.schema import ChoiceQuestion, DecideRequest, ScoreQuestion, YesNoQuestion

EXAMPLE = Path(__file__).parent.parent / "contract" / "examples" / "decide_request.example.json"


@pytest.fixture
def request_example() -> DecideRequest:
    return DecideRequest.model_validate_json(EXAMPLE.read_text())


def choice(n: int) -> ChoiceQuestion:
    return ChoiceQuestion(question="Pick one", options={f"k{i}": f"option {i}" for i in range(n)})


SCORE = ScoreQuestion(question="How urgent?", levels=["low", "medium", "high"])
YESNO = YesNoQuestion(question="Is it billing?")


def is_rotation(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and any(a[i:] + a[:i] == b for i in range(len(a)))


@pytest.mark.parametrize(
    ("question", "debias", "expected"),
    [
        (choice(3), "auto", 3),  # min(n_options, 4)
        (choice(6), "auto", 4),
        (choice(6), "off", 1),
        (choice(6), 2, 2),
        (choice(3), 8, 3),  # can't have more cyclic shifts than options
        (SCORE, "auto", 2),
        (SCORE, 8, 2),
        (YESNO, "auto", 2),
        (YESNO, "off", 1),
    ],
)
def test_n_permutations(question, debias, expected):
    assert n_permutations(question, debias) == expected
    assert len(permutations(question, debias)) == expected


def test_choice_permutations_are_distinct_cyclic_shifts():
    q = choice(6)
    perms = permutations(q, 4)
    original = tuple(q.options)
    assert perms[0] == original  # seed 0 starts from the original order
    assert len(set(perms)) == 4
    assert all(is_rotation(original, p) for p in perms)


def test_all_cyclic_shifts_put_every_option_at_every_label():
    q = choice(4)
    perms = permutations(q, 4)
    for position in range(4):
        assert {p[position] for p in perms} == set(q.options)


def test_seed_changes_choice_order_deterministically():
    q = choice(5)
    assert permutations(q, 2, seed=1) == permutations(q, 2, seed=1)
    assert permutations(q, 2, seed=1) != permutations(q, 2, seed=0)


def test_score_is_flipped_never_shuffled():
    assert permutations(SCORE, "auto") == [("0", "1", "2"), ("2", "1", "0")]


def test_yesno_swaps_order():
    assert permutations(YESNO, "auto") == [("yes", "no"), ("no", "yes")]


def test_branch_text_labels_match_keys(request_example):
    prompt = build_prompt(request_example)
    queue = [b for b in prompt.branches if b.question_id == "queue"]
    assert len(queue) == 3
    second = queue[1]
    assert second.keys == ("technical", "sales", "billing")
    assert "A. Technical Support\nB. Sales and Pre-Sales\nC. Billing and Payments" in second.text
    assert "Reply with one of: A, B, C." in second.text


def test_score_branch_says_which_way_the_scale_runs(request_example):
    prompt = build_prompt(request_example)
    up, down = [b for b in prompt.branches if b.question_id == "priority"]
    assert "lowest to highest" in up.text and "A. low" in up.text
    assert "highest to lowest" in down.text and "A. critical" in down.text


def test_prompt_layout(request_example):
    prompt = build_prompt(request_example)
    assert prompt.prefix.startswith("<|im_start|>system\n")
    assert request_example.state in prompt.prefix
    assert prompt.prefix.endswith("</state>\n")
    for b in prompt.branches:
        assert b.text.startswith("Question: ")
        assert b.text.endswith("<|im_end|>\n<|im_start|>assistant\n")  # default profile
    # question order is kept, permutations of a question are adjacent
    assert [b.question_id for b in prompt.branches] == ["queue"] * 3 + ["priority"] * 2 + [
        "is_billing"
    ] * 2


def test_no_think_tokens_unless_the_profile_needs_the_empty_stub(request_example):
    """We build the prompt by hand, so there are no <think> tokens, except the
    model's own empty non-thinking block when its profile asks for it."""
    plain = build_prompt(request_example)
    text = plain.prefix + "".join(b.text for b in plain.branches)
    assert "<think>" not in text and "</think>" not in text

    stub = build_prompt(request_example, profile=PromptProfile(empty_think=True))
    assert "<think>" not in stub.prefix
    for b in stub.branches:
        assert b.text.endswith("<|im_start|>assistant\n" + EMPTY_THINK)
        assert b.text.count("<think>") == 1


def test_profile_for_tokenizer_without_chat_template(tiny_qwen3):
    _, tok = tiny_qwen3
    assert profile_for(tok) == PromptProfile(empty_think=False)


def test_json_state_is_serialised_deterministically():
    q = {"q": YESNO}
    a = build_prompt(DecideRequest(state={"b": 1, "a": "x"}, questions=q))
    b = build_prompt(DecideRequest(state={"a": "x", "b": 1}, questions=q))
    assert a.prefix == b.prefix
    assert json.dumps({"a": "x", "b": 1}) in a.prefix


def test_yesno_descriptions_are_shown():
    q = YesNoQuestion(question="Refund?", yes="asks for money back", no="anything else")
    prompt = build_prompt(DecideRequest(state="s", questions={"r": q}, debias="off"))
    assert "A. Yes: asks for money back\nB. No: anything else" in prompt.branches[0].text


def test_too_many_options_for_labels():
    request = DecideRequest(state="s", questions={"q": choice(5)})
    with pytest.raises(ValueError, match="only 4 labels"):
        build_prompt(request, labels=("A", "B", "C", "D"))


def test_multi_token_label_is_rejected(tiny_qwen3):
    _, tok = tiny_qwen3
    with pytest.raises(ValueError, match="'yes' is 3 tokens"):
        label_token_ids(tok, ("yes", "no"))


def assert_split_tokenisation_matches(
    tok, request: DecideRequest, profile: PromptProfile = DEFAULT_PROFILE
):
    """Tokenising prefix and branch separately must equal tokenising them joined, or the
    packed forward would read different tokens than a normal prompt."""
    prompt = build_prompt(request, profile=profile)
    enc = encode_prompt(prompt, tok)
    for branch, ids in zip(prompt.branches, enc.branch_ids, strict=True):
        joined = tok.encode(prompt.prefix + branch.text, add_special_tokens=False)
        assert enc.prefix_ids + ids == joined


def test_split_tokenisation_tiny(tiny_qwen3, request_example):
    _, tok = tiny_qwen3
    assert_split_tokenisation_matches(tok, request_example)


@pytest.mark.real_weights
def test_split_tokenisation_real(real_qwen3, request_example):
    _, tok = real_qwen3
    assert_split_tokenisation_matches(tok, request_example)
    json_state = DecideRequest(state={"subject": "Refund", "body": "..."}, questions={"q": YESNO})
    assert_split_tokenisation_matches(tok, json_state)


@pytest.mark.real_weights
def test_real_default_labels_are_single_tokens(real_qwen3):
    _, tok = real_qwen3
    ids = label_token_ids(tok, DEFAULT_LABELS)
    assert len(set(ids)) == len(ids)


@pytest.mark.real_weights
def test_real_profile_matches_chat_template(real_qwen3, request_example):
    """Qwen3-0.6B is hybrid-thinking, like Qwen3-8B: its template's non-thinking answer
    starts after an empty think block, and our branches must end the same way."""
    _, tok = real_qwen3
    profile = profile_for(tok)
    assert profile.empty_think
    generation_prompt = tok.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ).split("x<|im_end|>\n")[-1]
    assert generation_prompt == "<|im_start|>assistant\n" + EMPTY_THINK
    for branch in build_prompt(request_example, profile=profile).branches:
        assert branch.text.endswith("<|im_end|>\n" + generation_prompt)
    stub_request = request_example.model_copy(update={"debias": "off"})
    assert_split_tokenisation_matches(tok, stub_request, profile)


# --- other chat formats --------------------------------------------------------------

# Minimal stand-ins for real templates: same output shape for system + user + generation.
CHATML_THINK_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n"
    "{% endif %}{% endif %}"
)
GEMMA4_TEMPLATE = (  # as rendered by google/gemma-4-E4B-it (thinking off)
    "<bos>{% for m in messages %}<|turn>{{ 'model' if m.role == 'assistant' else m.role }}\n"
    "{{ m.content }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}"
)


def with_template(tok, template):
    tok = copy.deepcopy(tok)
    tok.chat_template = template
    return tok


def test_profile_for_chatml_keeps_qwen_prompts_unchanged(tiny_qwen3, request_example):
    tok = with_template(tiny_qwen3[1], CHATML_THINK_TEMPLATE)
    profile = profile_for(tok)
    assert profile == PromptProfile(empty_think=True)  # no custom markers: plain ChatML
    prompt = build_prompt(request_example, profile=profile)
    assert prompt.prefix.startswith("<|im_start|>system\n")
    assert all(b.text.endswith("<|im_start|>assistant\n" + EMPTY_THINK) for b in prompt.branches)


def test_profile_for_gemma4_format(tiny_qwen3, request_example):
    tok = with_template(tiny_qwen3[1], GEMMA4_TEMPLATE)
    profile = profile_for(tok)
    assert profile.chat == ChatFormat(
        head="<bos><|turn>system\n",
        mid="<turn|>\n<|turn>user\n",
        tail="<turn|>\n<|turn>model\n",
    )
    assert not profile.empty_think
    prompt = build_prompt(request_example, profile=profile)
    assert prompt.prefix.startswith("<bos><|turn>system\n" + SYSTEM_PROMPT + "<turn|>\n")
    assert "<|im_start|>" not in prompt.prefix
    for b in prompt.branches:
        assert b.text.endswith(".<turn|>\n<|turn>model\n")
        assert "<|im_" not in b.text
    # The prompt is what the template itself renders for the same system + user text.
    first = prompt.branches[0]
    user = prompt.prefix.split("<turn|>\n<|turn>user\n", 1)[1] + first.text.removesuffix(
        "<turn|>\n<|turn>model\n"
    )
    rendered = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered == prompt.prefix + first.text

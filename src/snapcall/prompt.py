r"""Prompt layout: one shared prefix (the state) plus one branch per question permutation.

A request becomes::

    prefix   = <|im_start|>system ... <|im_end|>
               <|im_start|>user
               <state>
               ...the state text...
               </state>
    branch 1 = Question: Which queue should handle this ticket?
               Options:
               A. Billing and Payments
               B. Technical Support
               Reply with one of: A, B.<|im_end|>
               <|im_start|>assistant
               [<think>\n\n</think>\n\n   only if the model's profile needs it, see below]
    branch 2 = (same question, options shown in a different order)
    ...

The model reads the prefix once. Each branch is a question (in one option order) that the
model reads as if it were the only one; ``masking.py`` makes that true. The next-token
logits at the end of a branch (just after ``assistant``) are where the answer label would be
generated, and ``readout.py`` reads them there.

We build the prompt by hand instead of using the chat template, and nothing is ever
generated. The layout above is Qwen's ChatML. Models with another chat format (Gemma 4:
``<|turn>user ... <turn|>``) get the same text wrapped in their own markers, which
``profile_for`` reads from the model's chat template (``ChatFormat``). The other
model-specific detail is the *empty thinking block*. Hybrid-thinking
models such as Qwen3-8B write ``<think>\n\n</think>\n\n`` straight after ``assistant`` when
thinking is switched off; that is their documented non-thinking format, not reasoning. For
those models the answer label comes after that block, so we add it too (``PromptProfile``).
Models without a thinking mode, such as Qwen3-4B-Instruct-2507, get no ``<think>`` tokens.

Every question type is shown as labelled options, so the readout is the same for all three:

- ``choice``: the options, in ``K`` cyclic shifts (debiasing).
- ``score``: the levels, low to high and (if debiasing) also high to low. Never shuffled,
  because the order is the meaning.
- ``yesno``: two options, ``Yes`` and ``No``, and (if debiasing) also ``No`` then ``Yes``.

A ``Branch`` records which answer key sits behind each label, so the readout can map label
probabilities back to answer keys and average over permutations.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from snapcall.schema import (
    ChoiceQuestion,
    Debias,
    DecideRequest,
    ScoreQuestion,
    YesNoQuestion,
    answer_keys,
    state_text,
)

# Fixed letter labels. Don't randomise the alphabet per request.
DEFAULT_LABELS: tuple[str, ...] = tuple(string.ascii_uppercase)

SYSTEM_PROMPT = (
    "You make quick decisions about the state given by the user. "
    "For each question, reply with the label of exactly one option."
)

# What hybrid-thinking Qwen3 models emit after "assistant\n" when thinking is off.
EMPTY_THINK = "<think>\n\n</think>\n\n"

# Hard cap on permutations per question, whatever the request asks for.
MAX_PERMUTATIONS = 8

AnyQuestion = ChoiceQuestion | ScoreQuestion | YesNoQuestion


@dataclass(frozen=True)
class Branch:
    """One question, shown with its options in one order.

    ``keys[i]`` is the answer key behind the i-th label: with labels ``A, B, C`` and
    ``keys = ("sales", "billing", "technical")``, the model answering ``B`` means ``billing``.
    """

    question_id: str
    keys: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class ChatFormat:
    """A chat format's fixed text around the system prompt and the user turn.

    prompt = ``head`` + system prompt + ``mid`` + user text + ``tail``, and the answer label
    is the next token. ``tail`` ends the user turn and opens the assistant's.
    """

    head: str
    mid: str
    tail: str


# Qwen's ChatML. The default, so Qwen prompts are exactly as before ChatFormat existed.
CHATML = ChatFormat(
    head="<|im_start|>system\n",
    mid="<|im_end|>\n<|im_start|>user\n",
    tail="<|im_end|>\n<|im_start|>assistant\n",
)


@dataclass(frozen=True)
class PromptProfile:
    """Per-model prompt details.

    ``empty_think``: end every branch with the model's own empty thinking block, so the
    answer position is where the model expects its answer. ``profile_for`` picks this from
    the model's chat template; pass a profile explicitly to compare with and without.
    ``chat``: the chat markers; ``None`` means ChatML (Qwen). For other formats
    ``chat.tail`` already includes any empty thinking block, and ``empty_think`` only
    reports it.
    """

    empty_think: bool = False
    chat: ChatFormat | None = None

    def tail(self) -> str:
        if self.chat is not None:
            return self.chat.tail
        return CHATML.tail + (EMPTY_THINK if self.empty_think else "")

    def markers(self) -> ChatFormat:
        return self.chat or CHATML


_SYSTEM_SLOT, _USER_SLOT = "SNAPCALLSYSTEMSLOT", "SNAPCALLUSERSLOT"


def chat_format_from_template(tokenizer) -> ChatFormat:
    """Read ``head``/``mid``/``tail`` off the tokenizer's chat template by rendering
    placeholder messages (thinking off) and splitting around them."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": _SYSTEM_SLOT}, {"role": "user", "content": _USER_SLOT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # ignored by templates that have no thinking mode
    )
    head, rest = rendered.split(_SYSTEM_SLOT)
    mid, tail = rest.split(_USER_SLOT)
    return ChatFormat(head=head, mid=mid, tail=tail)


def profile_for(tokenizer) -> PromptProfile:
    """Ask the model's own chat template how a non-thinking answer is laid out."""
    if not getattr(tokenizer, "chat_template", None):
        return PromptProfile()
    chat = chat_format_from_template(tokenizer)
    empty_think = chat.tail.endswith(EMPTY_THINK)
    if chat.head.endswith(CHATML.head) and chat.mid == CHATML.mid:
        return PromptProfile(empty_think=empty_think)  # ChatML: Qwen's prompts, unchanged
    return PromptProfile(empty_think=empty_think, chat=chat)


DEFAULT_PROFILE = PromptProfile()


@dataclass(frozen=True)
class Prompt:
    prefix: str
    branches: list[Branch]
    labels: tuple[str, ...]


# --------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------


def prefix_text(state, profile: PromptProfile | None = None) -> str:
    """The shared part: system prompt and the state. Ends with a newline, so a branch
    always starts on a fresh line and tokenising prefix and branch separately gives the
    same tokens as tokenising them together."""
    chat = (profile or DEFAULT_PROFILE).markers()
    return f"{chat.head}{SYSTEM_PROMPT}{chat.mid}<state>\n{state_text(state)}\n</state>\n"


def option_texts(q: AnyQuestion) -> dict[str, str]:
    """What the model sees for each answer key."""
    if isinstance(q, ChoiceQuestion):
        return {k: (v.strip() or k) for k, v in q.options.items()}
    if isinstance(q, ScoreQuestion):
        return {str(i): level for i, level in enumerate(q.levels)}
    yes = f"Yes: {q.yes}" if q.yes else "Yes"
    no = f"No: {q.no}" if q.no else "No"
    return {"yes": yes, "no": no}


def branch_text(
    q: AnyQuestion,
    keys: tuple[str, ...],
    labels: tuple[str, ...],
    profile: PromptProfile = DEFAULT_PROFILE,
) -> str:
    texts = option_texts(q)
    question = q.question.strip()
    if isinstance(q, ScoreQuestion):
        direction = "lowest to highest" if keys[0] == "0" else "highest to lowest"
        question += f" (options run from {direction})"
    lines = [f"Question: {question}", "Options:"]
    lines += [f"{label}. {texts[key]}" for label, key in zip(labels, keys, strict=False)]
    used = ", ".join(labels[: len(keys)])
    lines.append(f"Reply with one of: {used}.")
    return "\n".join(lines) + profile.tail()


# --------------------------------------------------------------------------------------
# Permutations (debiasing)
# --------------------------------------------------------------------------------------


def n_permutations(q: AnyQuestion, debias: Debias) -> int:
    """How many orderings of ``q`` to ask. ``"auto"`` is the default per type."""
    n_options = len(answer_keys(q))
    # choice can use up to n cyclic shifts; score and yesno have just two orders.
    most = n_options if isinstance(q, ChoiceQuestion) else 2
    if debias == "off":
        return 1
    auto = 4 if isinstance(q, ChoiceQuestion) else 2
    wanted = auto if debias == "auto" else int(debias)
    return max(1, min(wanted, most, MAX_PERMUTATIONS))


def permutations(q: AnyQuestion, debias: Debias, seed: int = 0) -> list[tuple[str, ...]]:
    """The option orders to ask, each a tuple of answer keys (label A first).

    ``choice``: cyclic shifts, spread evenly over the n possible shifts and offset by
    ``seed``. Cyclic shifts move every option through every label *and* position, so
    averaging over all n cancels both kinds of bias. With ``seed=0``
    the first permutation is the original order.
    ``score``/``yesno``: the canonical order, then its reverse.
    """
    keys = tuple(answer_keys(q))
    k = n_permutations(q, debias)
    if not isinstance(q, ChoiceQuestion):
        return [keys, keys[::-1]][:k]
    n = len(keys)
    shifts = [(seed + (i * n) // k) % n for i in range(k)]
    return [keys[s:] + keys[:s] for s in shifts]


# --------------------------------------------------------------------------------------
# The whole prompt
# --------------------------------------------------------------------------------------


def build_prompt(
    request: DecideRequest,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    profile: PromptProfile = DEFAULT_PROFILE,
) -> Prompt:
    """Lay out ``request`` as a shared prefix plus branches (questions x permutations)."""
    branches = []
    for qid, q in request.questions.items():
        if len(answer_keys(q)) > len(labels):
            raise ValueError(f"{qid}: {len(answer_keys(q))} options but only {len(labels)} labels")
        for keys in permutations(q, request.debias, request.seed):
            branches.append(Branch(qid, keys, branch_text(q, keys, labels, profile)))
    return Prompt(prefix_text(request.state, profile), branches, labels)


@dataclass(frozen=True)
class EncodedPrompt:
    prefix_ids: list[int]
    branch_ids: list[list[int]]
    label_ids: list[int]  # token id of each label, in label order

    @property
    def n_tokens(self) -> int:
        """Tokens the model reads: the prefix once, plus every branch."""
        return len(self.prefix_ids) + sum(len(b) for b in self.branch_ids)


def label_token_ids(tokenizer, labels: tuple[str, ...]) -> list[int]:
    """Token id for each label. Raises if a label isn't exactly one token."""
    ids = []
    for label in labels:
        toks = tokenizer.encode(label, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"label {label!r} is {len(toks)} tokens, need exactly 1")
        ids.append(toks[0])
    return ids


def encode_prompt(prompt: Prompt, tokenizer) -> EncodedPrompt:
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return EncodedPrompt(
        prefix_ids=encode(prompt.prefix),
        branch_ids=[encode(b.text) for b in prompt.branches],
        label_ids=label_token_ids(tokenizer, prompt.labels),
    )

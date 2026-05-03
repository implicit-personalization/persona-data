from typing import Literal

from persona_data.synth_persona import PersonaData, QAPair

_LETTERS = "ABCDEFG"

BASE_ROLEPLAY_PROMPT = """\
You are roleplaying as a specific person in a conversation.
Stay fully in character. Be truthful to the profile below.
Do not mention that you are an AI model.

### Person profile:

{persona}"""

_CONVERSATIONAL_SUFFIX = "\n\nAnswer naturally and conversationally as this person."

BASELINE_PERSONA_ID = "baseline"
BASELINE_PERSONA_NAME = "Assistant"

_PERSONA_VIEWS = {"templated": "templated_view", "biography": "biography_view"}


def mc_answer_only_instruction(n_choices: int) -> str:
    """Return the trailing answer-only instruction for an MC prompt."""
    if not 1 <= n_choices <= len(_LETTERS):
        raise ValueError(f"n_choices={n_choices} outside [1, {len(_LETTERS)}]")
    labels = ", ".join(_LETTERS[:n_choices])
    return f"Answer only with the correct choice label ({labels})."


def format_prompt(
    persona: PersonaData | str | None = None,
    variant: Literal["baseline", "templated", "biography"] = BASELINE_PERSONA_ID,
    mode: Literal["roleplay", "conversational"] = "roleplay",
) -> str:
    """Build a standard persona system prompt.

    Args:
        persona: A ``PersonaData`` object, raw profile text, or ``None`` for
            the persona-less Assistant baseline.
        variant: Which standard view to read when ``persona`` is a
            ``PersonaData`` object. Ignored for raw profile text.
        mode: ``"roleplay"`` for the plain persona prompt, or
            ``"conversational"`` to append a natural chat instruction.
    """
    if mode not in ("roleplay", "conversational"):
        raise ValueError(f"Unsupported mode: {mode!r}")

    if isinstance(persona, str):
        profile_text = persona
    elif variant == BASELINE_PERSONA_ID:
        profile_text = BASELINE_PERSONA_NAME
    elif variant not in _PERSONA_VIEWS:
        raise ValueError(f"Unsupported persona prompt variant: {variant!r}")
    elif persona is None:
        raise ValueError(f"variant {variant!r} requires a persona")
    else:
        profile_text = getattr(persona, _PERSONA_VIEWS[variant])

    base = BASE_ROLEPLAY_PROMPT.format(persona=profile_text)
    return base + _CONVERSATIONAL_SUFFIX if mode == "conversational" else base


def format_mc_question(qa: QAPair) -> str:
    """Format an MC question and append the answer-only instruction."""
    lines = [qa.question, ""]
    for i, choice in enumerate(qa.choices):
        lines.append(f"{_LETTERS[i]}. {choice}")
    lines.append("")
    lines.append(mc_answer_only_instruction(len(qa.choices)))
    return "\n".join(lines)


def mc_correct_letter(qa: QAPair) -> str:
    """Return the label for the correct choice of an MC question."""
    if qa.correct_choice_index is None:
        raise ValueError(f"QAPair {qa.qid!r} has no correct_choice_index")
    return _LETTERS[qa.correct_choice_index]


def supports_system_role(tokenizer) -> bool:
    """Check if the tokenizer's chat template supports the 'system' role."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"}], tokenize=False
        )
        return True
    except Exception:
        return False


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge a leading system message into the first user message.

    Only needed when the tokenizer's chat template doesn't support the
    "system" role (e.g., Gemma 2).
    """
    if not messages or messages[0]["role"] != "system":
        return messages
    system_content = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0]["role"] == "user" and system_content:
        rest[0] = {
            "role": "user",
            "content": f"{system_content}\n\n{rest[0]['content']}",
        }
    return rest


def format_messages(
    messages: list[dict[str, str]],
    tokenizer,
    add_generation_prompt: bool = False,
) -> tuple[str, int]:
    """Format a conversation for the model using its chat template.

    Args:
        messages: List of ``{"role", "content"}`` dicts. Roles may be
            ``system``, ``user``, or ``assistant``.
        tokenizer: Tokenizer with chat-template support.
        add_generation_prompt: When ``False`` (extraction/training),
            ``messages`` ends with the assistant turn to score, and the
            returned index points at its first token. When ``True``
            (inference), the generation-prompt prefix is appended and the
            index equals the prompt length — slice generated sequences with
            ``sequences[:, response_start_idx:]``.

    Returns:
        ``(full_prompt, response_start_idx)``.
    """
    if not supports_system_role(tokenizer):
        messages = normalize_messages(messages)

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )

    # The boundary is "everything before the last assistant turn", which is
    # equivalent to tokenizing all messages with a generation prompt — except
    # in inference mode, where the boundary is the full prompt itself.
    prefix_messages = messages if add_generation_prompt else messages[:-1] or messages
    prefix_ids = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    response_start_idx = prefix_ids["input_ids"].shape[1]

    return full_prompt, response_start_idx

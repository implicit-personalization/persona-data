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
PromptMode = Literal["roleplay", "conversational"]


def mc_answer_only_instruction(n_choices: int) -> str:
    """Return the trailing answer-only instruction for an MC prompt."""
    if n_choices < 1 or n_choices > len(_LETTERS):
        raise ValueError(f"n_choices={n_choices} outside [1, {len(_LETTERS)}]")
    labels = ", ".join(_LETTERS[:n_choices])
    return f"Answer only with the correct choice label ({labels})."


def format_prompt(
    persona: PersonaData | str | None = None,
    variant: Literal["baseline", "templated", "biography"] = BASELINE_PERSONA_ID,
    mode: PromptMode = "roleplay",
) -> str:
    """Build a standard persona system prompt.

    Args:
        persona: A ``PersonaData`` object, raw profile text, or ``None`` for
            the persona-less Assistant baseline.
        variant: Which standard view to read when ``persona`` is a
            ``PersonaData`` object. Ignored for raw profile text.
        mode: Prompt style selector.
            - "roleplay": plain persona prompt
            - "conversational": persona prompt with a natural chat instruction

    Raises:
        ValueError: If `mode` or `variant` is not supported, or a persona is
            required but missing.
    """
    if isinstance(persona, str):
        profile_text = persona
    elif variant == BASELINE_PERSONA_ID:
        profile_text = BASELINE_PERSONA_NAME
    elif variant == "templated":
        if persona is None:
            raise ValueError(f"variant {variant!r} requires a persona")
        profile_text = persona.templated_view
    elif variant == "biography":
        if persona is None:
            raise ValueError(f"variant {variant!r} requires a persona")
        profile_text = persona.biography_view
    else:
        raise ValueError(f"Unsupported persona prompt variant: {variant!r}")

    base = BASE_ROLEPLAY_PROMPT.format(persona=profile_text)
    if mode == "conversational":
        return base + _CONVERSATIONAL_SUFFIX
    if mode != "roleplay":
        raise ValueError(f"Unsupported mode: {mode!r}")
    return base


def _format_mc_question_prompt(qa: QAPair) -> str:
    lines = [qa.question, ""]
    for i, choice in enumerate(qa.choices):
        lines.append(f"{_LETTERS[i]}. {choice}")
    return "\n".join(lines)


def format_mc_question(qa: QAPair) -> str:
    """Format an MC question and append the answer-only instruction."""
    instruction = mc_answer_only_instruction(len(qa.choices))
    return f"{_format_mc_question_prompt(qa).rstrip()}\n\n{instruction}"


def mc_correct_letter(qa: QAPair) -> str:
    """Return the label for the correct choice of an MC question."""
    if qa.correct_choice_index is None:
        raise ValueError(f"QAPair {qa.qid!r} has no correct_choice_index")
    return _LETTERS[qa.correct_choice_index]


def supports_system_role(tokenizer) -> bool:
    """Check if tokenizer's chat template supports the 'system' role."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"}],
            tokenize=False,
        )
        return True
    except Exception:
        return False


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge any leading system message into the first user message.

    Only needed when the tokenizer's chat template doesn't support
    the "system" role (e.g., Gemma 2).
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
        messages: List of message dicts with "role" and "content" keys.
                 Can include "system", "user", and "assistant" roles.
        tokenizer: The tokenizer with chat template support.
        add_generation_prompt: When False (default, extraction/training),
            ``messages`` is expected to end with an ``assistant`` turn whose
            content is the response to score. When True (inference), the
            generation-prompt prefix (e.g. ``<start_of_turn>model``) is
            appended so the rendered string is ready to feed to ``generate``.

    Returns:
        full_prompt: The full formatted prompt as a string.
        response_start_idx: Token index where the assistant response begins.
            With ``add_generation_prompt=False`` this points to the first
            token of the last assistant message in ``messages``. With
            ``add_generation_prompt=True`` it equals the prompt length, i.e.
            the index of the first token the model will generate.
    """
    if not supports_system_role(tokenizer):
        messages = normalize_messages(messages)

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )

    if add_generation_prompt:
        prefix_messages = messages
    elif len(messages) > 1:
        prefix_messages = messages[:-1]
    else:
        prefix_messages = messages

    prompt_without_response = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    response_start_idx = prompt_without_response["input_ids"].shape[1]

    return full_prompt, response_start_idx

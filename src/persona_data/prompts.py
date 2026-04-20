from typing import Literal

from persona_data.synth_persona import QAPair

_LETTERS = "ABCDEFG"

BASE_ROLEPLAY_PROMPT = """\
You are roleplaying as a specific person in a conversation.
Stay fully in character. Be truthful to the profile below.
Do not mention that you are an AI model.

### Person profile:

{persona}"""

_CONVERSATIONAL_SUFFIX = "\n\nAnswer naturally and conversationally as this person."

EMPTY_PERSONA_PLACEHOLDER = "Assistant"
PromptMode = Literal["roleplay", "conversational"]


def mc_answer_only_instruction(n_choices: int) -> str:
    """Return the trailing answer-only instruction for an MC prompt."""
    if n_choices < 1 or n_choices > len(_LETTERS):
        raise ValueError(f"n_choices={n_choices} outside [1, {len(_LETTERS)}]")
    labels = ", ".join(_LETTERS[:n_choices])
    return f"Answer only with the correct choice label ({labels})."


def format_roleplay_prompt(
    persona: str = EMPTY_PERSONA_PLACEHOLDER, mode: PromptMode = "roleplay"
) -> str:
    """Build a persona system prompt.

    Args:
        persona: The persona text (templated or biography view).
        mode: Prompt style selector.
            - "roleplay": plain persona prompt
            - "conversational": persona prompt with a natural chat instruction

    Raises:
        ValueError: If `mode` is not one of the supported values.
    """
    base = BASE_ROLEPLAY_PROMPT.format(persona=persona)
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


def format_messages(messages: list[dict[str, str]], tokenizer) -> tuple[str, int]:
    """Format a conversation for the model using its chat template.

    Args:
        messages: List of message dicts with "role" and "content" keys.
                 Can include "system", "user", and "assistant" roles.
        tokenizer: The tokenizer with chat template support.

    Returns:
        full_prompt: The full formatted prompt as a string.
        response_start_idx: The token index of the first token in the last
                            assistant message.
    """
    supports_system = supports_system_role(tokenizer)
    if not supports_system:
        messages = normalize_messages(messages)

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    if len(messages) <= 1:
        prompt_without_response = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        prompt_without_response = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    response_start_idx = prompt_without_response["input_ids"].shape[1]

    return full_prompt, response_start_idx

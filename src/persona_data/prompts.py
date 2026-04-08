from persona_data.synth_persona import QAPair

_LETTERS = "ABCDE"

BASE_ROLEPLAY_PROMPT = """\
You are roleplaying as a specific person in a conversation.
Stay fully in character. Be truthful to the profile below.
Do not mention that you are an AI model.

# Person profile:

{persona}

ROLEPLAY GUIDELINES:

- Answer naturally and conversationally as this person."""

EMPTY_PERSONA_PLACEHOLDER = "Assistant"
MC_ANSWER_ONLY_INSTRUCTION = "Answer only with the correct choice label (A, B, C, ...)."


def format_roleplay_prompt(persona: str = EMPTY_PERSONA_PLACEHOLDER) -> str:
    """System prompt using any persona view."""
    return BASE_ROLEPLAY_PROMPT.format(persona=persona)


def format_mc_prompt(prompt: str) -> str:
    """Append the multiple-choice answer constraint to any prompt."""
    return f"{prompt.rstrip()}\n\n{MC_ANSWER_ONLY_INSTRUCTION}"


def _format_mc_question_prompt(qa: QAPair) -> str:
    lines = [qa.question, ""]
    for i, choice in enumerate(qa.choices):
        lines.append(f"{_LETTERS[i]}. {choice}")
    return "\n".join(lines)


def format_mc_question(qa: QAPair) -> str:
    """Format an MC question with lettered choices for model evaluation."""
    return format_mc_prompt(_format_mc_question_prompt(qa))


def _supports_system_role(tokenizer) -> bool:
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
    supports_system = _supports_system_role(tokenizer)
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

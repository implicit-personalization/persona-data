# Prompt formatting

`persona_data.prompts` covers two evaluation patterns: chatting with a persona in character, and asking a persona to answer a multiple-choice question.

## Roleplay system prompts

```python
from persona_data.prompts import format_prompt
from persona_data.synth_persona import SynthPersonaDataset

dataset = SynthPersonaDataset()
persona = dataset[0]

system_prompt = format_prompt(persona, "biography")
```

`format_prompt(persona, variant, mode)`:

- `persona` is a `PersonaData` or raw profile text.
- `variant` is `"templated"` (short attribute list) or `"biography"` (full prose). Ignored when `persona` is a string.
- `mode="roleplay"` (default) returns the plain prompt; `mode="conversational"` appends a natural-chat instruction.

The persona-less Assistant baseline lives in the dataset as `BASELINE_PERSONA_ID` (`"baseline_assistant"`). Retrieve it via `dataset.baseline`, then pass it to `format_prompt` like any other persona. `BASELINE_PERSONA_NAME` (`"Assistant"`) is the display label for artifact naming.

| View | When to use |
|---|---|
| `persona.biography_view` | Rich prose biography; best for open-ended chat |
| `persona.templated_view` | Short attribute list; lower token cost |
| `persona.statements_view` | Bullet-point claims; useful for fact-checking |

## Multiple-choice prompts

```python
from persona_data.prompts import format_mc_question, mc_correct_letter

qa = next(qa for qa in dataset.get_qa(persona.id) if qa.choices)

question_prompt = format_mc_question(qa)  # body + lettered choices + answer-only instruction
correct         = mc_correct_letter(qa)   # ground-truth label
```

`mc_answer_only_instruction(n_choices)` returns just the trailing instruction (supports 1–7 choices).

Rendered output:

```
What is Ethan's primary occupation?

A. Software engineer
B. Teacher
C. Nurse
D. Accountant

Answer only with the correct choice label (A, B, C, D).
```

## Tokenizing for a local model

`format_messages(messages, tokenizer, add_generation_prompt=False)` applies the tokenizer's chat template and returns `(full_prompt, response_start_idx)`.

- `add_generation_prompt=False` (extraction / training): `messages` ends with the assistant turn to score; the index points at its first token.
- `add_generation_prompt=True` (inference): the generation-prompt prefix is appended; the index equals the prompt length, so slice generations with `sequences[:, response_start_idx:]`.

Tokenizers without a `"system"` role (e.g. Gemma 2) are handled automatically — system content is merged into the first user message via `normalize_messages`. `supports_system_role(tokenizer)` is exposed for callers that need to branch on this.

## Roleplay + MC combined

```python
from persona_data.prompts import format_messages, format_mc_question, format_prompt, mc_correct_letter

system_prompt   = format_prompt(persona, "biography")
question_prompt = format_mc_question(qa)
correct         = mc_correct_letter(qa)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": question_prompt},
]
full_prompt, response_start_idx = format_messages(messages, tokenizer)
```

The model should reply with a single letter; compare against `correct` to score.

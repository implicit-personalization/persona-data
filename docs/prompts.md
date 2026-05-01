# Prompt formatting

`persona_data.prompts` provides helpers for two evaluation patterns: chatting with a persona in character, and asking a persona to answer a multiple-choice question.

## Roleplay (chat with a specific persona)

`format_prompt` is the main entry point for persona prompts:

```python
from persona_data.synth_persona import SynthPersonaDataset
from persona_data.prompts import format_prompt

dataset = SynthPersonaDataset()
persona = dataset[0]

system_prompt = format_prompt(persona, "biography")
```

The resulting system prompt instructs the model to stay in character and not reveal it is an AI.

`format_prompt` accepts raw profile text, a `PersonaData` plus one of the standard variants, or no persona for the shared baseline. It also accepts a `mode` argument:

- `mode="roleplay"` for the plain persona prompt
- `mode="conversational"` to add a natural chat instruction

### Variant-Aware System Prompts

To iterate over persona variants, keep the experiment's choices local and pass each variant to `format_prompt`:

```python
from persona_data.prompts import format_prompt

for variant in ("templated", "biography"):
    system_prompt = format_prompt(persona, variant)
```

### Baseline Prompt

The persona-less Assistant baseline is `format_prompt()` with no arguments:

```python
from persona_data.prompts import (
    BASELINE_PERSONA_ID,
    BASELINE_PERSONA_NAME,
    format_prompt,
)

system_prompt = format_prompt()
```

Downstream packages can use `BASELINE_PERSONA_ID` and `BASELINE_PERSONA_NAME`
for artifact naming or UI labels.

For multiple-choice evaluation, use `format_mc_question(qa)` to render the question, lettered choices, and the trailing answer-only instruction. Use `mc_correct_letter(qa)` to get the ground-truth label.

`format_mc_question()` expects a `QAPair` with `choices` and `correct_choice_index` populated. `mc_answer_only_instruction(n_choices)` supports 1 to 7 choices.

### Tokenizing for a local model

`format_messages` applies the tokenizer's chat template and returns the full prompt string plus the token index where the assistant response begins. It supports two modes via `add_generation_prompt`:

```python
from persona_data.prompts import format_messages

# Extraction / training: messages end with the assistant turn to score.
# response_start_idx points at the first token of that last assistant message.
full_prompt, response_start_idx = format_messages(messages, tokenizer)

# Inference: messages end with a user turn. The generation-prompt prefix
# (e.g. <start_of_turn>model) is appended; response_start_idx equals the
# prompt length, so model output can be sliced with sequences[:, response_start_idx:].
full_prompt, response_start_idx = format_messages(
    messages, tokenizer, add_generation_prompt=True
)
```

Tokenizers that do not support the `"system"` role (e.g. Gemma 2) are handled automatically — the system content is merged into the first user message via the public `normalize_messages` helper. `supports_system_role(tokenizer)` is exposed for callers that need to branch on this.

### Persona views

| View | When to use |
|---|---|
| `persona.biography_view` | Rich prose biography; best for open-ended chat |
| `persona.templated_view` | Short attribute list; faster, lower token cost |
| `persona.statements_view` | Bullet-point claims; useful for fact-checking tasks |

---

## Multiple-choice evaluation

Use `format_mc_question` to format a `QAPair` into a lettered multiple-choice prompt. Use `mc_correct_letter` to get the ground-truth label.

```python
from persona_data.synth_persona import SynthPersonaDataset
from persona_data.prompts import format_mc_question, mc_correct_letter

dataset = SynthPersonaDataset()
persona = dataset[0]

# Retrieve a QA pair with choices populated
qa = next(qa for qa in dataset.get_qa(persona.id) if qa.choices)

question_prompt = format_mc_question(qa)
correct         = mc_correct_letter(qa)
```

`mc_answer_only_instruction(n_choices)` returns just the trailing instruction if you need it separately.

`format_mc_question` renders the question body, lettered choices (A, B, C, …), and appends a trailing instruction telling the model to reply with only the choice label:

```
What is Ethan's primary occupation?

A. Software engineer
B. Teacher
C. Nurse
D. Accountant

Answer only with the correct choice label (A, B, C).
```

### Combining roleplay + multiple-choice

To evaluate whether a model answers questions correctly when embodying a persona, combine both helpers:

```python
from persona_data.prompts import format_messages, format_mc_question, format_prompt, mc_correct_letter

system_prompt    = format_prompt(persona, "biography")
question_prompt  = format_mc_question(qa)
correct          = mc_correct_letter(qa)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": question_prompt},
]

full_prompt, response_start_idx = format_messages(messages, tokenizer)
```

The model should then reply with a single letter. Compare it against `correct` to score the response.

See [SynthPersona](synth_persona.md) for filtering QA pairs by `type` and `difficulty`.

# Prompt formatting

`persona_data.prompts` provides helpers for two evaluation patterns: chatting with a persona in character, and asking a persona to answer a multiple-choice question.

## Roleplay (chat with a specific persona)

Use `system_prompt_for_variant` when you are iterating over persona variants, and `format_roleplay_prompt` when you already have the exact persona text.

```python
from persona_data.synth_persona import SynthPersonaDataset
from persona_data.prompts import format_roleplay_prompt, system_prompt_for_variant

dataset = SynthPersonaDataset()
persona = dataset[0]

system_prompt = system_prompt_for_variant(persona, "biography")

# If you already have the raw persona text, this is the lower-level primitive.
# system_prompt = format_roleplay_prompt(persona.biography_view)
```

The resulting system prompt instructs the model to stay in character and not reveal it is an AI.

`format_roleplay_prompt` also accepts a `mode` argument:

- `mode="roleplay"` for the plain persona prompt
- `mode="conversational"` to add a natural chat instruction

### Variant-aware system prompts

When running the same pipeline over multiple persona variants (e.g. `templated`, `biography`, `statements`, or the persona-less `baseline`), use `system_prompt_for_variant` to avoid branching on `"baseline"` at every call site:

```python
from persona_data.prompts import system_prompt_for_variant

for variant in ("baseline", "templated", "biography"):
    system_prompt = system_prompt_for_variant(persona, variant)
```

`"baseline"` returns a persona-less prompt; any other variant reads `<variant>_view` from the persona. `mode` is forwarded to `format_roleplay_prompt`.

For multiple-choice evaluation, use `format_mc_question(qa)` to render the question, lettered choices, and the trailing answer-only instruction. Use `mc_correct_letter(qa)` to get the ground-truth label.

`format_mc_question()` expects a `QAPair` with `choices` and `correct_choice_index` populated. `mc_answer_only_instruction(n_choices)` supports 1 to 7 choices.

### Tokenizing for a local model

`format_messages` applies the tokenizer's chat template and returns the full prompt string plus the token index where the generated response should begin (used for loss masking or logit slicing):

```python
from persona_data.prompts import format_messages

full_prompt, response_start_idx = format_messages(messages, tokenizer)
```

Tokenizers that do not support the `"system"` role (e.g. Gemma 2) are handled automatically — the system content is merged into the first user message.

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
from persona_data.prompts import format_messages, format_mc_question, mc_correct_letter, system_prompt_for_variant

system_prompt    = system_prompt_for_variant(persona, "biography")
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

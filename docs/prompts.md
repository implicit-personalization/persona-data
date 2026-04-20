# Prompt formatting

`persona_data.prompts` provides helpers for two evaluation patterns: chatting with a persona in character, and asking a persona to answer a multiple-choice question.

## Roleplay (chat with a specific persona)

Use `format_roleplay_prompt` to build a system prompt that puts the model in character as a persona.

```python
from persona_data.synth_persona import SynthPersonaDataset
from persona_data.prompts import format_roleplay_prompt

dataset = SynthPersonaDataset()
persona = dataset[0]

# Pick the persona view you want the model to embody.
# biography_view gives the richest context; templated_view is more compact.
system_prompt = format_roleplay_prompt(persona.biography_view)
```

The resulting system prompt instructs the model to stay in character and not reveal it is an AI.

`format_roleplay_prompt` also accepts a `mode` argument:

- `mode="roleplay"` for the plain persona prompt
- `mode="conversational"` to add a natural chat instruction

### Variant-aware system prompts

When running the same pipeline over multiple persona variants (e.g. `templated`, `biography`, or the persona-less `baseline`), use `system_prompt_for_variant` to avoid branching on `"baseline"` at every call site:

```python
from persona_data.prompts import system_prompt_for_variant

for variant in ("baseline", "templated", "biography"):
    system_prompt = system_prompt_for_variant(persona, variant)
```

`"baseline"` returns a persona-less prompt; any other variant reads `<variant>_view` from the persona. `mode` is forwarded to `format_roleplay_prompt`.

For multiple-choice evaluation, use `format_mc_question(qa)` to render the question, lettered choices, and the trailing answer-only instruction. Use `mc_correct_letter(qa)` to get the ground-truth label.

### Tokenizing for a local model

`format_messages` applies the tokenizer's chat template and returns the full prompt string plus the token index where the last assistant turn begins (used for loss masking or logit slicing):

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

# Retrieve MC-capable QA pairs (type="explicit" or "implicit")
qa_pairs = dataset.get_qa(persona.id, type="explicit")
qa = qa_pairs[0]

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
system_prompt    = format_roleplay_prompt(persona.biography_view)
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

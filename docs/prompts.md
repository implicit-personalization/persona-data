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
- `mode="mc"` to add the multiple-choice answer constraint

### Building the message list

Pass the system prompt as the first message, then alternate user/assistant turns normally:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": "Where did you grow up?"},
]
```

To continue a conversation, append the model's previous reply and the next user message:

```python
messages += [
    {"role": "assistant", "content": "I grew up in Little Rock, Arkansas."},
    {"role": "user",      "content": "What did you study at university?"},
]
```

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

`format_mc_question` renders the question body, lettered choices (A, B, C, …), and appends a trailing instruction telling the model to reply with only the choice label:

```
What is Ethan's primary occupation?

A. Software engineer
B. Teacher
C. Nurse
D. Accountant

Answer only with the correct choice label (A, B, C, ...).
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

### Filtering QA pairs by difficulty

`get_qa` accepts `difficulty` to narrow the question pool:

```python
# Only easy questions (level 1)
easy_qa = dataset.get_qa(persona.id, type="implicit", difficulty=1)

# Medium and hard questions (levels 2 and 3)
harder_qa = dataset.get_qa(persona.id, type="explicit", difficulty=[2, 3])
```

Difficulty 1 questions have answers stated explicitly in the profile; higher levels require inference.

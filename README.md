# persona-data

Shared dataset loading, prompt formatting, and environment utilities for the [implicit-personalization](https://github.com/implicit-personalization) projects.

## Installation

Add as a uv git source in your project's `pyproject.toml`:

```toml
[project]
dependencies = ["persona-data"]

[tool.uv.sources]
persona-data = { git = "ssh://git@github.com/implicit-personalization/persona-data.git" }
```

Then run `uv sync`.

For local development alongside other repos, use an editable path source:

```toml
[tool.uv.sources]
persona-data = { path = "../persona-data", editable = true }
```

## Package layout

```
persona_data/
├── __init__.py            # empty
├── synth_persona.py       # SynthPersonaDataset, PersonaData, QAPair, BiographySection
├── persona_guess.py       # PersonaGuessDataset, GameRecord, Turn
├── prompts.py             # format_biography_prompt, format_templated_prompt, format_messages
└── environment.py         # load_env, set_seed, get_device, get_artifacts_dir
```

## Datasets

Each dataset is a module with its own types and a loader that downloads from HuggingFace (cached via `HF_HOME`). See [docs/datasets.md](docs/datasets.md) for the full guide on adding new datasets.

### SynthPersona

```python
from persona_data.synth_persona import SynthPersonaDataset

dataset = SynthPersonaDataset()

persona = dataset[0]
persona.name              # "Ethan Robinson"
persona.biography_md      # full markdown biography
persona.templated_prompt  # short attribute-based system prompt
persona.sections          # list of BiographySection

qa_pairs = dataset.get_qa(persona.id, type="implicit", difficulty=[1, 2])
questions = dataset.questions(persona.id, type="explicit")
```

### PersonaGuess

```python
from persona_data.persona_guess import PersonaGuessDataset

games = PersonaGuessDataset()
game = games[0]
turns = games.get_qa(game.game_id, player="A")
```

## Prompt formatting

```python
from persona_data.prompts import format_biography_prompt, format_messages

system_prompt = format_biography_prompt(persona.biography_md)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Where did you grow up?"},
    {"role": "assistant", "content": "I grew up in Little Rock, Arkansas."},
]
full_prompt, response_start_idx = format_messages(messages, tokenizer)
```

`format_messages` handles tokenizers that don't support the `"system"` role (e.g. Gemma 2) by merging system content into the first user message.

## Environment helpers

```python
from persona_data.environment import load_env, set_seed, get_device, get_artifacts_dir

load_env()            # loads .env from cwd (searches parent dirs)
set_seed(1337)        # sets random, numpy, and torch seeds
device = get_device() # cuda > mps > cpu
```

## Docs

- [docs/datasets.md](docs/datasets.md) — dataset loading guide and template for adding new datasets

## Used by

- [persona-vectors](https://github.com/implicit-personalization/persona-vectors) — activation extraction and steering
- [cues_attribution](https://github.com/implicit-personalization/io-analysis) — section-level ablation attribution
- [persona-2-lora](https://github.com/implicit-personalization/persona-2-lora) — LoRA-based persona internalization

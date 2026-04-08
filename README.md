# persona-data

[![Docs](https://img.shields.io/badge/docs-view-purple?logo=github)](https://github.com/implicit-personalization/persona-data/tree/main/docs)

Shared dataset loading, prompt formatting, and environment utilities for the [implicit-personalization](https://github.com/implicit-personalization) projects.

## Overview

`persona-data` provides the common dataset and prompt helpers used across the persona projects:

- `SynthPersonaDataset` for persona profiles plus QA pairs
- `PersonaGuessDataset` for turn-based persona games
- prompt helpers for roleplay and multiple-choice evaluation
- environment helpers for seeds, devices, and artifact paths

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
src/persona_data/
├── __init__.py
├── synth_persona.py       # SynthPersonaDataset, PersonaDataset, PersonaData, QAPair, BiographySection
├── persona_guess.py       # PersonaGuessDataset, GameRecord, Turn
├── prompts.py             # format_roleplay_prompt, format_mc_prompt, format_messages
└── environment.py         # load_env, set_seed, get_device, get_artifacts_dir
```

## Datasets

Each dataset is a module with its own types and a loader that downloads from Hugging Face, cached via `HF_HOME`.

### SynthPersona

```python
from persona_data.synth_persona import SynthPersonaDataset

dataset = SynthPersonaDataset()

persona = dataset[0]
persona.name              # "Ethan Robinson"
persona.templated_view    # short attribute-based system prompt
persona.biography_view    # full biography text
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
questions = games.questions(game.game_id, player="B")
```

## Prompt formatting

```python
from persona_data.prompts import format_messages, format_roleplay_prompt

system_prompt = format_roleplay_prompt(persona.biography_view)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Where did you grow up?"},
    {"role": "assistant", "content": "I grew up in Little Rock, Arkansas."},
]
full_prompt, response_start_idx = format_messages(messages, tokenizer)
```

`format_messages` handles tokenizers that do not support the `"system"` role (for example Gemma 2) by merging system content into the first user message.

For multiple-choice evaluation, use `format_mc_question(qa)` and `mc_correct_letter(qa)`.

## Environment helpers

```python
from persona_data.environment import load_env, set_seed, get_device, get_artifacts_dir

load_env()            # loads .env from cwd (searches parent dirs)
set_seed(1337)        # sets random, numpy, and torch seeds
device = get_device() # cuda > mps > cpu
```


## Used by

- [persona-vectors](https://github.com/implicit-personalization/persona-vectors) — activation extraction and steering
- [cues_attribution](https://github.com/implicit-personalization/io-analysis) — section-level ablation attribution
- [persona-2-lora](https://github.com/implicit-personalization/persona-2-lora) — LoRA-based persona internalization

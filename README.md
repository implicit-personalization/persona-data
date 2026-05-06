# persona-data

[![Docs](https://img.shields.io/badge/docs-view-purple?logo=materialformkdocs)](https://implicit-personalization.github.io/persona-data/)
[![PyPI](https://img.shields.io/pypi/v/persona-data?logo=pypi&label=PyPI)](https://pypi.org/project/persona-data/)

Shared dataset loading, prompt formatting, and environment utilities for the [implicit-personalization](https://github.com/implicit-personalization) projects.

## Overview

`persona-data` provides the common dataset and prompt helpers used across the persona projects:

- `SynthPersonaDataset` for persona profiles plus QA pairs
- `PersonaGuessDataset` for turn-based persona games
- `NemotronPersonasFranceDataset` for French persona profiles from NVIDIA
- `NemotronPersonasUSADataset` for US persona profiles from NVIDIA
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

### Testing

```bash
uv run --with pytest pytest tests/test_datasets.py
```

The release workflow also runs `tests/smoke_test.py` against the built wheel and source distribution.

## Package layout

```
src/persona_data/
├── __init__.py
├── synth_persona.py       # SynthPersonaDataset, PersonaDataset, PersonaData, QAPair, Statement
├── persona_guess.py       # PersonaGuessDataset, GameRecord, Turn
├── nemotron_personas.py   # NemotronPersonasFranceDataset, NemotronPersonasUSADataset
├── prompts.py             # format_prompt, format_mc_question, format_messages
└── environment.py         # set_seed, get_device, get_artifacts_dir
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
persona.statements        # list of Statement

qa_pairs = dataset.get_qa(persona.id, type="implicit", item_type="mcq")

# Leakage-aware split: train on individual FRQs, test on shared MCQs.
train_qa, test_qa = dataset.train_test_split(persona.id, n_train=50)
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
from persona_data.prompts import format_messages, format_prompt

system_prompt = format_prompt(persona, "biography")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Where did you grow up?"},
    {"role": "assistant", "content": "I grew up in Little Rock, Arkansas."},
]
full_prompt, response_start_idx = format_messages(messages, tokenizer)
```

`format_prompt` accepts a `PersonaData` plus one of the standard variants (`"templated"` or `"biography"`), or raw profile text. It also accepts `mode="roleplay"` (default) and `mode="conversational"`.

The persona-less Assistant baseline is just another persona in the dataset under `BASELINE_PERSONA_ID` (`"baseline_assistant"`). It appears in normal iteration when loaded, and `dataset.baseline` retrieves it directly:

```python
dataset = SynthPersonaDataset()
baseline = dataset.baseline  # PersonaData | None
system_prompt = format_prompt(baseline, "templated")
```

Use `BASELINE_PERSONA_ID` and `BASELINE_PERSONA_NAME` (both in `persona_data.prompts`) for artifact naming and UI labels.

For multiple-choice prompts, use `format_mc_question(qa)` to render the question, choices, and trailing answer-only instruction. Use `mc_answer_only_instruction(n_choices)` if you need just the instruction text, and `mc_correct_letter(qa)` to get the gold label.

`format_messages` handles tokenizers that do not support the `"system"` role (for example Gemma 2) by merging system content into the first user message. Pass `add_generation_prompt=True` to render an inference-ready prompt (messages ending in a `user` turn); the returned `response_start_idx` then equals the prompt length, ready to slice `model.generate` output.

## Environment helpers

```python
from persona_data.environment import set_seed, get_device, get_artifacts_dir

set_seed(1337)        # sets random, numpy, and torch seeds
device = get_device() # cuda > mps > cpu
```


## Used by

- [persona-vectors](https://github.com/implicit-personalization/persona-vectors) — activation extraction and steering
- [cues_attribution](https://github.com/implicit-personalization/io-analysis) — section-level ablation attribution
- [persona-2-lora](https://github.com/implicit-personalization/persona-2-lora) — LoRA-based persona internalization

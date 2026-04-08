# persona-data

`persona-data` is a Python library for loading and working with synthetic persona datasets from Hugging Face.

## Datasets

| Dataset | HuggingFace | Description |
|---|---|---|
| [SynthPersona](synth_persona.md) | [implicit-personalization/synth-persona](https://huggingface.co/datasets/implicit-personalization/synth-persona) | Persona profiles with biography views and QA pairs |
| [PersonaGuess](persona_guess.md) | [implicit-personalization/persona-guess](https://huggingface.co/datasets/implicit-personalization/persona-guess) | Turn-based games where two personas ask each other questions |

## Prompt helpers

The [Prompt formatting](prompts.md) page covers helpers for roleplay prompts and multiple-choice evaluation.

## Shared conventions

- Loaders download from Hugging Face with `hf_hub_download`.
- Dataset instances implement `__len__`, `__iter__`, and `__getitem__`.
- Query helpers return typed records plus convenience string-only helpers.
- New datasets should stay small, eager, and easy to inspect from a notebook.

See [Adding a dataset](adding-a-dataset.md) to contribute a new loader.

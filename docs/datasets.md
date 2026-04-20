# Datasets

`persona-data` keeps dataset docs split by loader so each source stays easy to maintain.

## Pages

- [SynthPersona](synth_persona.md)
- [PersonaGuess](persona_guess.md)
- [Nemotron Personas France](nemotron_personas.md)
- [Prompt formatting](prompts.md)
- [Adding a new dataset](adding-a-dataset.md)

## Shared conventions

- Loaders download from Hugging Face with `hf_hub_download`, including sharded parquet sources.
- Dataset instances implement `__len__`, `__iter__`, and `__getitem__`.
- Query helpers return typed records plus convenience string-only helpers.
- New datasets should stay small, eager, and easy to inspect from a notebook.

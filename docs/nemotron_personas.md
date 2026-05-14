# Nemotron Personas

Persona-only loaders for the NVIDIA Nemotron Personas datasets (no QA pairs).

| Class | Source |
|---|---|
| `NemotronPersonasFranceDataset` | `nvidia/Nemotron-Personas-France` |
| `NemotronPersonasUSADataset` | `nvidia/Nemotron-Personas-USA` |

```python
from persona_data.nemotron_personas import (
    NemotronPersonasFranceDataset,
    NemotronPersonasUSADataset,
)

dataset = NemotronPersonasFranceDataset(sample_size=200)
usa_dataset = NemotronPersonasUSADataset(sample_size=200)
```

The loader discovers every `data/train-*.parquet` shard in sorted order and reads rows until it has collected `sample_size` personas (a leading slice, not a random sample). `first_name` and `last_name` are derived from the persona text when possible.

Records use the shared [`PersonaData`](synth_persona.md#persona-fields) type, exposing `id`, `persona`, `templated_view`, `biography_view`, and the derived `name`.

```python
persona = dataset[0]

persona.id              # source uuid
persona.name            # derived display name, or uuid as fallback
persona.biography_view  # original persona text
persona.templated_view  # normalized name, demographics, location, and bio fields
dataset.get_persona(persona.id)
dataset.supports_qa     # False
```

These datasets are persona-only; they do not expose `get_qa()` or QA splits.

## Templated view differences

- France: location formatted as `commune, departement, country`; trailing `Household type` field.
- USA: location formatted as `city, state, zipcode, country`; `Bachelors field` inserted before `Marital status`.

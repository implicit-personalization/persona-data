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

Records use the shared [`PersonaData`](synth_persona.md#persona-fields) type, exposing `id`, `persona`, `templated_view`, `biography_view`, the derived `name`, and `get_persona(persona_id)` for lookups.

## Templated view differences

- France: location formatted as `commune, departement, country`; trailing `Household type` field.
- USA: location formatted as `city, state, zipcode, country`; `Bachelors field` inserted before `Marital status`.

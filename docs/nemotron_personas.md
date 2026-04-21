# Nemotron Personas

`NemotronPersonasFranceDataset` loads persona-only French profiles from `nvidia/Nemotron-Personas-France`.
`NemotronPersonasUSADataset` loads persona-only US profiles from `nvidia/Nemotron-Personas-USA`.

## Loader

```python
from persona_data.nemotron_personas import (
    NemotronPersonasFranceDataset,
    NemotronPersonasUSADataset,
)

dataset = NemotronPersonasFranceDataset(sample_size=200)
usa_dataset = NemotronPersonasUSADataset(sample_size=200)
```

The loaders download sharded Parquet files from the dataset repo with `hf_hub_download` and keep the requested slice in memory.

## Files

The loader discovers every `data/train-*.parquet` shard in sorted order and reads rows until it has collected `sample_size` personas.

## Records

- `PersonaData`: top-level persona record

## Persona fields

`PersonaData` includes:

- `id`
- `persona`
- `templated_view`
- `biography_view`

It also exposes `name` as a derived property and `get_persona(persona_id)` for lookups.

## Notes

- The dataset is persona-only and does not include QA pairs.
- `sample_size` can limit how many personas are kept in memory.
- `sample_size` keeps a leading slice rather than sampling randomly.
- The loader derives `first_name` and `last_name` from the persona text when possible.
- The US loader formats location as `city, state, zipcode, country` and includes `bachelors_field` in the templated view.

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

- `data/train-00000-of-00012.parquet`
- `data/train-00001-of-00012.parquet`
- `data/train-00002-of-00012.parquet`
- `data/train-00003-of-00012.parquet`
- `data/train-00004-of-00012.parquet`
- `data/train-00005-of-00012.parquet`
- `data/train-00006-of-00012.parquet`
- `data/train-00007-of-00012.parquet`
- `data/train-00008-of-00012.parquet`
- `data/train-00009-of-00012.parquet`
- `data/train-00010-of-00012.parquet`
- `data/train-00011-of-00012.parquet`

USA:

- `data/train-00000-of-00011.parquet`
- `data/train-00001-of-00011.parquet`
- `data/train-00002-of-00011.parquet`
- `data/train-00003-of-00011.parquet`
- `data/train-00004-of-00011.parquet`
- `data/train-00005-of-00011.parquet`
- `data/train-00006-of-00011.parquet`
- `data/train-00007-of-00011.parquet`
- `data/train-00008-of-00011.parquet`
- `data/train-00009-of-00011.parquet`
- `data/train-00010-of-00011.parquet`

## Records

- `PersonaData`: top-level persona record

## Persona fields

`PersonaData` includes:

- `id`
- `persona`
- `templated_view`
- `biography_view`

It also exposes `name` as a derived property.

## Notes

- The dataset is persona-only and does not include QA pairs.
- `sample_size` can limit how many personas are kept in memory.
- The loader derives `first_name` and `last_name` from the persona text when possible.
- The US loader formats location as `city, state, zipcode, country` and includes `bachelors_field` in the templated view.

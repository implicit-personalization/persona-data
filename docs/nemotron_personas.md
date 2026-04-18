# Nemotron Personas France

`NemotronPersonasFranceDataset` loads persona-only French profiles from the Hugging Face dataset `nvidia/Nemotron-Personas-France`.

## Loader

```python
from persona_data.nemotron_personas import NemotronPersonasFranceDataset

dataset = NemotronPersonasFranceDataset(sample_size=200)
```

The loader downloads sharded Parquet files from the dataset repo with `hf_hub_download` and keeps the requested slice in memory.

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
- `sample_size` and `offset` let you load a small slice for notebook use.
- The loader derives `first_name` and `last_name` from the persona text when possible.

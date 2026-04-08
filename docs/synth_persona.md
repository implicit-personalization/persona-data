# SynthPersona

`SynthPersonaDataset` loads persona profiles plus QA pairs from Hugging Face and exposes a small in-memory API for analysis and prompt generation.

## Loader

```python
from persona_data.synth_persona import SynthPersonaDataset

dataset = SynthPersonaDataset()
```

The default dataset source is `implicit-personalization/synth-persona`. The loader reads two JSONL files:

- `dataset_personas.jsonl`
- `dataset_qa.jsonl`

## Records

- `PersonaData`: top-level persona record
- `QAPair`: question-answer pair with type, difficulty, and optional multiple-choice fields
- `BiographySection`: structured subsection of a persona biography
- `Statement`: supporting claim record used by downstream tooling

## Persona fields

`PersonaData` includes:

- `id`
- `persona`
- `templated_view`
- `biography_view`
- `statements_view`
- `sections`
- `statements`

It also exposes `name` as a derived property.

## Queries

```python
persona = dataset[0]

qa_pairs = dataset.get_qa(persona.id)
qa_pairs = dataset.get_qa(persona.id, type="explicit", difficulty=[2, 3])

questions = dataset.questions(persona.id, type="implicit")
```

`get_qa()` returns typed `QAPair` records. `questions()` returns question strings only.

## Notes

- `difficulty` accepts one level or a list of levels.
- `type` can be `"explicit"` or `"implicit"`.
- The loader keeps the dataset eager and notebook-friendly rather than streaming.

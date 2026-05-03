# SynthPersona

`SynthPersonaDataset` loads persona profiles plus QA pairs from Hugging Face and exposes a small in-memory API for analysis and prompt generation.

## Loader

```python
from persona_data.synth_persona import SynthPersonaDataset

dataset = SynthPersonaDataset()
small = SynthPersonaDataset(sample_size=100)
```

The default dataset source is `implicit-personalization/synth-persona`. The loader reads two JSONL files:

- `dataset_personas.jsonl`
- `dataset_qa.jsonl`

`sample_size` keeps the leading personas and filters QA rows to the loaded persona IDs.

## Records

- `PersonaData`: top-level persona record
- `QAPair`: question-answer pair with type, item_type, and optional multiple-choice fields
- `Statement`: supporting claim record used by downstream tooling

`Statement` includes `sid`, `category`, `claim`, and `support_turns`.

`QAPair` includes `qid`, `type`, `item_type`, `question`, `answer`, `choices`, `correct_choice_index`, `evidence_sids`, and `tags`.

## Persona fields

`PersonaData` includes:

- `id`
- `persona`
- `templated_view`
- `biography_view`
- `statements_view`
- `statements`

It also exposes `name` as a derived property.

## Queries

```python
persona = dataset[0]

qa_pairs = dataset.get_qa(persona.id)
qa_pairs = dataset.get_qa(persona.id, type="explicit", item_type="mcq")

loaded_persona = dataset.get_persona("p1")
```

`get_qa()` returns typed `QAPair` records.

## Notes

- `type` can be `"explicit"` or `"implicit"`.
- `item_type` can be `"mcq"` (multiple-choice) or `"frq"` (free-response).
- `sample_size` keeps a leading slice rather than sampling randomly.
- The loader keeps the dataset eager and notebook-friendly rather than streaming.

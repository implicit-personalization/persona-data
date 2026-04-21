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
- `QAPair`: question-answer pair with type, difficulty, and optional multiple-choice fields
- `BiographySection`: structured subsection of a persona biography
- `Statement`: supporting claim record used by downstream tooling

`Statement` includes `sid`, `category`, `claim`, `support`, and `confidence`.

`QAPair` includes `qid`, `type`, `question`, `answer`, `difficulty`, `answer_format`, `choices`, `correct_choice_index`, `evidence_sids`, and `tags`.

## Persona fields

`PersonaData` includes:

- `id`
- `persona`
- `templated_view`
- `biography_view`
- `statements_view`
- `sections`
- `statements`

It also exposes convenience helpers:

- `get_section(section_id)`
- `get_sections_by_category(category)`
- `sections_by_id`
- `sections_by_category`
- `section_categories`

It also exposes `name` as a derived property.

## Queries

```python
persona = dataset[0]

qa_pairs = dataset.get_qa(persona.id)
qa_pairs = dataset.get_qa(persona.id, type="explicit", difficulty=[2, 3])

questions = dataset.questions(persona.id, type="implicit")
loaded_persona = dataset.get_persona("p1")
```

`get_qa()` returns typed `QAPair` records. `questions()` returns question strings only.

## Notes

- `difficulty` accepts one level or a list of levels.
- `type` can be `"explicit"` or `"implicit"`.
- `sample_size` can limit how many personas are kept in memory.
- `sample_size` keeps a leading slice rather than sampling randomly.
- The loader keeps the dataset eager and notebook-friendly rather than streaming.

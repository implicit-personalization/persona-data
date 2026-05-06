# SynthPersona

`SynthPersonaDataset` loads persona profiles plus QA pairs from Hugging Face and exposes a small in-memory API for analysis and prompt generation.

## Loader

```python
from persona_data.synth_persona import SynthPersonaDataset

dataset = SynthPersonaDataset()
small = SynthPersonaDataset(sample_size=100)
```

The default dataset source is `implicit-personalization/synth-persona`. The loader downloads these files:

- `dataset_personas.jsonl`
- `dataset_qa.jsonl`
- `implicit_shared_mc_bank.json`

`implicit_shared_mc_bank.json` is read internally to hydrate `QAPair.related_qids`; user code normally only calls `get_qa()`. `sample_size` keeps the leading personas and filters QA rows to the loaded persona IDs.

## Records

- `PersonaData`: top-level persona record
- `QAPair`: question-answer pair with type, item_type, and optional multiple-choice fields
- `Statement`: supporting claim record used by downstream tooling

`Statement` includes `sid`, `category`, `claim`, and `support_turns`.

`QAPair` keeps the cross-dataset interface small:

- core task fields: `qid`, `type`, `scope`, `item_type`, `question`, and `answer`
- multiple-choice fields: `choices`, `choice_labels`, `correct_choice_index`, and `correct_choice_letter`
- common query fields: `source`, `bank_id`, and `family_id`
- common provenance fields: `evidence_sids`, `tags`, and `is_baseline`
- split helpers: `split_group_id` and `related_qids`
- `metadata`: a dictionary containing dataset-specific public fields such as `source_turn`, `attribute_keys`, `statement_category`, `family_name`, `domain`, and `axis`

`QAPair.split_group_id` is a single leakage-group key when one exists. For SynthPersona explicit rows, it is `explicit:{bank_id}`. `QAPair.related_qids` is used when a row has many source rows. For SynthPersona implicit shared multiple-choice rows, it contains the public qids of individual implicit free-response rows used as source evidence for the shared item.

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
train_frq = dataset.get_qa(persona.id, item_type="frq")
test_mcq = dataset.get_qa(persona.id, item_type="mcq", scope="shared")
explicit_interview = dataset.get_qa(
    persona.id,
    type="explicit",
    item_type="mcq",
    source="interview",
)

loaded_persona = dataset.get_persona("p1")
```

`get_qa()` returns typed `QAPair` records.

## Notes

- `type` can be `"explicit"` or `"implicit"`.
- `item_type` can be `"mcq"` (multiple-choice) or `"frq"` (free-response).
- `scope` can be `"individual"` (persona-specific row) or `"shared"` (common bank row).
- `source` is only populated for explicit rows and can be `"seed_attribute"`, `"interview"`, or `"statement"`.
- `bank_id` and `family_id` can filter related rows for analysis or split construction.
- `sample_size` keeps a leading slice rather than sampling randomly.
- The loader keeps the dataset eager and notebook-friendly rather than streaming.

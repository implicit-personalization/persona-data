# SynthPersona

`SynthPersonaDataset` loads persona profiles plus QA pairs from Hugging Face and exposes a small in-memory API for analysis and prompt generation.

## Loader

```python
from persona_data.synth_persona import SynthPersonaDataset

dataset = SynthPersonaDataset()
small = SynthPersonaDataset(sample_size=100)
```

The default dataset source is `implicit-personalization/synth-persona`. The loader reads:

- `dataset_personas.jsonl`
- `dataset_qa.jsonl`
- `implicit_shared_mc_bank.json` (used to hydrate `QAPair.related_frq_qids` on implicit shared MCQs)

`sample_size` keeps the leading personas and filters QA rows to the loaded persona IDs. The persona-less Assistant baseline is an ordinary persona row (`id="baseline_assistant"`, exported as `BASELINE_PERSONA_ID`).

Retrieve the baseline directly via `dataset.baseline` (or `dataset.get_persona(BASELINE_PERSONA_ID)`):

```python
dataset = SynthPersonaDataset()
baseline = dataset.baseline
for persona in dataset:
    ...
```

## Records

- `PersonaData`: top-level persona record
- `QAPair`: question-answer pair with type, item_type, and optional multiple-choice fields
- `Statement`: supporting claim record used by downstream tooling

`Statement` includes `sid`, `category`, `claim`, and `support_turns`.

`QAPair` fields:

| Field | Meaning |
|---|---|
| `qid` | Globally unique question id. |
| `type` | `"explicit"` (directly supported by a seed attribute / interview / statement) or `"implicit"` (inferred from the persona biography). |
| `item_type` | `"frq"` (free-response) or `"mcq"` (multiple-choice). |
| `scope` | `"individual"` (one persona) or `"shared"` (same item bank across personas). |
| `question`, `answer` | Question text and reference answer. |
| `choices`, `correct_choice_index` | MCQ-only; empty/`None` for FRQs. The final MCQ option is always `"Not enough information from the context."`. |
| `bank_id` | Stable item / source-slot identifier. For explicit rows, FRQ and MCQ rows that share a `bank_id` come from the same seed attribute, interview answer, or statement. Used for leakage-aware splits. |
| `related_frq_qids` | Implicit shared MCQs only: qids of individual implicit FRQs used as evidence when constructing the MCQ. Used for leakage-aware splits. |
| `evidence_sids` | Optional list of supporting `Statement.sid`s. Empty when the dataset row carries no statement evidence. |
| `tags` | Optional free-form string tags for downstream slicing or analysis. |

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

### Train/test split

`dataset.train_test_split(persona_id, n_train=None, seed=0)` returns `(train, test)` for one persona:

- **train**: individual free-response questions (both explicit and implicit). Pass `n_train=50` or another integer to cap the train slice, or `n_train=None` for no cap.
- **test**: shared multiple-choice questions (both explicit and implicit), preserved in full. The shared bank is the same item set for every persona, so per-persona test scores are directly comparable.
- **seed**: optional `int` that shuffles the train candidates before capping (reproducible). Test order is left untouched.

To avoid train→test leakage, train rows are dropped if their `bank_id` matches a test MCQ `bank_id` (explicit FRQ↔MCQ from the same source slot) or if their `qid` appears in a test MCQ's `related_frq_qids` (implicit MCQ built from that FRQ as evidence).

## Notes

- `type` can be `"explicit"` or `"implicit"`.
- `item_type` can be `"mcq"` (multiple-choice) or `"frq"` (free-response).
- `sample_size` keeps a leading slice rather than sampling randomly.
- The loader keeps the dataset eager and notebook-friendly rather than streaming.

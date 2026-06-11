# persona-data

[![Docs](https://img.shields.io/badge/docs-view-purple?logo=materialformkdocs)](https://implicit-personalization.github.io/persona-data/)
[![PyPI](https://img.shields.io/pypi/v/persona-data?logo=pypi&label=PyPI)](https://pypi.org/project/persona-data/)
[![Dataset](https://img.shields.io/badge/🤗-synth--persona-yellow)](https://huggingface.co/datasets/implicit-personalization/synth-persona)

Dataset loaders and prompt utilities for the [implicit-personalization](https://github.com/implicit-personalization)
research effort, built around **SynthPersona** — an open synthetic-persona dataset for studying, steering, and
personalizing language models.

## The SynthPersona dataset

[`implicit-personalization/synth-persona`](https://huggingface.co/datasets/implicit-personalization/synth-persona)
is a fully open synthetic persona dataset (~1.41 GB, English) for research on implicit personalization, persona
steering, and persona-grounded evaluation.

- **1,000 personas** built from structured seed attributes and expanded into biographies, interview transcripts,
  and supporting statements, plus a `baseline_assistant` control.
- **788k QA rows** across three axes:
  - `type`: **explicit** (supported by a seed/interview/statement) vs. **implicit** (inferred from the biography).
  - `scope`: **individual** (one persona) vs. **shared** (same item across all personas, directly comparable).
  - `item_type`: **FRQ** (free-response, for training) vs. **MCQ** (multiple-choice, for evaluation).
- **Shared MCQ banks**: 418 implicit + 57 explicit items reused across personas, with a curated
  `study_model_evaluable_v1` subset (231 items) for 7B-scale evaluation.
- **18 topic groups** (e.g. `future_hopes_and_values`, `stress_coping_and_support`) for sliced analyses.
- **Leakage-aware splits**: each MCQ tracks its source FRQs/seeds (`bank_id`, `related_frq_qids`), so FRQ-train /
  MCQ-test splits avoid contamination.

| QA rows     | Implicit / FRQ | Explicit / FRQ | Explicit / MCQ | Implicit / Shared MCQ | Explicit / Shared MCQ |
| ----------- | -------------: | -------------: | -------------: | --------------------: | --------------------: |
| Count       |         40,000 |        174,336 |         98,156 |               418,000 |                57,000 |
| Per persona |             40 |           ~174 |            ~98 |    418 (shared bank)  |     57 (shared bank)  |

See the [dataset card](https://huggingface.co/datasets/implicit-personalization/synth-persona) for the full schema.

## Installation

```bash
pip install persona-data    # or: uv add persona-data
```

The dataset is downloaded from Hugging Face on first use and cached locally.

## Quick start

```python
from persona_data.synth_persona import SynthPersonaDataset
from persona_data.prompts import format_prompt, format_messages

dataset = SynthPersonaDataset()
persona = dataset[0]

system_prompt = format_prompt(persona, "biography")
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Where did you grow up?"},
]

# Leakage-aware split: individual FRQs for train, shared MCQs for test.
train_qa, test_qa = dataset.train_test_split(persona.id)

# Slice by topic or by curated evaluation subset.
religion = dataset.get_qa(persona.id, type="implicit",
                          topic_group_id="religion_spirituality_and_meaning")
eval_mc  = dataset.get_qa(persona.id, item_type="mcq",
                          question_set="study_model_evaluable_v1")

# Minimal-pair counterfactual: same persona, one attribute swapped in the
# templated view (binary attributes default to the opposite value).
from persona_data.templated import swap_attribute
base, swapped = swap_attribute(dataset, persona.id, "speak_other_language")
```

Pass `sample_size=N` to load only the first `N` personas.

## What else is in the package

- `SynthPersonaDataset` — personas + QA pairs ([docs](https://implicit-personalization.github.io/persona-data/synth_persona/))
- `NemotronPersonasFranceDataset` / `NemotronPersonasUSADataset` — NVIDIA persona-only datasets ([docs](https://implicit-personalization.github.io/persona-data/nemotron_personas/))
- `prompts` — roleplay and multiple-choice formatting helpers ([docs](https://implicit-personalization.github.io/persona-data/prompts/))
- `templated` — single-attribute counterfactual swaps on the templated view ([docs](https://implicit-personalization.github.io/persona-data/templated/))
- `environment` — `set_seed`, `get_device`, `get_artifacts_dir`

Full API reference: <https://implicit-personalization.github.io/persona-data/>.

## Used by

- [persona-vectors](https://github.com/implicit-personalization/persona-vectors) — activation extraction and steering
- [persona-2-lora](https://github.com/implicit-personalization/persona-2-lora) — LoRA-based persona internalization

## Citation

If you use SynthPersona, please cite the
[dataset card](https://huggingface.co/datasets/implicit-personalization/synth-persona) and link back to this repo.

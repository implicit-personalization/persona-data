# Attribute Swaps

`persona_data.templated` builds single-attribute counterfactuals of the
SynthPersona `templated_view`. The v4.0 view is a deterministic render of the
structured `persona` attributes, so a swap edits one attribute and re-renders
the whole view — composite sentences (age + sex, city + state), the rephrased
`born_in_us` line, and the conditional religion line stay coherent, and the
two views are guaranteed to form a minimal pair.

## Swapping one attribute

```python
from persona_data.synth_persona import SynthPersonaDataset
from persona_data.templated import swap_attribute

dataset = SynthPersonaDataset()
persona_id = dataset.persona_ids[0]

# Binary attributes default to the opposite value.
base, swapped = swap_attribute(dataset, persona_id, "speak_other_language")

# Categorical / ordinal attributes need an explicit value, validated against
# the attribute schema. Numeric (age) and high-cardinality (city, state)
# attributes accept any value.
base, swapped = swap_attribute(dataset, persona_id, "religion", "Jewish")
base, swapped = swap_attribute(dataset, persona_id, "age", 70)

print(base.templated_view)     # ... My religion is Catholic. ...
print(swapped.templated_view)  # ... My religion is Jewish. ...
```

`swap_attribute` returns `(original, swapped)` `PersonaData` records. The
swapped persona gets the id `"<persona_id>__swap_<attribute>"`, an updated
`persona` dict, and a re-rendered `templated_view`. Its biography and
statement views are cleared because they still describe the original value —
biography-level swaps (statement / section based) are not implemented yet.

Inspect valid values for an attribute via `dataset.attribute_info(name)`
(`ordered_values` for ordinals, `seed_values_sorted_by_count` otherwise).

## Running a model on both prompts

Both records work everywhere a `PersonaData` does, so comparing the model's
behavior under the base and counterfactual persona is the usual
`format_prompt` + `format_messages` flow, once per record:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_data.prompts import format_messages, format_prompt
from persona_data.templated import swap_attribute

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

base, swapped = swap_attribute(dataset, persona_id, "speak_other_language")
question = "Do you speak any language besides English?"

for persona in (base, swapped):
    messages = [
        {"role": "system", "content": format_prompt(persona, "templated")},
        {"role": "user", "content": question},
    ]
    full_prompt, start = format_messages(
        messages, tokenizer, add_generation_prompt=True
    )
    inputs = tokenizer(
        full_prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=64)
    print(persona.id, "->", tokenizer.decode(output[0, start:], skip_special_tokens=True))
```

To score instead of generate, ask the persona's own seed-attribute MCQ — the
shared bank has one per explicit seed attribute, with `bank_id`
`"explicit_seed_attribute_<attribute>"`:

```python
from persona_data.prompts import format_mc_question, mc_correct_letter

qa = next(
    q
    for q in dataset.get_qa(persona_id, type="explicit", item_type="mcq", scope="shared")
    if q.bank_id == "explicit_seed_attribute_speak_other_language"
)
user_prompt = format_mc_question(qa)
correct = mc_correct_letter(qa)  # correct under the *base* persona
```

Under the swapped system prompt, a model that reads the profile faithfully
should pick the swapped value's letter instead of `correct`.

## Building a minimal-pair dataset

Binary swaps work for every regular persona, so a full dataset is one
comprehension (skip the baseline — it has no attributes to swap):

```python
from persona_data.synth_persona import BASELINE_PERSONA_ID

pairs = [
    swap_attribute(dataset, pid, "speak_other_language")
    for pid in dataset.persona_ids
    if pid != BASELINE_PERSONA_ID
]
```

For categorical attributes, pick the counterfactual value per persona (it
must differ from the current one), e.g. mapping every persona onto a fixed
target value and skipping those that already have it.

## Errors

`swap_attribute` raises `ValueError` when:

- the persona's stored `templated_view` does not match the v4.0 render (e.g.
  the `baseline_assistant` persona, or a future dataset format change), so a
  minimal pair cannot be guaranteed;
- `new_value` is omitted for a non-binary attribute, or is not in the
  schema's closed value set;
- the swap leaves the view unchanged (same value, or an attribute such as
  `street_address` that is not rendered in the view).

## Rendering directly

`render_templated_view(attrs)` renders the v4.0 view from a raw attribute
dict, and is what `swap_attribute` uses for both the integrity check and the
counterfactual view.

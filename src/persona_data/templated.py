"""Single-attribute counterfactual swaps on the SynthPersona templated view.

The ``templated_view`` shipped with SynthPersona (v4.0) is a deterministic
render of the structured ``persona`` attributes. A counterfactual is built by
editing one attribute and re-rendering the whole view, which keeps composite
sentences (age + sex, city + state) and conditional lines (religion) coherent
instead of patching anchor sentences with string replacement.
"""

from dataclasses import replace
from typing import Any

from persona_data.synth_persona import PersonaData, PersonaDataset


def render_templated_view(attrs: dict) -> str:
    """Render the v4.0 templated view from structured persona attributes."""
    born = "" if attrs["born_in_us"] == "Yes" else "not "
    lines = [
        f"I am {attrs['first_name']} {attrs['last_name']}.",
        f"I am a {attrs['age']}-year-old {attrs['sex'].lower()}.",
        f"I live in {attrs['city']}, {attrs['state']}.",
        f"I was {born}born in the United States.",
        f"My ethnicity is {attrs['ethnicity']}.",
        f"My race is {attrs['race']}.",
        f"My Hispanic origin is {attrs['hispanic_origin']}.",
        f"My political views are {attrs['political_views']}.",
        f"My party identification is {attrs['party_identification']}.",
        f"My marital status is {attrs['marital_status']}.",
        f"My current work status is {attrs['work_status']}.",
        f"My highest degree received is {attrs['highest_degree_received']}.",
        f"My religion is {attrs['religion']}.",
        f"My answer to whether I speak another language is {attrs['speak_other_language']}.",
        f"My total wealth is {attrs['total_wealth']}.",
        f"When I was 16, I lived in {attrs['residence_at_16']}.",
        f"My answer to whether I have lived in the same place since age 16 is {attrs['same_residence_since_16']}.",
        f"My family structure at age 16 was {attrs['family_structure_at_16']}.",
        f"My family income at age 16 was {attrs['family_income_at_16']}.",
        f"My father's highest degree was {attrs['fathers_highest_degree']}.",
        f"My mother's highest degree was {attrs['mothers_highest_degree']}.",
        f"My mother's work history was {attrs['mothers_work_history']}.",
        f"My U.S. citizenship status is {attrs['us_citizenship_status']}.",
    ]
    if attrs["religion"] == "None":
        lines.remove("My religion is None.")
    return "\n".join(lines)


def _resolve_new_value(
    attribute: str, field_info: dict, old_value: Any, new_value: Any
) -> Any:
    """Default binaries to the opposite value; validate closed-set choices."""
    known = field_info.get("ordered_values") or [
        v["value"] for v in field_info.get("seed_values_sorted_by_count", [])
    ]
    if new_value is None:
        others = [v for v in known if v != old_value]
        if field_info.get("kind") != "binary" or len(others) != 1:
            raise ValueError(
                f"new_value is required for non-binary attribute {attribute!r}"
            )
        return others[0]
    if (
        known
        and field_info.get("kind") != "numeric"
        and not field_info.get("high_cardinality")
        and new_value not in known
    ):
        raise ValueError(
            f"Value {new_value!r} for {attribute!r} not in known values {known}"
        )
    return new_value


def swap_attribute(
    dataset: PersonaDataset,
    persona_id: str,
    attribute: str,
    new_value: Any = None,
) -> tuple[PersonaData, PersonaData]:
    """Return ``(original, swapped)`` personas differing in one attribute.

    The swapped persona has ``attribute`` set to ``new_value`` and its
    ``templated_view`` re-rendered, so the two views form a minimal pair.
    ``new_value`` may be omitted for binary attributes (defaults to the
    opposite value); categorical and numeric attributes require one, validated
    against the attribute schema when it lists a closed value set. The
    biography and statement views of the swapped persona are cleared because
    they still describe the original value.
    """
    persona = dataset.get_persona(persona_id)
    if persona is None:
        raise KeyError(persona_id)
    if attribute not in persona.persona:
        raise ValueError(f"Persona {persona_id!r} has no attribute {attribute!r}")
    try:
        rendered = render_templated_view(persona.persona)
    except KeyError:
        rendered = None  # incomplete attributes, e.g. the baseline persona
    if rendered != persona.templated_view:
        raise ValueError(
            f"templated_view of {persona_id!r} does not match the v4.0 template; "
            "cannot guarantee a minimal pair"
        )

    new_value = _resolve_new_value(
        attribute,
        dataset.attribute_info(attribute),
        persona.persona[attribute],
        new_value,
    )
    attrs = {**persona.persona, attribute: new_value}
    swapped_view = render_templated_view(attrs)
    if swapped_view == persona.templated_view:
        raise ValueError(
            f"Swapping {attribute!r} to {new_value!r} leaves the view unchanged"
        )
    swapped = replace(
        persona,
        id=f"{persona.id}__swap_{attribute}",
        persona=attrs,
        templated_view=swapped_view,
        biography_view="",
        statements_view="",
        statements=[],
    )
    return persona, swapped

"""Pytest suite for the three dataset loaders.

These tests do not hit the network — `hf_hub_download` and `list_repo_files`
are patched to point at local fixtures.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from persona_data.environment import get_device
from persona_data.nemotron_personas import (
    NemotronPersonasFranceDataset,
    NemotronPersonasUSADataset,
    _extract_display_name,
    _split_name,
)
from persona_data.persona_guess import PersonaGuessDataset
from persona_data.prompts import (
    format_mc_question,
    format_prompt,
    mc_answer_only_instruction,
)
from persona_data.synth_persona import (
    BASELINE_PERSONA_ID,
    BASELINE_PERSONA_NAME,
    PersonaData,
    PersonaDataset,
    QAPair,
    SynthPersonaDataset,
)

# --------------------------------------------------------------------------- #
# environment / prompts                                                       #
# --------------------------------------------------------------------------- #


def test_get_device_returns_known_kind():
    assert get_device().type in {"cpu", "cuda", "mps"}


def test_format_prompt_non_empty():
    assert format_prompt("hello")


def test_format_prompt_rejects_unknown_mode():
    with pytest.raises(ValueError):
        format_prompt("hello", mode="mc")


def test_baseline_persona_id_constant():
    """BASELINE_PERSONA_ID matches the public id used by the SynthPersona dataset."""
    assert BASELINE_PERSONA_ID == "baseline_assistant"
    assert BASELINE_PERSONA_NAME == "Assistant"


def test_format_prompt_with_persona_view():
    persona = PersonaData(
        id="p1",
        persona={"first_name": "A", "last_name": "B"},
        templated_view="TEMPLATED",
        biography_view="BIO",
    )
    assert "BIO" in format_prompt(persona.biography_view)
    assert "TEMPLATED" in format_prompt(persona.templated_view)
    assert "BIO" in format_prompt(persona, "biography")
    assert "TEMPLATED" in format_prompt(persona, "templated")


def test_format_prompt_rejects_unknown_variant():
    persona = PersonaData(
        id="p1",
        persona={"first_name": "A", "last_name": "B"},
        templated_view="TEMPLATED",
        biography_view="BIO",
    )
    with pytest.raises(ValueError, match="Unsupported persona prompt variant"):
        format_prompt(persona, variant="statements")  # type: ignore[arg-type]


def test_format_prompt_variant_forwards_mode():
    persona = PersonaData(
        id="p1",
        persona={"first_name": "A", "last_name": "B"},
        templated_view="TEMPLATED",
        biography_view="BIO",
    )

    prompt = format_prompt(persona, "biography", mode="conversational")
    assert "BIO" in prompt
    assert "Answer naturally and conversationally" in prompt


def test_mc_answer_only_instruction_uses_actual_labels():
    assert (
        mc_answer_only_instruction(3)
        == "Answer only with the correct choice label (A, B, C)."
    )


def test_format_mc_question_appends_instruction():
    qa = QAPair(
        qid="q1",
        type="explicit",
        item_type="mcq",
        scope="individual",
        question="Pick one?",
        answer="Beta",
        choices=["Alpha", "Beta"],
        correct_choice_index=1,
    )

    prompt = format_mc_question(qa)
    assert prompt.startswith("Pick one?\n\nA. Alpha\nB. Beta")
    assert prompt.endswith("Answer only with the correct choice label (A, B).")


# --------------------------------------------------------------------------- #
# SynthPersona / PersonaDataset                                               #
# --------------------------------------------------------------------------- #


def _write_persona_fixture(tmp: Path) -> tuple[Path, Path]:
    personas_path = tmp / "personas.jsonl"
    qa_path = tmp / "qa.jsonl"

    personas = [
        {
            "id": "p1",
            "persona": {
                "first_name": "Ada",
                "last_name": "Lovelace",
                "age": 36,
                "state": "CA",
                "political_views": "Liberal",
            },
            "templated_view": "Ada Lovelace",
            "biography_view": "Bio one",
        },
        {
            "id": "p2",
            "persona": {
                "first_name": "Grace",
                "last_name": "Hopper",
                "age": 45,
                "state": "NY",
                "political_views": "Moderate, middle of the road",
            },
            "templated_view": "Grace Hopper",
            "biography_view": "Bio two",
        },
        {
            "id": BASELINE_PERSONA_ID,
            "persona": {
                "first_name": BASELINE_PERSONA_NAME,
                "last_name": "",
            },
            "templated_view": BASELINE_PERSONA_NAME,
            "biography_view": BASELINE_PERSONA_NAME,
        },
    ]
    qa_rows = [
        # Explicit FRQ that shares its bank_id with a shared explicit MCQ
        # below — leakage filter should drop it from the train split.
        {
            "id": "p1",
            "qid": "q1",
            "type": "explicit",
            "item_type": "frq",
            "scope": "individual",
            "question": "Who is it?",
            "answer": "Ada",
            "bank_id": "explicit_name",
        },
        {
            "id": "p1",
            "qid": "q2",
            "type": "implicit",
            "item_type": "mcq",
            "scope": "individual",
            "question": "Hint?",
            "answer": "math",
            "choices": ["math", "science"],
            "correct_choice_index": 0,
        },
        {
            "id": "p1",
            "qid": "q3",
            "type": "explicit",
            "item_type": "mcq",
            "scope": "individual",
            "question": "Field?",
            "answer": "math",
            "choices": ["math", "art"],
            "correct_choice_index": 0,
        },
        # Shared implicit MCQ whose related_frq_qids include p1-q7 (an
        # implicit FRQ below) — that FRQ should be dropped from train.
        {
            "id": "p1",
            "qid": "q4",
            "type": "implicit",
            "item_type": "mcq",
            "scope": "shared",
            "question": "Shared implicit?",
            "answer": "yes",
            "choices": ["yes", "no"],
            "correct_choice_index": 0,
            "bank_id": "implicit_shared_1",
        },
        # Shared explicit MCQ that collides with q1 via bank_id.
        {
            "id": "p1",
            "qid": "q5",
            "type": "explicit",
            "item_type": "mcq",
            "scope": "shared",
            "question": "Shared explicit?",
            "answer": "yes",
            "choices": ["yes", "no"],
            "correct_choice_index": 0,
            "bank_id": "explicit_name",
        },
        {
            "id": "p2",
            "qid": "q6",
            "type": "explicit",
            "item_type": "frq",
            "scope": "individual",
            "question": "Who?",
            "answer": "Grace",
        },
        # Implicit FRQ used as evidence for shared implicit MCQ q4.
        {
            "id": "p1",
            "qid": "q7",
            "type": "implicit",
            "item_type": "frq",
            "scope": "individual",
            "question": "How does Ada talk about math?",
            "answer": "Enthusiastically.",
        },
        # Implicit FRQ NOT referenced by any shared MCQ — should survive.
        {
            "id": "p1",
            "qid": "q8",
            "type": "implicit",
            "item_type": "frq",
            "scope": "individual",
            "question": "How does Ada talk about art?",
            "answer": "Curiously.",
        },
    ]

    personas_path.write_text("\n".join(json.dumps(p) for p in personas) + "\n")
    qa_path.write_text("\n".join(json.dumps(q) for q in qa_rows) + "\n")
    return personas_path, qa_path


def test_persona_dataset_loads_personas_and_qa(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path)

    assert len(ds) == 3
    assert [p.id for p in ds] == ["p1", "p2", BASELINE_PERSONA_ID]
    assert ds.persona_ids == ("p1", "p2", BASELINE_PERSONA_ID)
    assert ds[0].name == "Ada Lovelace"
    assert ds.get_persona("p2").name == "Grace Hopper"
    assert ds.baseline.id == BASELINE_PERSONA_ID
    assert ds.get_persona("missing") is None
    assert ds.get_persona(BASELINE_PERSONA_ID).templated_view == BASELINE_PERSONA_NAME
    assert ds.attribute_names == ("age", "political_views", "state")


def test_persona_dataset_attribute_helpers_without_schema(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path)
    persona_ids = ["p2", "p1"]

    assert ds.attribute_values("state") == ["CA", "NY", None]
    assert ds.attribute_values("state", persona_ids) == ["NY", "CA"]
    assert ds.attribute_values("age", persona_ids, encode=True) == [45.0, 36.0]
    assert ds.attribute_info("age") == {}
    assert ds.attribute_values("missing", persona_ids, missing="unknown") == [
        "unknown",
        "unknown",
    ]
    with pytest.raises(KeyError):
        ds.attribute_values("state", ["missing"])


def test_persona_dataset_attribute_schema_drives_names_and_ordinal_encoding(
    tmp_path: Path,
):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    schema_path = tmp_path / "attribute_schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "persona_fields": {
                    "first_name": {"identifier": True},
                    "last_name": {"identifier": True},
                    "age": {"kind": "numeric", "analysis_recommendation": "keep"},
                    "born_in_us": {
                        "kind": "binary",
                        "analysis_recommendation": "keep",
                    },
                    "political_views": {
                        "kind": "ordinal",
                        "analysis_recommendation": "keep",
                        "ordered_values": [
                            "Extremely liberal",
                            "Liberal",
                            "Moderate, middle of the road",
                            "Conservative",
                        ],
                    },
                    "state": {
                        "kind": "nominal",
                        "analysis_recommendation": "inspect",
                    },
                    "street_address": {"analysis_recommendation": "drop"},
                }
            }
        )
    )

    ds = PersonaDataset(personas_path, qa_path, attribute_schema_path=schema_path)
    persona_ids = ["p1", "p2"]

    ds.get_persona("p1").persona["born_in_us"] = "Yes"

    assert ds.attribute_names == ("age", "born_in_us", "political_views", "state")
    assert ds.attribute_info("state") == {
        "kind": "nominal",
        "analysis_recommendation": "inspect",
    }
    assert ds.attribute_values("state", persona_ids, encode=True) == [None, None]
    assert ds.attribute_values("political_views", persona_ids) == [
        "Liberal",
        "Moderate, middle of the road",
    ]
    assert ds.attribute_values("political_views", persona_ids, encode=True) == [
        1.0,
        2.0,
    ]
    assert ds.attribute_values("born_in_us", ["p1"], encode=True) == [1.0]

    ds.get_persona("p1").persona["political_views"] = "Unknown"
    with pytest.raises(ValueError, match="Unknown ordinal value"):
        ds.attribute_values("political_views", ["p1"], encode=True)


def test_synth_persona_dataset_tolerates_missing_attribute_schema(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    bank_path = tmp_path / "implicit_shared_mc_bank.json"
    bank_path.write_text(json.dumps({"items": []}))
    requested_files: list[str] = []

    def fake_hf_hub_download(hf_repo: str, filename: str, repo_type: str) -> str:
        assert hf_repo == "test/repo"
        assert repo_type == "dataset"
        requested_files.append(filename)
        if filename == "dataset_personas.jsonl":
            return str(personas_path)
        if filename == "dataset_qa.jsonl":
            return str(qa_path)
        if filename == "implicit_shared_mc_bank.json":
            return str(bank_path)
        if filename == "attribute_schema.json":
            from huggingface_hub.utils import LocalEntryNotFoundError

            raise LocalEntryNotFoundError("missing attribute_schema.json")
        raise AssertionError(f"Unexpected download: {filename}")

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
        ds = SynthPersonaDataset(hf_repo="test/repo")
        assert "attribute_schema.json" not in requested_files
        assert ds.attribute_names == ("age", "political_views", "state")
        assert ds.attribute_schema == {}


def test_synth_persona_dataset_loads_question_registry_lazily(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    bank_path = tmp_path / "implicit_shared_mc_bank.json"
    bank_path.write_text(json.dumps({"items": []}))
    registry_path = tmp_path / "question_registry.jsonl"
    registry_path.write_text(
        json.dumps(
            {
                "bank_id": "implicit_shared_1",
                "topic_group_id": "religion_spirituality_and_meaning",
            }
        )
        + "\n"
    )
    requested_files: list[str] = []

    def fake_hf_hub_download(hf_repo: str, filename: str, repo_type: str) -> str:
        assert hf_repo == "test/repo"
        assert repo_type == "dataset"
        requested_files.append(filename)
        if filename == "dataset_personas.jsonl":
            return str(personas_path)
        if filename == "dataset_qa.jsonl":
            return str(qa_path)
        if filename == "implicit_shared_mc_bank.json":
            return str(bank_path)
        if filename == "question_registry.jsonl":
            return str(registry_path)
        raise AssertionError(f"Unexpected download: {filename}")

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
        ds = SynthPersonaDataset(hf_repo="test/repo")
        assert "question_registry.jsonl" not in requested_files
        assert [
            q.qid
            for q in ds.get_qa(
                "p1",
                topic_group_id="religion_spirituality_and_meaning",
            )
        ] == ["q4"]
        assert "question_registry.jsonl" in requested_files



def test_persona_dataset_baseline_is_a_normal_sampled_persona(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path, sample_size=3)

    assert [p.id for p in ds] == ["p1", "p2", BASELINE_PERSONA_ID]
    assert ds.baseline is ds[2]
    assert ds.get_persona(BASELINE_PERSONA_ID) is ds.baseline


def test_persona_dataset_qa_filters_by_type_and_item_type(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path)

    assert [q.qid for q in ds.get_qa("p1")] == [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q7",
        "q8",
    ]
    assert [q.qid for q in ds.get_qa("p1", type="explicit")] == ["q1", "q3", "q5"]
    assert [q.qid for q in ds.get_qa("p1", item_type="mcq")] == ["q2", "q3", "q4", "q5"]
    assert [q.qid for q in ds.get_qa("p1", scope="shared")] == ["q4", "q5"]


def test_persona_dataset_qa_filters_by_question_registry(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    registry_path = tmp_path / "question_registry.jsonl"
    registry_rows = [
        {
            "qid": "q2",
            "topic_group_id": "work_identity_and_competence",
        },
        {
            "bank_id": "implicit_shared_1",
            "topic_group_id": "religion_spirituality_and_meaning",
            "question_sets": ["qa_eval_v1"],
        },
        {
            "bank_id": "explicit_name",
            "topic_group_id": "demographics_and_background",
            "question_sets": ["explicit_control_v1"],
        },
    ]
    registry_path.write_text("\n".join(json.dumps(row) for row in registry_rows) + "\n")
    ds = PersonaDataset(personas_path, qa_path, question_registry_path=registry_path)

    assert [
        q.qid
        for q in ds.get_qa("p1", topic_group_id="work_identity_and_competence")
    ] == ["q2"]
    assert [
        q.qid
        for q in ds.get_qa(
            "p1",
            type="implicit",
            topic_group_id="religion_spirituality_and_meaning",
        )
    ] == ["q4"]
    assert [
        q.qid
        for q in ds.get_qa(
            "p1",
            type="explicit",
            topic_group_id="demographics_and_background",
        )
    ] == ["q1", "q5"]
    assert [
        q.qid
        for q in ds.get_qa(
            "p1",
            item_type="mcq",
            question_set="qa_eval_v1",
        )
    ] == ["q4"]
    assert [
        q.qid
        for q in ds.get_qa(
            "p1",
            type="implicit",
            item_type="mcq",
            topic_group_id="religion_spirituality_and_meaning",
            question_set="qa_eval_v1",
        )
    ] == ["q4"]
    assert (
        ds.get_qa(
            "p1",
            topic_group_id="work_identity_and_competence",
            question_set="qa_eval_v1",
        )
        == []
    )


def test_persona_dataset_qa_metadata_filters_require_registry(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path)

    with pytest.raises(ValueError, match="Question registry"):
        ds.get_qa("p1", topic_group_id="religion_spirituality_and_meaning")


def test_persona_dataset_rejects_invalid_question_registry(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    registry_path = tmp_path / "question_registry.jsonl"
    registry_path.write_text(
        json.dumps({"qid": "q2", "question_sets": "qa_eval_v1"}) + "\n"
    )
    ds = PersonaDataset(personas_path, qa_path, question_registry_path=registry_path)

    with pytest.raises(ValueError, match="question_sets"):
        ds.get_qa("p1", question_set="qa_eval_v1")


def test_persona_dataset_train_test_split_drops_leaking_train_rows(tmp_path: Path):
    """Train FRQs that share a bank_id or are listed as evidence for a test
    MCQ are dropped; test set is preserved in full."""
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(
        personas_path,
        qa_path,
        related_frq_qids_by_bank_id={"implicit_shared_1": ["q7"]},
    )

    train, test = ds.train_test_split("p1")
    # q1 dropped (bank_id collision with q5), q7 dropped (evidence for q4),
    # q8 kept (no leak).
    assert [q.qid for q in train] == ["q8"]
    assert [q.qid for q in test] == ["q4", "q5"]


def test_persona_dataset_train_test_split_respects_n_train(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    # No bank mapping -> no implicit leakage filter; q1 still drops via bank_id.
    ds = PersonaDataset(personas_path, qa_path)

    train, _ = ds.train_test_split("p1", n_train=1)
    assert [q.qid for q in train] == ["q7"]


def test_persona_dataset_train_test_split_seed_shuffles_reproducibly(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path)

    default_no_cap, _ = ds.train_test_split("p1")
    a, _ = ds.train_test_split("p1", n_train=None, seed=0)
    default_seed, _ = ds.train_test_split("p1", n_train=None)
    b, _ = ds.train_test_split("p1", n_train=None, seed=0)
    c, _ = ds.train_test_split("p1", n_train=None, seed=1)
    no_seed, _ = ds.train_test_split("p1", n_train=None, seed=None)

    assert [q.qid for q in default_no_cap] == [q.qid for q in default_seed]
    qids_a = [q.qid for q in a]
    qids_no_seed = [q.qid for q in no_seed]
    assert qids_a == [q.qid for q in default_seed]
    assert qids_a == [q.qid for q in b]  # reproducible
    assert sorted(qids_a) == sorted(qids_no_seed)  # same items, possibly reordered
    # at least one of the two seeds reorders relative to dataset order
    assert qids_a != qids_no_seed or [q.qid for q in c] != qids_no_seed


def test_persona_dataset_sample_size_drops_trailing_personas_and_their_qa(
    tmp_path: Path,
):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path, sample_size=1)

    assert [p.id for p in ds] == ["p1"]
    assert ds.get_qa("p2") == []


def test_persona_dataset_sample_size_zero_is_empty(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path, sample_size=0)

    assert len(ds) == 0
    assert list(ds) == []


# --------------------------------------------------------------------------- #
# PersonaGuessDataset                                                         #
# --------------------------------------------------------------------------- #


def _write_games_fixture(path: Path) -> Path:
    games = [
        {
            "game_id": "g1",
            "persona_a_id": "pa",
            "persona_b_id": "pb",
            "turns": [
                {"round": 1, "asker": "A", "question": "Q1a", "answer": "A1a"},
                {"round": 2, "asker": "B", "question": "Q1b", "answer": "A1b"},
            ],
        },
        {
            "game_id": "g2",
            "persona_a_id": "pc",
            "persona_b_id": "pd",
            "turns": [
                {"round": 1, "asker": "B", "question": "Q2", "answer": "A2"},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(g) for g in games) + "\n")
    return path


def test_persona_guess_loads_and_filters_by_player(tmp_path: Path):
    games_path = _write_games_fixture(tmp_path / "games.jsonl")

    with patch("persona_data.persona_guess.hf_hub_download", return_value=games_path):
        games = PersonaGuessDataset()

    assert len(games) == 2
    assert [g.game_id for g in games] == ["g1", "g2"]
    assert [t.question for t in games.get_qa("g1")] == ["Q1a", "Q1b"]
    assert [t.question for t in games.get_qa("g1", player="A")] == ["Q1a"]


def test_persona_guess_sample_size_limits_games(tmp_path: Path):
    games_path = _write_games_fixture(tmp_path / "games.jsonl")

    with patch("persona_data.persona_guess.hf_hub_download", return_value=games_path):
        games = PersonaGuessDataset(sample_size=1)

    assert len(games) == 1
    assert games[0].game_id == "g1"
    with pytest.raises(KeyError):
        games.get_qa("g2")


# --------------------------------------------------------------------------- #
# NemotronPersonasFranceDataset                                               #
# --------------------------------------------------------------------------- #


def _row(uuid: str, persona_text: str, **overrides) -> dict:
    base = {
        "uuid": uuid,
        "persona": persona_text,
        "cultural_background": "French",
        "skills_and_expertise": "Analysis",
        "hobbies_and_interests": "Music",
        "career_goals_and_ambitions": "Research",
        "sex": "Femme",
        "age": 30,
        "marital_status": "Célibataire",
        "household_type": "Personne seule",
        "education_level": "Bac+5",
        "occupation": "Cadres",
        "commune": "Paris",
        "departement": "Paris",
        "country": "France",
    }
    base.update(overrides)
    return base


def _usa_row(uuid: str, persona_text: str, **overrides) -> dict:
    base = {
        "uuid": uuid,
        "persona": persona_text,
        "cultural_background": "American",
        "skills_and_expertise": "Analysis",
        "skills_and_expertise_list": ["Analysis"],
        "hobbies_and_interests": "Music",
        "hobbies_and_interests_list": ["Music"],
        "career_goals_and_ambitions": "Research",
        "sex": "Female",
        "age": 30,
        "marital_status": "never_married",
        "education_level": "bachelors",
        "bachelors_field": "stem",
        "occupation": "scientist",
        "city": "Madison",
        "state": "WI",
        "zipcode": "53717",
        "country": "USA",
    }
    base.update(overrides)
    return base


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def nemotron_shards(tmp_path: Path):
    shard_1 = tmp_path / "train-00000-of-00002.parquet"
    shard_2 = tmp_path / "train-00001-of-00002.parquet"

    _write_parquet(
        shard_1,
        [
            _row("p1", "Ada Lovelace est une mathématicienne britannique."),
            _row("p2", "Marie Curie est une chercheuse en chimie."),
        ],
    )
    _write_parquet(
        shard_2,
        [
            _row("p3", "Sophie Germain est une spécialiste des mathématiques."),
            _row("p4", "?? inconnu"),  # no capitalised leader → fallback path
        ],
    )

    mapping = {
        "data/train-00000-of-00002.parquet": shard_1,
        "data/train-00001-of-00002.parquet": shard_2,
    }

    with (
        patch(
            "persona_data.nemotron_personas.list_repo_files",
            return_value=list(mapping),
        ),
        patch(
            "persona_data.nemotron_personas.hf_hub_download",
            side_effect=lambda _repo, filename, repo_type="dataset": str(
                mapping[filename]
            ),
        ),
    ):
        yield mapping


@pytest.fixture
def nemotron_usa_shards(tmp_path: Path):
    shard_1 = tmp_path / "train-00000-of-00002.parquet"
    shard_2 = tmp_path / "train-00001-of-00002.parquet"

    _write_parquet(
        shard_1,
        [
            _usa_row("u1", "Mary Alberti is a front-line food service specialist."),
            _usa_row("u2", "Alicia Gonzalez is a machine learning researcher."),
        ],
    )
    _write_parquet(
        shard_2,
        [
            _usa_row("u3", "Deeva Cintron is a community finance volunteer."),
            _usa_row("u4", "?? unknown"),
        ],
    )

    mapping = {
        "data/train-00000-of-00002.parquet": shard_1,
        "data/train-00001-of-00002.parquet": shard_2,
    }

    with (
        patch(
            "persona_data.nemotron_personas.list_repo_files",
            return_value=list(mapping),
        ),
        patch(
            "persona_data.nemotron_personas.hf_hub_download",
            side_effect=lambda _repo, filename, repo_type="dataset": str(
                mapping[filename]
            ),
        ),
    ):
        yield mapping


def test_nemotron_loads_requested_sample(nemotron_shards):
    ds = NemotronPersonasFranceDataset(sample_size=2)

    assert len(ds) == 2
    assert [p.id for p in ds] == ["p1", "p2"]
    assert ds[0].name == "Ada Lovelace"
    assert ds.supports_qa is False


def test_nemotron_templated_view_includes_expected_sections(nemotron_shards):
    ds = NemotronPersonasFranceDataset(sample_size=1)
    view = ds[0].templated_view

    assert view.startswith("Name: Ada Lovelace")
    assert "Location: Paris, Paris, France" in view
    assert "Career goals and ambitions: Research" in view


def test_nemotron_cross_shard_slicing(nemotron_shards):
    ds = NemotronPersonasFranceDataset(sample_size=3)

    assert [p.id for p in ds] == ["p1", "p2", "p3"]
    assert ds[2].name == "Sophie Germain"


def test_nemotron_get_persona_miss_returns_none(nemotron_shards):
    ds = NemotronPersonasFranceDataset(sample_size=2)

    assert ds.get_persona("missing") is None
    assert ds.get_persona("p1").id == "p1"


def test_nemotron_name_fallback_uses_uuid_when_no_match(nemotron_shards):
    ds = NemotronPersonasFranceDataset(sample_size=4)

    persona_p4 = ds.get_persona("p4")
    assert persona_p4 is not None
    assert persona_p4.persona["first_name"] == "p4"
    assert persona_p4.persona["last_name"] == ""
    assert persona_p4.name == "p4"


def test_nemotron_usa_loads_requested_sample(nemotron_usa_shards):
    ds = NemotronPersonasUSADataset(sample_size=2)

    assert len(ds) == 2
    assert [p.id for p in ds] == ["u1", "u2"]
    assert ds[0].name == "Mary Alberti"
    assert ds.supports_qa is False


def test_nemotron_usa_templated_view_includes_expected_sections(
    nemotron_usa_shards,
):
    ds = NemotronPersonasUSADataset(sample_size=1)
    view = ds[0].templated_view

    assert view.startswith("Name: Mary Alberti")
    assert "Location: Madison, WI, 53717, USA" in view
    assert "Bachelors field: stem" in view


def test_nemotron_usa_cross_shard_slicing(nemotron_usa_shards):
    ds = NemotronPersonasUSADataset(sample_size=3)

    assert [p.id for p in ds] == ["u1", "u2", "u3"]
    assert ds[2].name == "Deeva Cintron"


def test_nemotron_usa_get_persona_miss_returns_none(nemotron_usa_shards):
    ds = NemotronPersonasUSADataset(sample_size=2)

    assert ds.get_persona("missing") is None
    assert ds.get_persona("u1").id == "u1"


def test_nemotron_usa_name_fallback_uses_uuid_when_no_match(nemotron_usa_shards):
    ds = NemotronPersonasUSADataset(sample_size=4)

    persona_u4 = ds.get_persona("u4")
    assert persona_u4 is not None
    assert persona_u4.persona["first_name"] == "u4"
    assert persona_u4.persona["last_name"] == ""
    assert persona_u4.name == "u4"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ada Lovelace est une mathématicienne.", "Ada Lovelace"),
        ("Jean-Paul Sartre écrit.", "Jean-Paul Sartre"),
        ("Zoé est curieuse.", "Zoé"),
        ("", "fallback-id"),
        ("?? inconnu", "fallback-id"),
    ],
)
def test_extract_display_name(text, expected):
    assert _extract_display_name(text, "fallback-id") == expected


def test_split_name_falls_back_to_id_for_empty():
    assert _split_name("", "uuid-1") == ("uuid-1", "")
    assert _split_name("Solo", "uuid-1") == ("Solo", "")
    assert _split_name("Marie Sklodowska Curie", "uuid-1") == (
        "Marie",
        "Sklodowska Curie",
    )

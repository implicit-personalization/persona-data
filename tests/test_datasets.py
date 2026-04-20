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
    _extract_display_name,
    _split_name,
)
from persona_data.persona_guess import PersonaGuessDataset
from persona_data.prompts import (
    format_mc_question,
    format_roleplay_prompt,
    mc_answer_only_instruction,
)
from persona_data.synth_persona import PersonaDataset, QAPair

# --------------------------------------------------------------------------- #
# environment / prompts                                                       #
# --------------------------------------------------------------------------- #


def test_get_device_returns_known_kind():
    assert get_device().type in {"cpu", "cuda", "mps"}


def test_format_roleplay_prompt_non_empty():
    assert format_roleplay_prompt("hello")


def test_format_roleplay_prompt_rejects_unknown_mode():
    with pytest.raises(ValueError):
        format_roleplay_prompt("hello", mode="mc")


def test_mc_answer_only_instruction_uses_actual_labels():
    assert (
        mc_answer_only_instruction(3)
        == "Answer only with the correct choice label (A, B, C)."
    )


def test_format_mc_question_appends_instruction():
    qa = QAPair(
        qid="q1",
        type="explicit",
        question="Pick one?",
        answer="Beta",
        difficulty=1,
        answer_format="choice",
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
            "persona": {"first_name": "Ada", "last_name": "Lovelace"},
            "templated_view": "Ada Lovelace",
            "biography_view": "Bio one",
            "sections": [
                {
                    "section_id": "s1",
                    "category": "career",
                    "title": "Career",
                    "paragraphs": [{"text": "Worked on math."}],
                }
            ],
        },
        {
            "id": "p2",
            "persona": {"first_name": "Grace", "last_name": "Hopper"},
            "templated_view": "Grace Hopper",
            "biography_view": "Bio two",
            "sections": [],
        },
    ]
    qa_rows = [
        {
            "id": "p1",
            "qid": "q1",
            "type": "explicit",
            "question": "Who is it?",
            "answer": "Ada",
            "difficulty": 1,
        },
        {
            "id": "p1",
            "qid": "q2",
            "type": "implicit",
            "question": "Hint?",
            "answer": "math",
            "difficulty": 3,
        },
        {
            "id": "p2",
            "qid": "q3",
            "type": "explicit",
            "question": "Who?",
            "answer": "Grace",
            "difficulty": 2,
        },
    ]

    personas_path.write_text("\n".join(json.dumps(p) for p in personas) + "\n")
    qa_path.write_text("\n".join(json.dumps(q) for q in qa_rows) + "\n")
    return personas_path, qa_path


def test_persona_dataset_loads_personas_and_qa(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path)

    assert len(ds) == 2
    assert [p.id for p in ds] == ["p1", "p2"]
    assert ds[0].name == "Ada Lovelace"
    assert ds[0].sections[0].text == "Worked on math."
    assert ds.get_persona("p2").name == "Grace Hopper"
    assert ds.get_persona("missing") is None


def test_persona_dataset_qa_filters_by_type_and_difficulty(tmp_path: Path):
    personas_path, qa_path = _write_persona_fixture(tmp_path)
    ds = PersonaDataset(personas_path, qa_path)

    assert [q.qid for q in ds.get_qa("p1")] == ["q1", "q2"]
    assert [q.qid for q in ds.get_qa("p1", type="explicit")] == ["q1"]
    assert [q.qid for q in ds.get_qa("p1", difficulty=[1, 2])] == ["q1"]
    assert ds.questions("p1", type="implicit") == ["Hint?"]


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
    assert games.questions("g1", player="B") == ["Q1b"]


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
    assert persona_p4.name == "p4 "


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

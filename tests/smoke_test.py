"""Minimal publish smoke test.

This script is intentionally executable directly via `python tests/smoke_test.py`.
It avoids network access so package publishing does not depend on Hugging Face.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from persona_data.environment import get_device
from persona_data.nemotron_personas import (
    NemotronPersonasFranceDataset,
    NemotronPersonasUSADataset,
)
from persona_data.persona_guess import PersonaGuessDataset
from persona_data.prompts import format_roleplay_prompt
from persona_data.synth_persona import PersonaDataset


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def main() -> None:
    assert get_device().type in {"cpu", "cuda", "mps"}
    assert format_roleplay_prompt("hello")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        personas_path = tmp / "personas.jsonl"
        qa_path = tmp / "qa.jsonl"

        personas_path.write_text(
            json.dumps(
                {
                    "id": "p1",
                    "persona": {"first_name": "Ada", "last_name": "Lovelace"},
                    "templated_view": "Ada Lovelace",
                    "biography_view": "Bio",
                    "sections": [
                        {
                            "section_id": "s1",
                            "category": "career",
                            "title": "Career",
                            "paragraphs": [{"text": "Worked on math."}],
                        }
                    ],
                }
            )
            + "\n"
        )
        qa_path.write_text(
            json.dumps(
                {
                    "id": "p1",
                    "qid": "q1",
                    "type": "explicit",
                    "question": "Who is it?",
                    "answer": "Ada Lovelace",
                    "difficulty": 1,
                }
            )
            + "\n"
        )

        synth = PersonaDataset(personas_path, qa_path)
        assert len(synth) == 1
        assert synth[0].name == "Ada Lovelace"
        assert synth.get_qa("p1")[0].question == "Who is it?"
        assert synth[0].sections[0].text == "Worked on math."

        synth_sampled = PersonaDataset(personas_path, qa_path, sample_size=0)
        assert len(synth_sampled) == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shard_1 = tmp / "train-00000-of-00002.parquet"
        shard_2 = tmp / "train-00001-of-00002.parquet"

        _write_parquet(
            shard_1,
            [
                {
                    "uuid": "p1",
                    "persona": "Ada Lovelace est une mathématicienne britannique.",
                    "cultural_background": "British",
                    "skills_and_expertise": "Programming",
                    "hobbies_and_interests": "Reading",
                    "career_goals_and_ambitions": "Teach others",
                    "sex": "Femme",
                    "age": 36,
                    "marital_status": "Célibataire",
                    "household_type": "Personne seule",
                    "education_level": "Bac+5",
                    "occupation": "Professions intermédiaires",
                    "commune": "Paris",
                    "departement": "Paris",
                    "country": "France",
                },
                {
                    "uuid": "p2",
                    "persona": "Marie Curie est une chercheuse en chimie.",
                    "cultural_background": "French",
                    "skills_and_expertise": "Research",
                    "hobbies_and_interests": "Walking",
                    "career_goals_and_ambitions": "Lead a lab",
                    "sex": "Femme",
                    "age": 44,
                    "marital_status": "Marié(e)",
                    "household_type": "Couple sans enfant",
                    "education_level": "Bac+5",
                    "occupation": "Cadres",
                    "commune": "Lyon",
                    "departement": "Rhône",
                    "country": "France",
                },
            ],
        )
        _write_parquet(
            shard_2,
            [
                {
                    "uuid": "p3",
                    "persona": "Sophie Germain est une spécialiste des mathématiques.",
                    "cultural_background": "French",
                    "skills_and_expertise": "Analysis",
                    "hobbies_and_interests": "Music",
                    "career_goals_and_ambitions": "Publish research",
                    "sex": "Femme",
                    "age": 29,
                    "marital_status": "Célibataire",
                    "household_type": "Personne seule",
                    "education_level": "Bac+5",
                    "occupation": "Professions intermédiaires",
                    "commune": "Lille",
                    "departement": "Nord",
                    "country": "France",
                }
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
            nemotron = NemotronPersonasFranceDataset(sample_size=2)
            assert len(nemotron) == 2
            assert nemotron[0].name == "Ada Lovelace"
            assert nemotron[1].name == "Marie Curie"
            assert nemotron.get_persona("p2").templated_view.startswith(
                "Name: Marie Curie"
            )
            assert "Career goals and ambitions" in nemotron[0].templated_view
            assert nemotron.supports_qa is False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shard_1 = tmp / "train-00000-of-00002.parquet"
        shard_2 = tmp / "train-00001-of-00002.parquet"

        _write_parquet(
            shard_1,
            [
                {
                    "uuid": "u1",
                    "persona": "Mary Alberti is a front-line food service specialist.",
                    "cultural_background": "American",
                    "skills_and_expertise": "POS operation",
                    "skills_and_expertise_list": ["POS operation"],
                    "hobbies_and_interests": "Running",
                    "hobbies_and_interests_list": ["Running"],
                    "career_goals_and_ambitions": "Promotion",
                    "sex": "Female",
                    "age": 28,
                    "marital_status": "never_married",
                    "education_level": "bachelors",
                    "bachelors_field": "stem",
                    "occupation": "food_service",
                    "city": "Madison",
                    "state": "WI",
                    "zipcode": "53717",
                    "country": "USA",
                },
                {
                    "uuid": "u2",
                    "persona": "Alicia Gonzalez is a machine learning researcher.",
                    "cultural_background": "American",
                    "skills_and_expertise": "Machine learning",
                    "skills_and_expertise_list": ["Machine learning"],
                    "hobbies_and_interests": "Painting",
                    "hobbies_and_interests_list": ["Painting"],
                    "career_goals_and_ambitions": "Lead a team",
                    "sex": "Female",
                    "age": 29,
                    "marital_status": "divorced",
                    "education_level": "graduate",
                    "bachelors_field": "stem",
                    "occupation": "research_scientist",
                    "city": "Chico",
                    "state": "CA",
                    "zipcode": "95928",
                    "country": "USA",
                },
            ],
        )
        _write_parquet(
            shard_2,
            [
                {
                    "uuid": "u3",
                    "persona": "Deeva Cintron is a community finance volunteer.",
                    "cultural_background": "American",
                    "skills_and_expertise": "Budgeting",
                    "skills_and_expertise_list": ["Budgeting"],
                    "hobbies_and_interests": "Quilting",
                    "hobbies_and_interests_list": ["Quilting"],
                    "career_goals_and_ambitions": "Stay independent",
                    "sex": "Female",
                    "age": 85,
                    "marital_status": "married_present",
                    "education_level": "some_college",
                    "bachelors_field": "",
                    "occupation": "not_in_workforce",
                    "city": "Saline",
                    "state": "MI",
                    "zipcode": "48176",
                    "country": "USA",
                }
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
            nemotron_usa = NemotronPersonasUSADataset(sample_size=2)
            assert len(nemotron_usa) == 2
            assert nemotron_usa[0].name == "Mary Alberti"
            assert nemotron_usa[1].name == "Alicia Gonzalez"
            assert nemotron_usa.get_persona("u2").templated_view.startswith(
                "Name: Alicia Gonzalez"
            )
            assert "Location: Chico, CA, 95928, USA" in nemotron_usa[1].templated_view
            assert nemotron_usa.supports_qa is False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        games_path = tmp / "games.jsonl"
        games_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "game_id": "g1",
                            "persona_a_id": "pa",
                            "persona_b_id": "pb",
                            "turns": [
                                {
                                    "round": 1,
                                    "asker": "A",
                                    "question": "Q1",
                                    "answer": "A1",
                                }
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "game_id": "g2",
                            "persona_a_id": "pc",
                            "persona_b_id": "pd",
                            "turns": [
                                {
                                    "round": 1,
                                    "asker": "B",
                                    "question": "Q2",
                                    "answer": "A2",
                                }
                            ],
                        }
                    ),
                ]
            )
            + "\n"
        )

        with patch(
            "persona_data.persona_guess.hf_hub_download", return_value=games_path
        ):
            games = PersonaGuessDataset(sample_size=1)
            assert len(games) == 1
            assert games[0].game_id == "g1"
            assert games.get_qa("g1")[0].question == "Q1"


if __name__ == "__main__":
    main()

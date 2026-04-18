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
from persona_data.nemotron_personas import NemotronPersonasFranceDataset
from persona_data.persona_guess import PersonaGuessDataset
from persona_data.prompts import format_roleplay_prompt
from persona_data.synth_persona import PersonaDataset


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def main() -> None:
    assert get_device().type in {"cpu", "cuda", "mps"}
    assert format_roleplay_prompt("hello")
    assert PersonaGuessDataset is not None

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
            nemotron = NemotronPersonasFranceDataset(sample_size=2, offset=1)
            assert len(nemotron) == 2
            assert nemotron[0].name == "Marie Curie"
            assert nemotron[1].name == "Sophie Germain"
            assert nemotron.get_persona("p2").templated_view.startswith(
                "Name: Marie Curie"
            )
            assert "Career goals and ambitions" in nemotron[0].templated_view
            assert nemotron.supports_qa is False


if __name__ == "__main__":
    main()

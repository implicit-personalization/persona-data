"""Minimal publish smoke test.

This script is intentionally executable directly via `python tests/smoke_test.py`.
It avoids network access so package publishing does not depend on Hugging Face.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from persona_data.environment import get_device
from persona_data.persona_guess import PersonaGuessDataset
from persona_data.prompts import format_roleplay_prompt
from persona_data.synth_persona import PersonaDataset


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


if __name__ == "__main__":
    main()

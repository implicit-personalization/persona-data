from __future__ import annotations

import re
from typing import Any, Iterator

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from persona_data.synth_persona import PersonaData

_NAME_RE = re.compile(r"^\s*([A-ZÀ-ÖØ-Þ][\w'’.-]+(?:\s+[A-ZÀ-ÖØ-Þ][\w'’.-]+){1,3})\b")


def _list_parquet_files(hf_repo: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            file_name
            for file_name in list_repo_files(hf_repo, repo_type="dataset")
            if file_name.startswith("data/train-") and file_name.endswith(".parquet")
        )
    )


def _fetch_rows(*, hf_repo: str, offset: int, length: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    remaining = length
    current_offset = offset

    for file_name in _list_parquet_files(hf_repo):
        parquet_file = pq.ParquetFile(
            hf_hub_download(hf_repo, file_name, repo_type="dataset")
        )
        file_rows = parquet_file.metadata.num_rows

        if current_offset >= file_rows:
            current_offset -= file_rows
            continue

        batch_size = min(remaining, file_rows - current_offset)
        batch_rows = parquet_file.read().slice(current_offset, batch_size).to_pylist()
        rows.extend(dict(row) for row in batch_rows)
        remaining -= len(batch_rows)
        current_offset = 0
        if remaining <= 0:
            break

    return rows


def _extract_display_name(persona_text: str, fallback_id: str) -> str:
    match = _NAME_RE.match(persona_text or "")
    if match:
        return match.group(1).strip()
    words = [w for w in (persona_text or "").strip().split() if w]
    if len(words) >= 2:
        return " ".join(words[:2])
    if words:
        return words[0]
    return fallback_id


def _split_name(display_name: str) -> tuple[str, str]:
    parts = [part for part in display_name.split() if part]
    if not parts:
        return "Unknown", "Persona"
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _templated_view(row: dict[str, Any], display_name: str) -> str:
    lines = [
        f"Name: {display_name}",
        f"Age: {row.get('age', 'Unknown')}",
        f"Sex: {row.get('sex', 'Unknown')}",
        (
            "Location: "
            f"{row.get('commune', 'Unknown')}, "
            f"{row.get('departement', 'Unknown')}, "
            f"{row.get('country', 'Unknown')}"
        ),
        f"Occupation: {row.get('occupation', 'Unknown')}",
        f"Education: {row.get('education_level', 'Unknown')}",
        f"Marital status: {row.get('marital_status', 'Unknown')}",
        f"Household type: {row.get('household_type', 'Unknown')}",
        "",
        f"Cultural background: {row.get('cultural_background', '')}".strip(),
        f"Skills and expertise: {row.get('skills_and_expertise', '')}".strip(),
        f"Hobbies and interests: {row.get('hobbies_and_interests', '')}".strip(),
        (
            f"Career goals and ambitions: {row.get('career_goals_and_ambitions', '')}"
        ).strip(),
    ]
    return "\n".join(line for line in lines if line.strip())


def _row_to_persona(row: dict[str, Any]) -> PersonaData:
    persona_text = str(row.get("persona", "")).strip()
    persona_id = str(row.get("uuid", "")).strip()
    display_name = _extract_display_name(persona_text, persona_id)
    first_name, last_name = _split_name(display_name)
    persona_dict = dict(row)
    persona_dict["first_name"] = first_name
    persona_dict["last_name"] = last_name
    return PersonaData(
        id=persona_id,
        persona=persona_dict,
        templated_view=_templated_view(row, display_name),
        biography_view=persona_text,
    )


class NemotronPersonasFranceDataset:
    """French persona-only dataset backed by `nvidia/Nemotron-Personas-France`."""

    DEFAULT_REPO = "nvidia/Nemotron-Personas-France"
    supports_qa = False

    def __init__(
        self,
        hf_repo: str = DEFAULT_REPO,
        *,
        sample_size: int = 200,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.hf_repo = hf_repo
        self.sample_size = sample_size
        if rows is None:
            rows = _fetch_rows(hf_repo=hf_repo, offset=0, length=sample_size)
        self._personas = [_row_to_persona(row) for row in rows]
        self._personas_by_id = {persona.id: persona for persona in self._personas}

    def __len__(self) -> int:
        return len(self._personas)

    def __iter__(self) -> Iterator[PersonaData]:
        return iter(self._personas)

    def __getitem__(self, idx: int) -> PersonaData:
        return self._personas[idx]

    def get_persona(self, persona_id: str) -> PersonaData | None:
        return self._personas_by_id.get(persona_id)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(hf_repo={self.hf_repo!r}, "
            f"sample_size={len(self._personas)})"
        )

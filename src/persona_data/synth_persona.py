import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

# NOTE: Bank/provisional question-family metadata is intentionally omitted for now.
# The loader keeps only persona records and per-persona QA pairs to stay simple.


@dataclass
class QAPair:
    qid: str
    type: Literal["explicit", "implicit"]
    question: str
    answer: str
    difficulty: int  # 1 = easy, 2 = medium, 3 = hard
    answer_format: str = ""  # "free_text" or "choice"
    choices: list[str] = field(default_factory=list)
    correct_choice_index: int | None = None
    tags: list[str] = field(default_factory=list)
    validation: dict | None = None

    def __repr__(self):
        return f"QAPair(qid={self.qid!r}, type={self.type!r}, difficulty={self.difficulty})"


@dataclass
class Statement:
    sid: str
    category: str
    claim: str
    support: list[dict] = field(default_factory=list)
    confidence: str = ""

    def __repr__(self):
        return f"Statement(sid={self.sid!r}, category={self.category!r})"


@dataclass
class BiographySection:
    """A semantically coherent segment of a persona biography."""

    section_id: str
    category: str  # e.g. "upbringing", "education", "career"
    title: str
    text: str  # Concatenated paragraph texts

    def __repr__(self):
        return f"BiographySection(id={self.section_id!r}, category={self.category!r})"


@dataclass
class PersonaData:
    id: str
    persona: dict
    templated_view: str
    biography_view: str
    statements_view: str = ""
    # transcript: str = "" # NOTE: This is not needed for our usecase for now
    sections: list[BiographySection] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.persona['first_name']} {self.persona['last_name']}"

    @property
    def sections_by_id(self) -> dict[str, BiographySection]:
        return {section.section_id: section for section in self.sections}

    @property
    def sections_by_category(self) -> dict[str, list[BiographySection]]:
        grouped: dict[str, list[BiographySection]] = defaultdict(list)
        for section in self.sections:
            grouped[section.category].append(section)
        return dict(grouped)

    @property
    def section_categories(self) -> list[str]:
        """Unique section categories in order."""
        return list(self.sections_by_category)

    def get_section(self, section_id: str) -> BiographySection | None:
        return self.sections_by_id.get(section_id)

    def get_sections_by_category(self, category: str) -> list[BiographySection]:
        return self.sections_by_category.get(category, [])

    def __repr__(self):
        return f"PersonaData(id={self.id!r}, name={self.name!r})"


class PersonaDataset:
    """Persona dataset loaded from local JSONL files."""

    def __init__(self, personas_path: Path | str, qa_path: Path | str) -> None:
        self._personas: list[PersonaData] = []
        self._personas_by_id: dict[str, PersonaData] = {}
        with open(personas_path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                sections = [
                    BiographySection(
                        section_id=sec["section_id"],
                        category=sec["category"],
                        title=sec["title"],
                        text="\n\n".join(p["text"] for p in sec.get("paragraphs", [])),
                    )
                    for sec in d.get("sections", [])
                ]
                persona = PersonaData(
                    id=d["id"],
                    persona=d["persona"],
                    templated_view=d["templated_view"],
                    biography_view=d.get("biography_view", ""),
                    statements_view=d.get("statements_view", ""),
                    sections=sections,
                    statements=[
                        Statement(
                            sid=s["sid"],
                            category=s["category"],
                            claim=s["claim"],
                            support=s.get("support", []),
                            confidence=s.get("confidence", ""),
                        )
                        for s in d.get("statements", [])
                    ],
                )
                self._personas.append(persona)
                self._personas_by_id[persona.id] = persona

        self._qa: dict[str, list[QAPair]] = defaultdict(list)
        with open(qa_path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                self._qa[d["id"]].append(
                    QAPair(
                        qid=d["qid"],
                        type=d["type"],
                        question=d["question"],
                        answer=d["answer"],
                        difficulty=d["difficulty"],
                        answer_format=d.get("answer_format", "free_text"),
                        choices=d.get("choices", []),
                        correct_choice_index=d.get("correct_choice_index"),
                        tags=d.get("tags", []),
                        validation=d.get("validation"),
                    )
                )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_personas={len(self._personas)})"

    def __len__(self) -> int:
        return len(self._personas)

    def __iter__(self) -> Iterator[PersonaData]:
        return iter(self._personas)

    def __getitem__(self, idx: int) -> PersonaData:
        return self._personas[idx]

    def get_persona(self, persona_id: str) -> PersonaData | None:
        return self._personas_by_id.get(persona_id)

    def get_qa(
        self,
        persona_id: str,
        type: Literal["explicit", "implicit"] | None = None,
        difficulty: int | list[int] | None = None,
    ) -> list[QAPair]:
        """Return QA pairs for a persona, optionally filtered by type and/or difficulty."""
        pairs = self._qa.get(persona_id, [])
        if type is not None:
            pairs = [p for p in pairs if p.type == type]
        if difficulty is not None:
            levels = {difficulty} if isinstance(difficulty, int) else set(difficulty)
            pairs = [p for p in pairs if p.difficulty in levels]
        return pairs

    def questions(
        self,
        persona_id: str,
        type: Literal["explicit", "implicit"] | None = None,
        difficulty: int | list[int] | None = None,
    ) -> list[str]:
        """Like get_qa but returns question strings only."""
        return [qa.question for qa in self.get_qa(persona_id, type, difficulty)]


class SynthPersonaDataset(PersonaDataset):
    """SynthPersona dataset loaded from HuggingFace."""

    def __init__(self, hf_repo: str = "implicit-personalization/synth-persona") -> None:
        from huggingface_hub import hf_hub_download

        # HF Hub caches locally under HF_HOME so repeat runs are instant.
        super().__init__(
            personas_path=hf_hub_download(
                hf_repo, "dataset_personas.jsonl", repo_type="dataset"
            ),
            qa_path=hf_hub_download(hf_repo, "dataset_qa.jsonl", repo_type="dataset"),
        )

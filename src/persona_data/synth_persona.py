import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal


@dataclass
class QAPair:
    qid: str
    type: Literal["explicit", "implicit"]
    item_type: Literal["mcq", "frq"]
    scope: Literal["individual", "shared"]
    question: str
    answer: str
    choices: list[str] = field(default_factory=list)
    correct_choice_index: int | None = None
    evidence_sids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __repr__(self):
        return f"QAPair(qid={self.qid!r}, type={self.type!r}, item_type={self.item_type!r})"


@dataclass
class Statement:
    sid: str
    category: str
    claim: str
    support_turns: list[int] = field(default_factory=list)

    def __repr__(self):
        return f"Statement(sid={self.sid!r}, category={self.category!r})"


@dataclass
class PersonaData:
    id: str
    persona: dict
    templated_view: str
    biography_view: str
    statements_view: str = ""
    statements: list[Statement] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.persona['first_name']} {self.persona['last_name']}"

    def __repr__(self):
        return f"PersonaData(id={self.id!r}, name={self.name!r})"


class PersonaDataset:
    """Persona dataset loaded from local JSONL files."""

    def __init__(
        self,
        personas_path: Path | str,
        qa_path: Path | str,
        *,
        sample_size: int | None = None,
    ) -> None:
        self.sample_size = sample_size
        self._personas: list[PersonaData] = []
        self._personas_by_id: dict[str, PersonaData] = {}
        with open(personas_path) as f:
            for line in f:
                if not line.strip():
                    continue
                if sample_size is not None and len(self._personas) >= sample_size:
                    break
                d = json.loads(line)
                persona = PersonaData(
                    id=d["id"],
                    persona=d["persona"],
                    templated_view=d["templated_view"],
                    biography_view=d.get("biography_view", ""),
                    statements_view=d.get("statements_view", ""),
                    statements=[
                        Statement(
                            sid=s["sid"],
                            category=s["category"],
                            claim=s["claim"],
                            support_turns=s.get("support_turns", []),
                        )
                        for s in d.get("statements", [])
                    ],
                )
                self._personas.append(persona)
                self._personas_by_id[persona.id] = persona

        loaded_ids = set(self._personas_by_id)
        self._qa: dict[str, list[QAPair]] = defaultdict(list)
        with open(qa_path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["id"] not in loaded_ids:
                    continue
                self._qa[d["id"]].append(
                    QAPair(
                        qid=d["qid"],
                        type=d["type"],
                        item_type=d["item_type"],
                        scope=d["scope"],
                        question=d["question"],
                        answer=d["answer"],
                        choices=d.get("choices") or [],
                        correct_choice_index=d.get("correct_choice_index"),
                        evidence_sids=d.get("evidence_sids", []),
                        tags=d.get("tags", []),
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
        item_type: Literal["mcq", "frq"] | None = None,
        scope: Literal["individual", "shared"] | None = None,
    ) -> list[QAPair]:
        """Return QA pairs for a persona, optionally filtered by type / item_type."""
        pairs = self._qa.get(persona_id, [])
        if type is not None:
            pairs = [p for p in pairs if p.type == type]
        if item_type is not None:
            pairs = [p for p in pairs if p.item_type == item_type]
        if scope is not None:
            pairs = [p for p in pairs if p.scope == scope]
        return pairs

    def train_test_split(
        self,
        persona_id: str,
        *,
        n_train: int = 25,
        train_type: Literal["explicit", "implicit"] | None = "explicit",
        train_item_type: Literal["mcq", "frq"] | None = "mcq",
    ) -> tuple[list[QAPair], list[QAPair]]:
        """Return ``(train, test)`` QA splits for one persona.

        Train: ``scope='individual'`` QAs — persona-specific items used to
        derive a persona representation (Doc-to-LoRA conditioning, steering
        vector, SAE features, …). Capped at ``n_train`` and by default
        narrowed to explicit MCQs.
        Test: ``scope='shared'`` QAs — the questions every persona answers,
        kept as a held-out evaluation set with whatever mix of
        explicit/implicit is present in the data.
        """
        train = self.get_qa(
            persona_id,
            type=train_type,
            item_type=train_item_type,
            scope="individual",
        )[:n_train]
        test = self.get_qa(persona_id, scope="shared")
        return train, test


class SynthPersonaDataset(PersonaDataset):
    """SynthPersona dataset loaded from HuggingFace."""

    def __init__(
        self,
        hf_repo: str = "implicit-personalization/synth-persona",
        *,
        sample_size: int | None = None,
    ) -> None:
        from huggingface_hub import hf_hub_download

        # HF Hub caches locally under HF_HOME so repeat runs are instant.
        super().__init__(
            personas_path=hf_hub_download(
                hf_repo, "dataset_personas.jsonl", repo_type="dataset"
            ),
            qa_path=hf_hub_download(hf_repo, "dataset_qa.jsonl", repo_type="dataset"),
            sample_size=sample_size,
        )

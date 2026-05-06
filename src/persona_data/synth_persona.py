import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

QAType = Literal["explicit", "implicit"]
QAItemType = Literal["mcq", "frq"]
QAScope = Literal["individual", "shared"]
QASource = Literal["seed_attribute", "interview", "statement"]


@dataclass
class QAPair:
    qid: str
    type: QAType
    item_type: QAItemType
    scope: QAScope
    question: str
    answer: str
    choices: list[str] = field(default_factory=list)
    choice_labels: list[str] = field(default_factory=list)
    correct_choice_index: int | None = None
    correct_choice_letter: str | None = None
    source: QASource | str | None = None
    bank_id: str | None = None
    family_id: str | None = None
    evidence_sids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    is_baseline: bool = False
    split_group_id: str | None = None
    related_qids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

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


def _load_related_qids_by_bank_id(bank_path: Path | str) -> dict[str, list[str]]:
    """Load generic related-qid mappings from a shared QA bank file."""
    with open(bank_path) as f:
        payload = json.load(f)
    related: dict[str, list[str]] = {}
    for item in payload.get("items", []):
        bank_id = item.get("bank_id")
        if bank_id:
            related[str(bank_id)] = item.get("related_qids") or item.get(
                "related_frq_qids"
            ) or []
    return related


class PersonaDataset:
    """Persona dataset loaded from local JSONL files."""

    def __init__(
        self,
        personas_path: Path | str,
        qa_path: Path | str,
        *,
        related_qids_by_bank_id: Mapping[str, list[str]] | None = None,
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
        related_qids_by_bank_id = related_qids_by_bank_id or {}

        with open(qa_path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["id"] not in loaded_ids:
                    continue
                related_qids = d.get("related_qids") or d.get("related_frq_qids") or []
                bank_id = d.get("bank_id")
                if bank_id in related_qids_by_bank_id:
                    related_qids = list(related_qids_by_bank_id[str(bank_id)])
                split_group_id = d.get("split_group_id")
                if split_group_id is None and d.get("type") == "explicit" and d.get("bank_id"):
                    split_group_id = f"explicit:{d['bank_id']}"
                metadata = {
                    key: value
                    for key, value in d.items()
                    if key
                    not in {
                        "id",
                        "qid",
                        "type",
                        "item_type",
                        "scope",
                        "question",
                        "answer",
                        "choices",
                        "choice_labels",
                        "correct_choice_index",
                        "correct_choice_letter",
                        "source",
                        "bank_id",
                        "family_id",
                        "evidence_sids",
                        "tags",
                        "is_baseline",
                        "split_group_id",
                        "related_qids",
                        "related_frq_qids",
                    }
                }
                self._qa[d["id"]].append(
                    QAPair(
                        qid=d["qid"],
                        type=d["type"],
                        item_type=d["item_type"],
                        scope=d["scope"],
                        question=d["question"],
                        answer=d["answer"],
                        choices=d.get("choices") or [],
                        choice_labels=d.get("choice_labels") or [],
                        correct_choice_index=d.get("correct_choice_index"),
                        correct_choice_letter=d.get("correct_choice_letter"),
                        source=d.get("source"),
                        bank_id=d.get("bank_id"),
                        family_id=d.get("family_id"),
                        evidence_sids=d.get("evidence_sids") or [],
                        tags=d.get("tags") or [],
                        is_baseline=d.get("is_baseline", False),
                        split_group_id=split_group_id,
                        related_qids=related_qids,
                        metadata=metadata,
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
        type: QAType | None = None,
        item_type: QAItemType | None = None,
        scope: QAScope | None = None,
        source: QASource | str | None = None,
        bank_id: str | None = None,
        family_id: str | None = None,
    ) -> list[QAPair]:
        """Return QA pairs for one persona, optionally filtered by schema fields."""
        pairs = self._qa.get(persona_id, [])
        if type is not None:
            pairs = [p for p in pairs if p.type == type]
        if item_type is not None:
            pairs = [p for p in pairs if p.item_type == item_type]
        if scope is not None:
            pairs = [p for p in pairs if p.scope == scope]
        if source is not None:
            pairs = [p for p in pairs if p.source == source]
        if bank_id is not None:
            pairs = [p for p in pairs if p.bank_id == bank_id]
        if family_id is not None:
            pairs = [p for p in pairs if p.family_id == family_id]
        return pairs

    def train_test_split(
        self,
        persona_id: str,
        *,
        n_train: int = 25,
        train_type: QAType | None = "explicit",
        train_item_type: QAItemType | None = "mcq",
    ) -> tuple[list[QAPair], list[QAPair]]:
        """Return ``(train, test)`` QA splits for one persona.

        Train: individual explicit multiple-choice QAs by default. This keeps
        the historical helper behavior stable. Use `get_qa(..., item_type="frq")`
        directly when building FRQ-trained steering vectors.
        Test: shared multiple-choice QAs by default.
        """
        train = self.get_qa(
            persona_id,
            type=train_type,
            item_type=train_item_type,
            scope="individual",
        )[:n_train]
        test = self.get_qa(persona_id, item_type="mcq", scope="shared")
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
        implicit_bank_path = hf_hub_download(
            hf_repo, "implicit_shared_mc_bank.json", repo_type="dataset"
        )
        super().__init__(
            personas_path=hf_hub_download(
                hf_repo, "dataset_personas.jsonl", repo_type="dataset"
            ),
            qa_path=hf_hub_download(hf_repo, "dataset_qa.jsonl", repo_type="dataset"),
            related_qids_by_bank_id=_load_related_qids_by_bank_id(implicit_bank_path),
            sample_size=sample_size,
        )

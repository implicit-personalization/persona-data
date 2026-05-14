import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

BASELINE_PERSONA_ID = "baseline_assistant"
BASELINE_PERSONA_NAME = "Assistant"
_DEFAULT_ATTRIBUTE_EXCLUDE = {"first_name", "last_name"}


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
    bank_id: str | None = None
    related_frq_qids: list[str] = field(default_factory=list)
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
        return f"{self.persona['first_name']} {self.persona['last_name']}".strip()

    def __repr__(self):
        return f"PersonaData(id={self.id!r}, name={self.name!r})"


def _parse_persona(d: dict) -> "PersonaData":
    return PersonaData(
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


def _parse_qa(d: dict, related_map: dict[str, list[str]]) -> "QAPair":
    bank_id = d.get("bank_id")
    return QAPair(
        qid=d["qid"],
        type=d["type"],
        item_type=d["item_type"],
        scope=d["scope"],
        question=d["question"],
        answer=d["answer"],
        choices=d.get("choices") or [],
        correct_choice_index=d.get("correct_choice_index"),
        bank_id=bank_id,
        related_frq_qids=d.get("related_frq_qids") or related_map.get(bank_id, []),
        evidence_sids=d.get("evidence_sids", []),
        tags=d.get("tags", []),
    )


def _read_jsonl(path: Path | str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_json(path: Path | str | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def _ordinal_value_map(field_info: dict) -> dict[Any, float]:
    ordered = field_info.get("ordered_values")
    if not isinstance(ordered, list):
        return {}
    return {value: float(idx) for idx, value in enumerate(ordered)}


def _encode_attribute(
    value: Any,
    field_info: dict,
    ordinal_map: dict[Any, float] | None = None,
) -> float | None:
    """Convert analysis-friendly attributes to numeric values when possible."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if ordinal_map:
        if value in ordinal_map:
            return ordinal_map[value]
        raise ValueError(f"Unknown ordinal value {value!r}")
    if field_info.get("kind") == "binary" and value in {"No", "Yes"}:
        return 1.0 if value == "Yes" else 0.0
    return None


class PersonaDataset:
    """Persona dataset loaded from local JSONL files. The persona-less
    Assistant baseline is an ordinary row with id ``BASELINE_PERSONA_ID``.
    """

    def __init__(
        self,
        personas_path: Path | str,
        qa_path: Path | str,
        *,
        related_frq_qids_by_bank_id: dict[str, list[str]] | None = None,
        attribute_schema_path: Path | str | None = None,
        attribute_schema_loader: Callable[[], dict] | None = None,
        sample_size: int | None = None,
    ) -> None:
        self._attribute_schema_path = attribute_schema_path
        self._attribute_schema_loader = attribute_schema_loader
        self._attribute_schema: dict | None = None
        self._attribute_fields: dict[str, dict] = {}
        self._attribute_ordinal_maps: dict[str, dict[Any, float]] = {}
        self._personas: list[PersonaData] = []
        self._personas_by_id: dict[str, PersonaData] = {}

        for d in _read_jsonl(personas_path):
            if sample_size is not None and len(self._personas) >= sample_size:
                break
            persona = _parse_persona(d)
            self._personas.append(persona)
            self._personas_by_id[persona.id] = persona
        self._persona_ids = tuple(persona.id for persona in self._personas)

        loaded_ids = set(self._personas_by_id)
        related_map = related_frq_qids_by_bank_id or {}
        self._qa: dict[str, list[QAPair]] = defaultdict(list)
        for d in _read_jsonl(qa_path):
            if d["id"] in loaded_ids:
                self._qa[d["id"]].append(_parse_qa(d, related_map))

        self._inferred_attribute_names = tuple(
            sorted(
                {
                    name
                    for persona in self._personas
                    for name in persona.persona
                    if name not in _DEFAULT_ATTRIBUTE_EXCLUDE
                }
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

    @property
    def persona_ids(self) -> tuple[str, ...]:
        """Loaded persona ids in dataset order."""
        return self._persona_ids

    @property
    def attribute_names(self) -> tuple[str, ...]:
        """Non-identifier attribute names."""
        self._load_attribute_schema()
        if self._attribute_fields:
            return self._schema_attribute_names()
        return self._inferred_attribute_names

    @property
    def attribute_schema(self) -> dict:
        """Persona attribute schema, loaded lazily when configured."""
        return self._load_attribute_schema()

    def _load_attribute_schema(self) -> dict:
        if self._attribute_schema is not None:
            return self._attribute_schema
        if self._attribute_schema_loader is not None:
            self._attribute_schema = self._attribute_schema_loader()
        elif self._attribute_schema_path is not None:
            self._attribute_schema = _read_json(self._attribute_schema_path)
        else:
            self._attribute_schema = {}
        self._attribute_fields = self._attribute_schema.get("persona_fields", {})
        self._attribute_ordinal_maps = {
            name: ordinal_map
            for name, info in self._attribute_fields.items()
            if (ordinal_map := _ordinal_value_map(info))
        }
        return self._attribute_schema

    def _schema_attribute_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, info in self._attribute_fields.items()
            if info.get("analysis_recommendation") != "drop"
            and not info.get("identifier")
        )

    def get_persona(self, persona_id: str) -> PersonaData | None:
        return self._personas_by_id.get(persona_id)

    @property
    def baseline(self) -> PersonaData | None:
        """Persona-less Assistant baseline if loaded, else ``None``."""
        return self._personas_by_id.get(BASELINE_PERSONA_ID)

    def attribute_info(self, name: str) -> dict:
        """Return schema metadata for an attribute, if available."""
        self._load_attribute_schema()
        return self._attribute_fields.get(name, {})

    def attribute_values(
        self,
        name: str,
        persona_ids: list[str] | None = None,
        *,
        encode: bool = False,
        missing: Any = None,
    ) -> list[Any]:
        """Return attribute values aligned to ``persona_ids`` or dataset order."""
        ids = persona_ids if persona_ids is not None else self._persona_ids
        field_info = self.attribute_info(name)
        ordinal_map = self._attribute_ordinal_maps.get(name)
        values: list[Any] = []
        for persona_id in ids:
            persona = self._personas_by_id[persona_id]
            value = persona.persona.get(name, missing)
            values.append(
                _encode_attribute(value, field_info, ordinal_map) if encode else value
            )
        return values

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
        n_train: int | None = None,
        seed: int | None = 0,
    ) -> tuple[list[QAPair], list[QAPair]]:
        """Return leakage-filtered ``(train, test)`` QA splits for one persona.

        Train: individual free-response questions (explicit and implicit).
        Test:  shared multiple-choice questions (explicit and implicit) —
               the same item bank for every persona, so per-persona test
               scores are directly comparable.

        Some shared MCQs are constructed from individual FRQs of the same
        persona — explicit MCQs reuse the FRQ's ``bank_id`` slot, and
        implicit shared MCQs list the source FRQ qids in
        ``related_frq_qids``. To avoid train-test leakage when training
        steering methods on FRQs and evaluating on MCQs, train rows that
        match either are dropped. The test set is preserved in full.

        ``n_train`` caps the train slice. Pass ``None`` to return every
        non-leaking FRQ. ``seed`` shuffles the train candidates before
        capping; pass ``seed=None`` to preserve dataset order. Test order is
        left untouched.
        """
        test = self.get_qa(persona_id, item_type="mcq", scope="shared")

        test_bank_ids = {q.bank_id for q in test if q.bank_id}
        test_qids = {qid for q in test for qid in q.related_frq_qids}

        train = [
            q
            for q in self.get_qa(persona_id, item_type="frq", scope="individual")
            if not (q.bank_id and q.bank_id in test_bank_ids) and q.qid not in test_qids
        ]
        if seed is not None:
            random.Random(seed).shuffle(train)
        if n_train is not None:
            train = train[:n_train]
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
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

        # HF Hub caches locally under HF_HOME so repeat runs are instant.
        bank_path = hf_hub_download(
            hf_repo, "implicit_shared_mc_bank.json", repo_type="dataset"
        )
        with open(bank_path) as f:
            bank = json.load(f)
        related_map = {
            item["bank_id"]: item.get("related_frq_qids", [])
            for item in bank.get("items", [])
            if item.get("bank_id")
        }
        def load_attribute_schema() -> dict:
            try:
                attribute_schema_path = hf_hub_download(
                    hf_repo, "attribute_schema.json", repo_type="dataset"
                )
            except (EntryNotFoundError, LocalEntryNotFoundError):
                return {}
            return _read_json(attribute_schema_path)

        super().__init__(
            personas_path=hf_hub_download(
                hf_repo, "dataset_personas.jsonl", repo_type="dataset"
            ),
            qa_path=hf_hub_download(hf_repo, "dataset_qa.jsonl", repo_type="dataset"),
            related_frq_qids_by_bank_id=related_map,
            attribute_schema_loader=load_attribute_schema,
            sample_size=sample_size,
        )

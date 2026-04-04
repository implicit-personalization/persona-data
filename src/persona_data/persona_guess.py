import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from huggingface_hub import hf_hub_download


@dataclass
class Turn:
    round: int
    asker: Literal["A", "B"]
    question: str
    answer: str

    def __repr__(self):
        return f"Turn(round={self.round}, asker={self.asker!r})"


@dataclass
class GameRecord:
    game_id: str
    persona_a_id: str
    persona_b_id: str
    turns: list[Turn]

    def __repr__(self):
        return f"GameRecord(game_id={self.game_id!r}, turns={len(self.turns)})"


class PersonaGuessDataset:
    """PersonaGuess game dataset loaded from HuggingFace."""

    DEFAULT_REPO = "implicit-personalization/persona-guess"

    def __init__(self, hf_repo: str = DEFAULT_REPO) -> None:
        path = Path(hf_hub_download(hf_repo, "games.jsonl", repo_type="dataset"))
        with open(path) as f:
            self._games: list[GameRecord] = [
                GameRecord(
                    game_id=d["game_id"],
                    persona_a_id=d["persona_a_id"],
                    persona_b_id=d["persona_b_id"],
                    turns=[Turn(**t) for t in d["turns"]],
                )
                for d in (json.loads(line) for line in f)
            ]
        self._games_by_id: dict[str, GameRecord] = {g.game_id: g for g in self._games}

    def __repr__(self) -> str:
        return f"PersonaGuessDataset(n_games={len(self._games)})"

    def __len__(self) -> int:
        return len(self._games)

    def __iter__(self) -> Iterator[GameRecord]:
        return iter(self._games)

    def __getitem__(self, idx: int) -> GameRecord:
        return self._games[idx]

    def get_qa(
        self, game_id: str, player: Literal["A", "B"] | None = None
    ) -> list[Turn]:
        game = self._games_by_id[game_id]
        return [t for t in game.turns if player is None or t.asker == player]

    def questions(
        self, game_id: str, player: Literal["A", "B"] | None = None
    ) -> list[str]:
        return [t.question for t in self.get_qa(game_id, player)]

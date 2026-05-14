# PersonaGuess

`PersonaGuessDataset` loads turn-based games where two personas ask each other questions from `implicit-personalization/persona-guess` (file: `games.jsonl`).

```python
from persona_data.persona_guess import PersonaGuessDataset

games = PersonaGuessDataset()
small = PersonaGuessDataset(sample_size=10)
```

`sample_size` keeps the leading games rather than sampling randomly.

## Records

- `GameRecord(game_id, persona_a_id, persona_b_id, turns)`
- `Turn(round, asker, question, answer)` — `asker` is `"A"` or `"B"`.

## Queries

```python
game = games[0]

turns_all = games.get_qa(game.game_id)             # all turns
turns_a   = games.get_qa(game.game_id, player="A") # only A's turns

for turn in turns_a:
    print(turn.round, turn.question, turn.answer)
```

Use `persona_a_id` and `persona_b_id` to join a game back to persona metadata
from another dataset. `get_qa()` raises `KeyError` for a game id that was not
loaded, which can happen when `sample_size` excludes later rows.

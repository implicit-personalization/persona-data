# PersonaGuess

`PersonaGuessDataset` loads turn-based games where two personas ask each other questions.

## Loader

```python
from persona_data.persona_guess import PersonaGuessDataset

games = PersonaGuessDataset()
```

The default dataset source is `implicit-personalization/persona-guess`. The loader reads `games.jsonl`.

## Records

- `GameRecord`: one match between two personas
- `Turn`: one question-answer turn within a game

## Game fields

`GameRecord` includes:

- `game_id`
- `persona_a_id`
- `persona_b_id`
- `turns`

`Turn` includes:

- `round`
- `asker`
- `question`
- `answer`

## Queries

```python
game = games[0]

turns = games.get_qa(game.game_id)
turns = games.get_qa(game.game_id, player="A")

questions = games.questions(game.game_id, player="B")
```

`get_qa()` returns typed `Turn` records. `questions()` returns question strings only.

## Notes

- `player` can be `"A"`, `"B"`, or omitted for all turns.
- The dataset is small enough to load eagerly into memory.

# Dataset Loading

Each dataset has its own module in `persona_data/` (e.g. `synth_persona.py`).
Construction downloads files from HuggingFace (cached locally via `HF_HOME`) and
loads everything eagerly in `__init__`.

`SynthPersonaDataset` is the reference implementation. New datasets should follow
a similar structure.

---

## SynthPersona

```python
from persona_data.synth_persona import SynthPersonaDataset

dataset = SynthPersonaDataset()                          # default repo
dataset = SynthPersonaDataset("my-org/my-repo")          # custom repo
```

Iterate and index over personas:

```python
len(dataset)          # number of personas
dataset[0]            # PersonaData
for persona in dataset: ...
```

Query QA pairs — all filters are optional and combinable:

```python
persona = dataset[0]

# This returns the object
dataset.get_qa(persona.id)                                      # all pairs
dataset.get_qa(persona.id, type="explicit", difficulty=[2, 3])  # combined

# This returns the strings directly which is a useful utility
dataset.questions(persona.id, type="implicit")
```

---

## PersonaGuess

```python
from persona_data.persona_guess import PersonaGuessDataset

games = PersonaGuessDataset()

game = games[0]
questions = games.questions(game.game_id, player="A")
```

---

## Adding a new dataset

Create `persona_data/<name>.py`. The pattern is:

1. Define `@dataclass` records for your data.
2. Write a plain class with `__init__` that downloads files via
   `hf_hub_download`, loads lightweight records.
3. Expose `__len__`, `__iter__`, `__getitem__`, and other relevant query methods.

### Template

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download


@dataclass
class MyRecord:
    id: str
    # ... your fields


class MyDataset:
    DEFAULT_REPO = "implicit-personalization/my-dataset"

    def __init__(self, hf_repo: str = DEFAULT_REPO) -> None:
        path = Path(hf_hub_download(hf_repo, "data.jsonl", repo_type="dataset"))
        with open(path) as f:
            self._records: list[MyRecord] = [
                MyRecord(**json.loads(line))
                for line in f
            ]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[MyRecord]:
        return iter(self._records)

    def __getitem__(self, idx: int) -> MyRecord:
        return self._records[idx]
```

Usage:

```python
from persona_data.my_dataset import MyDataset

data = MyDataset()
record = data[0]
```

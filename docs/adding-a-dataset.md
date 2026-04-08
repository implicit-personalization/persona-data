# Adding a New Dataset

New dataset modules should match the style of the existing loaders: eager, typed, and notebook-friendly.

## Recommended shape

1. Define dataclasses for the records you need.
2. Download source files from Hugging Face with `hf_hub_download`.
3. Load the records in `__init__` and keep them in memory.
4. Implement `__len__`, `__iter__`, and `__getitem__`.
5. Add query helpers for the dataset shape, such as `get_qa()` and `questions()`.

## Minimal example

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download


@dataclass
class MyRecord:
    id: str


class MyDataset:
    DEFAULT_REPO = "implicit-personalization/my-dataset"

    def __init__(self, hf_repo: str = DEFAULT_REPO) -> None:
        path = Path(hf_hub_download(hf_repo, "data.jsonl", repo_type="dataset"))
        with open(path) as f:
            self._records: list[MyRecord] = [
                MyRecord(**json.loads(line))
                for line in f
                if line.strip()
            ]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[MyRecord]:
        return iter(self._records)

    def __getitem__(self, idx: int) -> MyRecord:
        return self._records[idx]
```

## Documentation checklist

- Add the loader to `docs/datasets.md`.
- Add a dataset-specific page if the source is reused by other projects.
- Mention the default Hugging Face repo and any important file names.

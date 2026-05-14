# Environment helpers

`persona_data.environment` keeps small runtime utilities shared across the
persona projects.

```python
from persona_data.environment import get_artifacts_dir, get_device, set_seed

set_seed(1337)
device = get_device()
artifacts_dir = get_artifacts_dir()
```

## Helpers

| Helper | Behavior |
|---|---|
| `set_seed(seed)` | Seeds Python `random`, NumPy, Torch CPU, CUDA when available, and MPS when available. |
| `get_device()` | Returns `cuda`, then `mps`, then `cpu` based on local Torch availability. |
| `get_artifacts_dir()` | Returns `Path(os.environ["ARTIFACTS_DIR"])`, or `Path("artifacts")` when the variable is unset. |

# EXP-136: 2×T4,12 часов

PREPARED, не новый результат. [Протокол](EXP-135-141-protocol.md).

```python
import importlib
import os
import subprocess
import sys

repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()
for module in ("extent.competitor_tpu", "extent.reset_followup", "scripts.m3q_reset_followup_campaign"):
    if module in sys.modules:
        importlib.reload(sys.modules[module])
from scripts.m3q_reset_followup_campaign import main as run_reset_experiment

result = run_reset_experiment([
    "--experiment", "136",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-steps", "131072",
    "--max-new-hf-gb", "4",
])
```

Итоговый файл: /kaggle/working/extent-results/extent-m3q-reset-136.json.
Та же ячейка продолжает сохранённое состояние этого нового эксперимента.

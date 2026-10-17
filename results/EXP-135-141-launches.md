# Ячейки EXP-135–141

Существующий GPU notebook для135–138; TPU notebook для139–141. Все лимиты включают резерв.

# EXP-135: 2×T4,12 часов

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
    "--experiment", "135",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "4",
])
```

Итоговый файл: /kaggle/working/extent-results/extent-m3q-reset-135.json.
Та же ячейка продолжает сохранённое состояние этого нового эксперимента.

Один процесс использует ОБЕ GPU для одной траектории130 с3434 до32768. До принятия обновления проверяется источник; после8 новых шагов появляется оценка скорости. Полное прохождение32768 за одну сессию не обещано.


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
    "--max-new-hf-gb", "4",
])
```

Итоговый файл: /kaggle/working/extent-results/extent-m3q-reset-136.json.
Та же ячейка продолжает сохранённое состояние этого нового эксперимента.


# EXP-137: 2×T4,12 часов

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
    "--experiment", "137",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "4",
])
```

Итоговый файл: /kaggle/working/extent-results/extent-m3q-reset-137.json.
Та же ячейка продолжает сохранённое состояние этого нового эксперимента.


# EXP-138: 2×T4,12 часов

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
    "--experiment", "138",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "4",
])
```

Итоговый файл: /kaggle/working/extent-results/extent-m3q-reset-138.json.
Та же ячейка продолжает сохранённое состояние этого нового эксперимента.


# EXP-139: TPUv5e8,9 часов

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
    "--experiment", "139",
    "--max-wall-hours", "8.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "24",
])
```

Итоговый файл: /kaggle/working/extent-results/extent-m3q-reset-139.json.
Та же ячейка продолжает сохранённое состояние этого нового эксперимента.


# EXP-140: TPUv5e8,9 часов

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
    "--experiment", "140",
    "--max-wall-hours", "8.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "4",
])
```

Итоговый файл: /kaggle/working/extent-results/extent-m3q-reset-140.json.
Та же ячейка продолжает сохранённое состояние этого нового эксперимента.


# EXP-141: TPUv5e8,9 часов

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
    "--experiment", "141",
    "--max-wall-hours", "8.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "40",
])
```

Итоговый файл: /kaggle/working/extent-results/extent-m3q-reset-141.json.
Та же ячейка продолжает сохранённое состояние этого нового эксперимента.

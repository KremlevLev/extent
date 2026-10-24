# EXP159 — последняя ячейка существующего T4×2 notebook

Предыдущая GPU-настройка и HF/Telegram secrets сохраняются.
Заменить последнюю ячейку целиком. TPU157 не менять.
Пакеты прежние requirements-t4.txt, установленные ДО первого JAX import.

```python
import importlib
import os
import subprocess
import sys

repo = "/kaggle/working/extent"
subprocess.run(
    ["git", "pull", "--ff-only"], cwd=repo,
    env=globals().get("git_env", os.environ.copy()), check=True,
)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()

from scripts.m3q_hidden_distillation_159 import main
result = main([
    "--max-wall-hours", "11.8",
    "--save-reserve-minutes", "18",
    "--max-new-hf-gb", "8",
])
```

GPU0: CE+выходное распределение Qwen. GPU1: то же плюс нормализованные
внутренние представления после блоков7/14/21/28. Обе ветки от CURRENT118304,
rank256, свежий Adam, одинаковые learning rate/окна.

11.8ч включает18мин резерва; внутри12ч остаётся12мин на первоначальный
notebook setup. При завершении всех262144новых шагов может закончить раньше,
но этот потолок не короткий лимит.8192не останавливает запуск.

Уведомления: старт и итог. По каждому worker отдельные checkpoints/results
на HF; прислать только **extent-m3q-hidden-distill-159.json**.

Результаты физической T4 ещё не получены; CPU-проверки математики и runner
не гарантируют конкретного числа шагов за сессию.

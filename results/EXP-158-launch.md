# EXP-158 — 2×T4: качество шага80260 против118304

В существующем соответствующем notebook заменить последнюю ячейку целиком.
Setup и HF/Telegram secrets прежние. Запускать после свежего старта kernel;
версии JAX/jaxlib0.11.0, Flax0.12.8, Optax0.2.8, как в исходном EXP155.
Для TPU использовать requirements-continuation-tpu.txt ДО первого импорта JAX.
Для T4 прежний requirements-t4.txt и GPU T4×2.

```python
WALL_HOURS = 11.5
RESERVE_MINUTES = 18

import importlib
import os
import subprocess
import sys
import time

repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo,
               env=globals().get("git_env", os.environ.copy()), check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()
now = time.time()
os.environ["EXTENT_HF_DOWNLOAD_DEADLINE_UNIX"] = str(now + WALL_HOURS * 3600 - RESERVE_MINUTES * 60)
os.environ["EXTENT_HF_SAVE_DEADLINE_UNIX"] = str(now + WALL_HOURS * 3600 - 60)
os.environ["EXTENT_HF_RETRY_STATE_DIR"] = "/kaggle/working/extent-hf-download"
from extent.hf_download_retry import install_download_retries
install_download_retries()
from scripts.m3q_quality_followup_158 import main
result = main([
    "--max-wall-hours", str(WALL_HOURS),
    "--save-reserve-minutes", str(RESERVE_MINUTES),
])
```

Это ограниченная диагностика, а не повтор полного бенчмарка на30737вопросов. Каждая T4 оценивает один замороженный чекпойнт: старый80260 и новый118304. По128окон свежих английских/русских текстов, контексты128/256/512/1024, по256вопросов каждого из6тестов,24greedyгенерации. После полного набора заканчивает раньше лимита; точная длительность не обещается. Обучения нет. Оба чекпойнта проходят собственную сверку прежнего NLL. Продолжение собственной частичной диагностики автоматически через отчёт HF.

Прислать **extent-m3q-quality-158.json**. В paired/paired_nll `old` означает80260, `new`118304. Меньше NLL лучше; больше accuracy лучше.

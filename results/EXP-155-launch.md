# EXP-155: продолжение лучшей R256 на TPU, 9 часов

Существующий TPU notebook. Первая ячейка клонирования и secrets прежние.
Установка до импорта JAX (в свежем kernel):

```python
%cd /kaggle/working/extent
!pip install -q -r requirements-continuation-tpu.txt
```

Последняя ячейка:

```python
import importlib
import os
import subprocess
import sys
import time

WALL_HOURS = 8.7  # Включает загрузку/JIT/обучение и резерв сохранения.
RESERVE_MINUTES = 45
repo = "/kaggle/working/extent"
subprocess.run(
    ["git", "pull", "--ff-only"], cwd=repo,
    env=globals().get("git_env", os.environ.copy()), check=True,
)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()
now = time.time()
os.environ["EXTENT_HF_DOWNLOAD_DEADLINE_UNIX"] = str(
    now + WALL_HOURS * 3600 - RESERVE_MINUTES * 60)
os.environ["EXTENT_HF_SAVE_DEADLINE_UNIX"] = str(
    now + WALL_HOURS * 3600 - 60)
os.environ["EXTENT_HF_RETRY_STATE_DIR"] = "/kaggle/working/extent-hf-download"
from extent.hf_download_retry import install_download_retries
install_download_retries()
from scripts.m3q_rank_continue_155_campaign import main
result = main([
    "--experiment", "155",
    "--max-wall-hours", str(WALL_HOURS),
    "--save-reserve-minutes", str(RESERVE_MINUTES),
    "--max-steps", "262144",
    "--max-new-hf-gb", "8",
])
```

Kaggle Accelerator: TPU v5e-8. Internet/secrets должны быть включены.
Runtime должен быть свежим: установка pinned requirements после уже
импортированного JAX не меняет загруженную версию.

Старт: CURRENT37910 / Adam6784, унаследованный BEST37270.
Ожидаем source_replay.passed=true, adam_count_at_restore=[6784]×6;
execution_checks покажет compiled либо streamed fallback на той же TPU.
Дальше растёт globalstep от37910, счетчик NEW=global−37910.
Окончание по времени может дать deadline_partial — смотрим сохранение отдельно.
Скачать **один** файл `extent-results/extent-m3q-rank-continue-155.json`.
Повтор этой ячейки восстанавливает собственный155 checkpoint.

[Протокол и ограничения](EXP-155-protocol.md).

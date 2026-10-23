# EXP-156: существующий 2×T4 notebook, 12 часов

GPU0 исходный Qwen3-1.7B-Base, GPU1 гибрид CURRENT80260. Только оценка.
TPU155 может продолжать обучение независимо. Фиксированная точка156 не меняется.
Новый notebook не нужен: setup/secrets предыдущего T4 notebook прежние,
`requirements-t4.txt` до первого JAX import. Нужен обычный Kaggle GPU image
с PyTorch для CPU-проверки оригинального Qwen через Transformers.

Последняя ячейка:

```python
import importlib
import os
import subprocess
import sys
import time

WALL_HOURS = 11.5
RESERVE_MINUTES = 30
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
from scripts.m3q_quality_benchmark_156 import main
result = main([
    "--max-wall-hours", str(WALL_HOURS),
    "--save-reserve-minutes", str(RESERVE_MINUTES),
    "--max-new-hf-gb", "1",
])
```

В Settings: GPU T4×2, Internet, прежние HF/Telegram secrets. Номер156 уже
задан runner; `--experiment` не нужен. 11.5ч включает всё от загрузок до
сохранения, 30мин резерва внутри. Распределение вопросов не останавливается
после пилота: идёт до всех30737 на модель или лимита времени. Если не успеет,
тот же запуск продолжит оставшиеся задания по собственному отчёту с HF.

Прислать один файл **extent-results/extent-m3q-quality-156.json**.
В нём scores/генерации/статусы обоих worker, одинаковые questionIDs,
paired comparison и исходная диагностика ошибок. Не путать с155 training report.

[Подробный протокол](EXP-156-protocol.md).

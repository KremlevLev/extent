# EXP-154: запуск

Существующий T4 notebook, setup-ячейки прежние.

Последняя ячейка:

```python
import importlib
import os
import subprocess
import sys
import time

EXP = 154
WALL_HOURS = 5.8
RESERVE_MINUTES = 30
HF_CAP_GB = 4
repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo,
               env=globals().get("git_env", os.environ.copy()), check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()
now = time.time()
os.environ["EXTENT_HF_DOWNLOAD_DEADLINE_UNIX"] = str(
    now + WALL_HOURS * 3600 - RESERVE_MINUTES * 60)
os.environ["EXTENT_HF_SAVE_DEADLINE_UNIX"] = str(now + WALL_HOURS * 3600 - 60)
os.environ["EXTENT_HF_RETRY_STATE_DIR"] = "/kaggle/working/extent-hf-download"
from extent.hf_download_retry import install_download_retries
install_download_retries()
from scripts.m3q_continuation_matrix_campaign import main
result = main([
    "--experiment", str(EXP),
    "--max-wall-hours", str(WALL_HOURS),
    "--save-reserve-minutes", str(RESERVE_MINUTES),
    "--max-steps", "262144",
    "--max-new-hf-gb", str(HF_CAP_GB),
])
```

Весь runner 5.8ч, включая резерв 30мин; загрузка/JIT/оценки внутри.
Повтор той же ячейки продолжает собственный checkpoint. Один итоговый файл:
extent-results/extent-m3q-continuation-matrix-154.json

Сравнения и ограничения: [протокол](EXP-149-154-protocol.md).

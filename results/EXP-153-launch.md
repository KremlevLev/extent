# EXP-153 v2: заменить последнюю ячейку T4 notebook

Использовать новый scripts.m3q_rank_growth_153_campaign, НЕ старый
m3q_continuation_matrix_campaign. GPU0:32+32=rank64;GPU1:32+224=rank256.
Сохраняем обученные147факторы, не выполняем дополнительный folding147.
Свежий Adam0, исходный source147current31126. Старые failed states не загружаем.

```python
import importlib
import os
import subprocess
import sys
import time

WALL_HOURS = 11.7  # Включая сохранение; уменьши, если квоты осталось меньше.
RESERVE_MINUTES = 45
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
from scripts.m3q_rank_growth_153_campaign import main
result = main([
    "--experiment", "153",
    "--max-wall-hours", str(WALL_HOURS),
    "--save-reserve-minutes", str(RESERVE_MINUTES),
    "--max-steps", "262144",
    "--max-new-hf-gb", "16",
])
```

Скачать один файл extent-results/extent-m3q-rank-growth-153.json.
Установка/предыдущие ячейки T4 notebook прежние.11.7ч включает45мин резерва.
[Протокол](EXP-153-rank-growth-v2-protocol.md).

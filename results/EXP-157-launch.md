# EXP-157 — TPU v5e8: rank256 против всего декодера

В существующем соответствующем notebook заменить последнюю ячейку целиком.
Setup и HF/Telegram secrets прежние. Запускать после свежего старта kernel;
версии JAX/jaxlib0.11.0, Flax0.12.8, Optax0.2.8, как в исходном EXP155.
Для TPU использовать requirements-continuation-tpu.txt ДО первого импорта JAX.
Для T4 прежний requirements-t4.txt и GPU T4×2.

```python
WALL_HOURS = 9.0
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
from scripts.m3q_decoder_recovery_157_campaign import main
result = main([
    "--max-wall-hours", str(WALL_HOURS),
    "--save-reserve-minutes", str(RESERVE_MINUTES),
    "--max-new-hf-gb", "80",
])
```

9 часов включают загрузку, компиляцию, обучение, оценку и отправку. Рабочий дедлайн — 8ч42мин от входа runner; последние18мин на сохранение/диагностику. Реальная тренировка короче из-за загрузок, оценок и промежуточных сохранений. При плохой сети окончание отправки за18мин не гарантируется; несохранённый хвост явно отмечается pending.

Две ветки чередуются на ОДНОЙ восьмиустройственной TPU, не две отдельные сессии. Обе получают одинаковый префикс FIT по глобальному курсору; итоговый хвост может различаться до одного блока2048. Первые8 обновлений каждой выполняются до длинных блоков. Потолок262144 НОВЫХ шагов на ветку не является ранней целью;8192 также не останавливает кампанию.

Полный декодер: ~1.364B обучаемых параметров. Эмбеддинги и финальная выходная RMSNorm заморожены. Нормализация/tanh из источника сохранены; нового сброса динамики нет. Точная проверка текущего источника на160окнах до обновлений, затем фактическая сверка compiled/streamed градиентов.

Большое состояние ~21.83GB; локально2048шагов (и короткий старт), на HF примерно каждые4ч и при выходе, лимит80GB на сессию. Старые локальные поколения удаляются самим атомарным checkpoint store после публикации нового локального указателя; серверные поколения автоматически не удаляются.

Прислать **extent-m3q-decoder-recovery-157.json**.

Перед тяжёлыми загрузками проверяется50GB свободного места для атомарной смены двух поколений. Если /dev/shm достаточно большой, состояния пишутся туда, иначе в рабочий каталог. При недостатке обоих запуск останавливается до обучения с понятной причиной. Это временные состояния; долговременное хранение — HF.

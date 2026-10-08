# Два аккаунта: EXP106 / EXP107, до8h на каждый

Обнови main в прежней подготовительной ячейке ДО imports. Существующие установка
и secrets подходят: HF_TOKEN, EXTENT_HF_CHECKPOINT_REPO, TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID. Запуск через import в kernel, без !python/subprocess с JAX.

Аккаунт1 — полный decoder, обычный устойчивый глобальный клиппинг:

```python
from scripts.m3q_layer_clip_campaign import main

result = main([])
print("status:", result["status"])
print("progress:", {s: r["DECODER-CE"]["step"] for s, r in result["branches"].items()})
print("HF pending:", result["pending_slots"])
print("notifications:", result.get("notifications"))
```

Аккаунт2 — тот же decoder, послойный адаптивный клиппинг:

```python
from scripts.m3q_adaptive_layer_clip_campaign import main

result = main([])
print("status:", result["status"])
print("progress:", {s: r["DECODER-CE"]["step"] for s, r in result["branches"].items()})
print("HF pending:", result["pending_slots"])
print("notifications:", result.get("notifications"))
```

Два seed на каждом,32768 шагов/seed, одинаковые исходные098, тексты и LR.
Горизонт увеличен16x/ветвь; число ветвей уменьшено с4 до2, общий объём на аккаунт8x.
Сессия до8h:6.5h загрузка/компиляция/обучение,1.5h сохранение. Если setup уже
потратил30min, оба вызова должны использовать main(["--max-wall-hours","7.5"]).
Длительность не гарантирована; при deadline_partial тот же вызов продолжит
совместимый HF checkpoint, сохранив Adam и порядок данных. Подготовленные кампании
могут потребовать следующую сессию. Не запускать две копии одного EXP одновременно.

Перед обучением автоматические проверки: хеши реальных токенов, небольшой
бинарный HF roundtrip, затем отправка полного исходного optimizer checkpoint
с обратным чтением manifest/одного chunk. Если HF недоступен, обучение не начинается.
Завершение обучения и сохранение в HF — разные уведомления. Индивидуальный
primary_gate=None ожидаем: контроль находится на другом аккаунте.

Пришли из /kaggle/working/output:

- extent-m3q-layer-clip-106-summary.txt и extent-m3q-layer-clip-106.json;
- extent-m3q-layer-clip-107-summary.txt и extent-m3q-layer-clip-107.json.

Для общего сравнения после получения обоихJSON:

```python
import json
from scripts.m3q_layer_clip_campaign import compare_campaigns

with open("extent-m3q-layer-clip-106.json") as f:
    control = json.load(f)
with open("extent-m3q-layer-clip-107.json") as f:
    adaptive = json.load(f)
print(compare_campaigns(control, adaptive))
# При одинаковых достигнутых4096/8192/...: target=8192 и т.п.
# Это промежуточный разбор, primary_gate останется None.
```

Код проверен локально на CPU/восьми виртуальных устройствах. Реальные TPU
скорость, дополнительные расходы диагностики, большие HF upload и Telegram
доставка ещё не измерены. Подробный контракт: EXP-106-107-layer-clipping-protocol.md.

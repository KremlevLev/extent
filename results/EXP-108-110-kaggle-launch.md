# Три аккаунта, по одной TPU v5e8 сессии до9 часов

Обнови main в существующей setup ячейке ДО imports. Прежние установка и secrets
подходят. Запускай внутри notebook kernel, без !python/subprocess с JAX.
Никакой предварительный EXP106/107 запуск не нужен.

Аккаунт1 — EXP108, matched batch1/global control:

```python
from scripts.m3q_batch_recovery_campaign import main
result = main(["--max-wall-hours", "8.8", "--save-reserve-minutes", "90"])
```

Аккаунт2 — EXP109, effective batch4/global:

```python
from scripts.m3q_batch_recovery_109_campaign import main
result = main(["--max-wall-hours", "8.8", "--save-reserve-minutes", "90"])
```

Аккаунт3 — EXP110, effective batch4/layer-group adaptive:

```python
from scripts.m3q_batch_recovery_110_campaign import main
result = main(["--max-wall-hours", "8.8", "--save-reserve-minutes", "90"])
```

После любого вызова:

```python
print("status:", result["status"])
print("windows:", {s: r["DECODER-CE"]["step"] for s,r in result["branches"].items()})
print("HF pending:", result["pending_slots"])
print("notifications:", result.get("notifications"))
```

8.8h включает загрузку, компиляцию, обучение и90min резерв сохранения.
12min остаётся снаружи на setup. Если setup потратил больше, вычти время из8.8.
Все три — длинные кампании65536 окон/seed, могут завершить сессию deadline_partial.
Следующая сессия с той же ячейкой продолжает состояние автоматически; не стирай
HF prefixes. Файлы extent-m3q-batch-108/109/110.json и -summary.txt в output.
При pending не уничтожай живой runtime: тот же main с --sync-only синхронизирует
состояния, пока они доступны. После исчезновения RAM несохранённые шаги теряются.

Сопоставление JSON после получения matched endpoints:

```python
import json
from scripts.m3q_batch_recovery_campaign import compare_campaigns
def read(path):
    with open(path) as f:
        return json.load(f)
control, batch, adaptive = [read(f"/kaggle/working/output/extent-m3q-batch-{n}.json")
                            for n in (108,109,110)]
print(compare_campaigns(control, batch))
print(compare_campaigns(batch, adaptive))
# Промежуточный endpoint: compare_campaigns(control, batch, target=8192)
```

JSON разных аккаунтов сначала нужно собрать в указанную папку. По одному JSON
общий gate определить нельзя. Полный протокол: EXP-108-110-batch-recovery-protocol.md.

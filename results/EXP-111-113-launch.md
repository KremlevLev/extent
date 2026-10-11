# EXP-111–113: три аккаунта, до девяти часов каждый

PREPARED locally; TPU feasibility и результаты пока не измерены.
Использовать уже имеющиеся setup/install/secrets cells. Перед новой сессией
обновить checkout `main`; после изменения кода в уже работающем Python kernel
начать чистую сессию. Не запускать JAX subprocess после инициализации TPU.

В каждом аккаунте выбрать TPU v5e-8, Internet on, прежние HF/Telegram secrets.
Одна из трёх ячеек ниже — в существующем notebook process. `8.8` часов включает
restore/compile/диагностику/upload, из них последние90min зарезервированы на sync.
Внешний setup запас12min; если setup занял больше, уменьшить `--max-wall-hours`.
32768/seed — научный горизонт, не обещание завершить его за одну сессию.
Продолжение запускается той же ячейкой: checkpoint/Adam/cursor восстанавливаются.

Аккаунт1 / EXP-111 — protected rank32, исходный EXP-072 ONPOLICY:

```python
%cd /kaggle/working/extent
from scripts.m3q_efficiency_campaign import main as run_exp111
result = run_exp111([
    "--max-wall-hours", "8.8",
    "--save-reserve-minutes", "90",
    "--max-new-hf-gb", "2",
])
```

Аккаунт2 / EXP-112 — обычный full-decoder CE из ТОГО ЖЕ исходного ONPOLICY:

```python
%cd /kaggle/working/extent
from scripts.m3q_efficiency_112_campaign import main as run_exp112
result = run_exp112([
    "--max-wall-hours", "8.8",
    "--save-reserve-minutes", "90",
    "--max-new-hf-gb", "35",
])
```

Аккаунт3 / EXP-113 — тот же full-decoder CE, случайны только Mamba:

```python
%cd /kaggle/working/extent
from scripts.m3q_efficiency_113_campaign import main as run_exp113
result = run_exp113([
    "--max-wall-hours", "8.8",
    "--save-reserve-minutes", "90",
    "--max-new-hf-gb", "35",
])
```

112/113: сначала три LR pilots по1024 updates на обоих seeds; выбор по validation,
потом MAIN с оплачиваемым replay выбранного pilot. Если диапазон LR неработоспособен,
не выдавать это за доказательство превосходства111. Общая preparation cost069/072
пока UNKNOWN; marginal recovery speed считается отдельно.

HF: только metrics периодически; большие состояния локально каждые4096/deadline,
затем в конце вызова одна актуальная generation на активный slot. Обычно112/113
добавляют около33GB каждый; cap35 decimal GB на вызов,111 cap2GB. Это ограничение
payload, не проверенная оставшаяся HF квота. Следующие сессии добавляют generations.
Старые remote файлы/история не удаляются. Внезапное падение VM до sync теряет локальные
шаги; сохранённый JSON откатывается к durable cursor, стоимость повторений учитывается.

Нужен запас host RAM>=80GiB; initial/memory/compiler/data/source checks выполняются
до train updates. FP32-forward probe по умолчанию выключен; включать отдельным
`--fp32-probe` только при подтверждённой memory feasibility. Основная диагностика
по слоям/components/позициям, accepted updates/BF16 и rollback включена.

Пришли после каждой сессии две небольшие сводки из `/kaggle/working/output`:
`extent-m3q-efficiency-NNN.json` и `extent-m3q-efficiency-NNN-summary.txt`.
`deadline_partial` означает остановку по бюджету. `gate=null` не PASS/FAIL.
Уведомления отдельно: start, failure, training stopped/finished, saved или HF pending.
Если есть pending, продолжать sync до выключения исходной VM:

```python
# В том же аккаунте и исходном живом runtime; импорт run_expNNN уже выше.
result = run_exp112(["--sync-only", "--max-wall-hours", "0.9",
    "--save-reserve-minutes", "15", "--max-new-hf-gb", "35"])
```

Для111/113 заменить функцию и cap. Sync-only не восстановит невыгруженные файлы
из уже уничтоженной VM. Время sync-only также расходует доступную сессию.

Протокол: [EXP-111-113-recovery-efficiency-protocol.md](EXP-111-113-recovery-efficiency-protocol.md).
Read-only storage audit: [EXP-111-113-hf-storage-readonly.json](EXP-111-113-hf-storage-readonly.json).

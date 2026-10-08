# Два аккаунта Kaggle: EXP104-v2 + EXP105, до8h каждый

**8 октября:** оба коротких протокола completed по2048 шагов (~1h), wall cap8h
не означает обязательных8h обучения. Финальные состояния в присланныхJSON PENDING.
Если RAM/runtime ещё живы, НЕ останавливай сессию до отправки; обычная новая
сессия не восстановит незагруженный RAM checkpoint. Для отправки по существующим
путям обнови код, перезагрузи импортированные модули (не уничтожая runtime/tmpfs)
и вызови соответствующий `main(["--sync-only"])`. Теперь Telegram включён по
умолчанию, нужны доступные `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`; JSON сохраняет
`notifications`, `checkpoint_sync_errors`, `checkpoint_upload_status`. Пропуск
уведомления больше не скрыт; `--no-telegram` отключает отправку. Завершение обучения
и успешное сохранение в HF сообщаются отдельно. Тип/HTTP-код новых sync ошибок
логируется; причину старых pending по имеющимсяJSON восстановить нельзя.

Обнови main в существующих подготовительных ячейках; перезапусти процесс ноутбука,
если старые модули уже были импортированы. Используй прежние установки и secrets
`HF_TOKEN`, `EXTENT_HF_CHECKPOINT_REPO`. Оба аккаунта должны читать фиксированные
исходники и писать в нужный HF dataset. Разные prefixes защищают от смешения
кампаний; не запускай две копии одного EXP одновременно в одном destination.
Не используй subprocess после инициализации JAX.

Аккаунт1 — EXP104, продолжить полный decoder:

```python
from scripts.m3q_paper_decoder_recovery_campaign import main

result = main([])
print("status:", result["status"])
print("gate:", result["aggregate"])
print("HF pending:", result["pending_slots"])
```

Контроль seed123 продолжится с сохранённых512 и прежних моментов Adam. Dense
seed123 начнёт с0 исходного098. Протокол/источник/данные/LR остаются прежними.
Стартовые NLL пересчитываются из первоначального источника на общем BF16 layout;
старые NLL сохраняются в `start_uncanonical_v1`. Допуск1e-4 не увеличен.

Аккаунт2 — EXP105, полностью открыть только Mamba:

```python
from scripts.m3q_mamba_capacity_recovery_campaign import main

result = main([])
print("status:", result["status"])
print("gate:", result["aggregate"])
print("HF pending:", result["pending_slots"])
```

Все параметры24 Mamba открыты; MLP/GQA/внешние нормы/словарь заморожены. Два
seed и protectedrank32 контроль,2048 шагов на ветвь, CE/AdamW1e-5 как в104.
Цель — проверить ограничение rank32 отдельно от дообучения остальных Qwen слоёв.
Контроли и тексты общие с104, это согласованное сравнение, не независимая репликация.

Для каждого запуска8h включают1.5h запаса на сохранение. Если верхние ячейки
потратили30min изсессии, используй `main(["--max-wall-hours", "7.5"])`.
Checkpoint defaults в RAM (`/dev/shm/extent-exp104-checkpoints` либоexp105),
исходники/результаты в прежних disk-каталогах. Код проверяет RAM/место и HBM.
Это исправляет недостаток20.6GB диска, а не гарантирует скорость или память TPU.
Полное состояние104~16.37GB,105~3.08GB. HF публикация manifest после всех частей.

Тот же вызов продолжит последний совместимый durable checkpoint. Если не успело
загрузиться в HF, уничтожение runtime теряет незагруженные локальные шаги.
Проверяй `pending_slots`, `summary_sync_error`, `durability_error`.
`main(["--sync-only"])` отправит сохранившиеся локальные состояния без обучения.

Пришли четыре файла из `/kaggle/working/output/`:

- `extent-m3q-paper-decoder-recovery-summary.txt` и `.json`;
- `extent-m3q-mamba-capacity-recovery-summary.txt` и `.json`.

Локальные CPU/virtual8 тесты прошли; реальный TPU104-v2 и105 ещё не запускались.
Все4 ветви каждого EXP могут не уложиться в одну сессию; при partial gateNULL,
при OOM/паритете пришли JSON и сообщение из ячейки. Не обходи защитные проверки.

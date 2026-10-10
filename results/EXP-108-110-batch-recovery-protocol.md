# EXP-108–110: effective batch и клиппинг, три аккаунта

PREPARED, 10 октября 2026. TPU результатов ещё нет. Это продолжение проверки
восстановления гибрида Qwen3-1.7B/Mamba-3 MIMO; не обучение новой модели с нуля.

## Основание

Полученный EXP-106 completed: два seed по 32768 обновлений, checkpoint durable.
Gain относительно исходного состояния: Wiki 0.01447 / 0.06496; PG19
0.01236 / 0.08737. Устойчивого восстановления >=0.1 нет. Суммарное время
трёх runner-сессий около 9.214h; это не одна девятичасовая сессия.
Исходный присланный EXP-107 — deadline_partial, seed123=27799, seed456=24576.
На общем endpoint24576 его PG19 относительно106: хуже на0.01070 у123,
лучше на0.00402 у456. Победа AGC не подтверждена. Отдельный gate106 NULL
требует сопоставления обеих кампаний; NULL не превращаем в положительный результат.
Исходные логи: EXP-106-completed.json, EXP-107-session-1.json.

## Зарегистрированное сравнение

| Аккаунт | Эксперимент | Microbatch | Effective batch | Clip после усреднения |
|---|---|---|---|---|
| 1 | EXP-108 | 1x256 | 1 окно | устойчивый global norm1 |
| 2 | EXP-109 | 1x256, четыре последовательных прохода | 4 окна | global norm1 |
| 3 | EXP-110 | то же | 4 окна | group relative0.01*max(weightnorm,0.001) |

В109/110 четыре НЕклиппированных FP32 градиента на неизменённых весах
усредняются; только затем clip и один AdamW update. Потери — CE по истинным
следующим токенам; дополнительной дистилляции здесь нет. Скан не держит четыре
набора активаций одновременно. Проверяются все microbatch logits/loss/gradients,
средний gradient, предложенные веса и optimizer state. Nonfinite proposal
отбрасывается целиком; сохраняется последнее безопасное состояние.

Во всех аккаунтах фиксированные исходные EXP098 final16384 для seed123/456,
проверенные исходным source manifest/SHA. Полный decoder открыт, vocabulary
заморожен; FP32 masters/moments, BF16 forward, прежний FSDP/tensor mesh1x4x2.
Новый отдельный runner/core не изменяют контракты104–107.

Endpoint65536 прочитанных context256 окон НА SEED: 16,777,216 входных токенов,
33,554,432 на аккаунт. CE предсказывает255 позиций в каждом окне.
108:65536 Adam updates/seed;109/110:16384. Milestones каждые8192 окон,
порядок seed чередуется. Никаких остановок из-за слабой промежуточной NLL.

Одинаковое расписание по ЧИСЛУ ОКОН: warmup1024, peak1e-5, cosine до1e-6
на65536. В optimizer schedule count умножается на accumulation.
AdamW betas0.9/0.95, eps1e-8, decay0.1 только matrices/tensors без bias.
Это сравнение режима effective batch при одинаковом бюджете токенов.
Число Adam updates, суммарный per-update weight decay и вычислительные затраты
не одинаковы; возможный выигрыш нельзя приписать исключительно «меньшему шуму».
109 против108 — основное сравнение batch;110 против109 — добавка клиппинга.
110 против108 — вторичное сравнение комбинации.

## Данные и критерий

Train WikiText103 train: offset48,758,784,65536x256; диапазон заканчивается
перед65,536,000. Продолжение идёт по cursor окон, без повтора этих окон.
Wiki validation offset114688/32 и test294912/24 переиспользованы:
вспомогательная диагностика, не новая независимая проверка.
PG19 test offset360448/128 — новый основной срез после диапазона106/107.
Версии tokenizer/datasets и SHA всех окон фиксируются data-preflight.json.
Контекст256 и короткие тестовые срезы ограничивают переносимость выводов.

Для каждого основного сравнения на зарегистрированном FINAL обоих seed:
PG19 candidate лучше control >=0.1 NLL И своего start >=0.1;
Wiki candidate не хуже своего start более чем0.1. Отдельные аккаунты имеют
gateNULL; compare_campaigns требует совпадения контрактов, количества тестовых
окон, finite metrics и стартового NLL по каждому окну с точностью1e-4.
Промежуточные matched milestones только exploratory, gateNULL.

## Девятичасовая сессия и продолжение

Default вызов8.8h:12min внешний запас на setup,90min ВНУТРИ вызова на save/sync;
до7.3h на data/model restore/compile/training. Если setup дольше12min,
уменьшить max-wall-hours. Допускается лимит9h, но внешний запас исчезает.
Endpoint длиннее предыдущего106 вдвое и может требовать следующую сессию.
Длительность загрузки/компиляции/upload непредсказуема: ровно9h вычислений
или полный final в одной сессии не обещаются. deadline_partial штатен.

Тот же вызов автоматически восстанавливает weights, Adam, schedule и cursor
из HF. Cursor кратен accumulation, оба optimizer counters равны cursor/accumulation.
Checkpoint только между завершёнными updates; частичный scan не checkpointится.
Все аккаунты используют отдельные HF prefixes exp108/109/110-batch-recovery.
Telegram: started, failed, stopped/finished, finished-and-saved или pending;
требуются существующие secrets, журнал notifications сохраняет результат отправки.

Full checkpoint около16.37GB. История immutable chunks может занять сотни GB
на аккаунт, даже в общем HF repo. Без автоматического удаления исходников.
Проверка durability — remote manifest плюс sample chunk SHA, НЕ полный reload
каждого16GB checkpoint. HF pending должен стать пустым до уничтожения runtime.
429 cooldown учитывается; резерв не гарантирует успешную отправку.

## Проверки

До запуска: реальный pinned tokenizer/data capacity/SHA локально; CPU/virtual8
tiny-model accumulation, один Adam update на четыре microbatch, бинарный
checkpoint/следующий update, finite guard и matching contracts. На TPU runner
дополнительно проверяет host RAM, allocator HBM и compiler memory. Policy estimate
включает дополнительный FP32 gradient carry; это не измерение полной TPU модели.
Большая модель и реальные HF/Telegram в локальных тестах не запускались.

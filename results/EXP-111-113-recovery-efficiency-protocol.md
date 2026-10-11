# EXP-111–113: скорость восстановления и локализация ухудшений

DESIGN, 11 октября2026. Пользователь разрешил расписать1–3 эксперимента.
Решение: три независимых аккаунта, по одной сессии до9h. Протокол подготовлен;
новые runners, source/data SHA preflight и TPU feasibility ещё НЕ готовы.
Этот документ не является инструкцией запуска. Старые111–113 результатов нет.

## Что проверяем

EXP108–110 partial: накопление4 не улучшает качество на matched24576;
AGC устраняет часть ухудшений seed456 в batch4, но не превосходит batch1
устойчиво. Все checkpoints durable. Эти наблюдения не равны final gate.
Одни конечные градиенты не доказывают успешную оптимизацию. Градиенты разных
слоёв сильно различаются, однако нормы не учитывают размеры групп и ещё
не устанавливают причинность. Нужны функциональные и причинные замеры.

Быстрый спад12→7 наблюдался в rank32 CE recovery из EXP072 ONPOLICY;
его не сравнивали здесь с обычным full-decoder CE из ТОГО ЖЕ старта.
EXP104–110 начинались из уже восстановленного EXP098, около7.5,
и не отвечают на вопрос скорости спада12→7.

## Три эксперимента

| EXP / аккаунт | Исходное состояние | Обучение | Вопрос |
|---|---|---|---|
| 111 /1 | pinned EXP072-v2 ONPOLICY, БЕЗ EXP098 adapters | protected INOUT rank32, CE | повторяется ли быстрый спад и где возникает потолок? |
| 112 /2 | точно тот же ONPOLICY | обычный full-decoder CE | насколько полезна именно защищённая low-rank коррекция? |
| 113 /3 | Qwen outside mixers, случайные Mamba | точно тот же full-decoder CE, что112 | приносит ли предварительное обучение Mamba пользу обычному recovery? |

Исходные model seeds123/456, архитектура1.7B:24 Mamba-3 MIMO и4 GQA
на[6,13,20,27]. Vocabulary frozen во всех трёх; никаких MLA/KD новых
training objectives. CE true-token; batch1/context256; без accumulation.
FP32 trainable weights/coordinates/moments, BF16 forward, stable global clip1.

В111 все base weights frozen, только zero-change correction coordinates
в in/out kernels; dt/decay/trapezoid/angle columns protected, rank32.
Начальный effective model должен быть bitwise равен ONPOLICY.
В112 открыт весь decoder, остальные Qwen параметры первоначально неизменны.
Начальный effective BF16 model112 должен быть bitwise равен111; NLL parity
по каждому оценочному окну<=1e-4 до обучения. Свежие optimizer states.

В113 случайны ТОЛЬКО mamba subtrees: native Mamba3MIMO initializer, фиксированные
PRNG123/456, конфигурация/shape/dtype как112. Norms/MLP/attention/vocabulary —
точно тот же оригинальный Qwen. Использовать compose_parameters и отдельный
manifest random subtree hashes. Нельзя случайно заменить весь decoder или
использовать ранее обученный EXP069 mixer как «random». Не подгонять random
до NLL12 и не вычитать стоимость разогрева из сравнения.

111 vs112 сравнивает практические recovery recipes: число параметров,
параметризация и LR различаются. Это не чистая абляция protection.
112 vs113 сравнивает initializers при одной процедуре full-decoder обучения.
111 vs113 — сравнение целых pipelines с обязательным учётом подготовки ONPOLICY.
Называть111 «чистый перенос весов» нельзя: ONPOLICY уже включает обучение.

## LR: контроль не должен проиграть из-за одного неудачного значения

111: установленный Adam3e-4 без decay/warmup, как SAFE-PROTECTED-R32.
112/113: Adam betas0.9/0.999, eps1e-8, decay0; три prespecified peak LR:
1e-5 /1e-4 /3e-4. Linear warmup128 updates, далее constant до registered final.
Adam для111 — те же betas/eps; дополнительного LR search111 не заявляем.

Каждый full-decoder LR pilot:1024 окон/seed из одного и того же начала train.
Выбор одного LR на аккаунт по СРЕДНЕМУ validation NLL обоих seed на1024.
Сначала все pilots обоих seed, потом selection; test не участвует.
Eligible только если оба seed finite. Если все pilots failed, результат
«данный ordinary optimizer range неработоспособен», а не доказательство
скорости нашего метода. Ошибки HBM/compile/storage не являются training loss.

Сохранять выбранные1024-step states каждого seed, продолжать с их Adam и
cursor1024; не перезапускать их бесплатно и не сбрасывать моменты.
Каждый pilot использует глобальные step-dependent настройки того же расписания.
Все успешные и failed trial updates, compile/eval/upload время и repeated
training tokens входят в стоимость LR search. Только selected trajectory
считается основной кривой качества; не смешивать её со стоимостью поиска.
Для initializer абляции дополнительно показать112/113 matched1024 результаты
на КАЖДОМ одинаковом LR, чтобы выбор разных LR не выдавать за чистый эффект init.

## Горизонт и данные

32768 уникальных окон/seed=8,388,608 input tokens на выбранной траектории;
16,777,216 на аккаунт без discarded pilots. Все аккаунты получают одни окна.
112/113 дополнительно до1,048,576 input tokens discarded pilots/аккаунт
(2 проигравших LR x1024 x256 x2seed). Указывать input tokens и255 CE targets/window.
Начала данных ниже — CANDIDATE, требуется реальный capacity/SHA preflight:

- Wiki103 train offset65,536,000 /32768 windows: после EXP108–110 train.
- Wiki103 validation offset122,880 /64: LR selection.
- Новый Wiki103 locked test offset301,056 /64; не выбирать checkpoint по нему.
- Новый PG19 test offset393,216 /128; test manifest/revision как прежние runs.
- Исторический reproduction panel Wiki103 test offset32,768 /64:
  тот же диапазон, на котором EXP097 дал NLL12.968/12.277→7.267/7.310.
  Он переиспользован и не является независимой проверкой качества.

Pinned Qwen tokenizer ea980cb0a6c2ae4b936e82123acc929f1cec04c1,
Wiki dataset b08601e04326c79dfdd32d625aee71d232d685c3,
PG19 manifest4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b.
Если capacity fail: остановить preflight и зарегистрировать другой диапазон
ДО обучения. Не брать modulo/повторы и не менять test после просмотра результатов.
Подготовка исходного ONPOLICY: metadata/SHA pins EXP069→EXP072 как current engine;
не подставлять latest или EXP098 final. Заморозить конкретный HF revision до run.

## Скорость: считать честно

Validation каждые256 updates, reproduction/locked Wiki каждые1024;
PG19 только start и4096/8192/16384/24576/32768. Чередовать seeds по4096 блокам.
Thresholds исторической Wiki панели: NLL<=12/10/8/7.5/7.0.
Основной practical target7.5, а вопрос буквального «до7» показывать отдельно:
если7.0 не достигнут, писать «не достигнут», не округлять7.5 до7.
Устойчивое пересечение: два последовательных замера ниже порога. Указать
интервал между last-above и first-below; точный момент внутри не известен.
Новые Wiki/PG19 кривые показывать параллельно для контроля переноса.
Никакой остановки/выбора best checkpoint по threshold/test.

Для каждого threshold и каждого seed вывести:
1. Selected-path updates, input/target tokens и чистое training time.
2. Полную стоимость этого запуска, включая LR pilots и диагностику.
3. Полную pipeline стоимость EXP069/072 + recovery для111/112.
4. Отдельно engineering wall: download/restore/compile/eval/save/upload.

Историческую preparation стоимость учитывать только для реально использованного
пути; search over other architectures/methods выводить отдельной строкой.
Если по069/072 нет достоверного token/compute учёта, total ускорение пометить
UNKNOWN. Cached checkpoint не делает подготовку бесплатной. Нельзя суммировать
wall всех параллельных исторических ветвей как время одной выбранной модели.
TPU queue wait отдельно от TPU compute; teacher forward tokens отдельно от
student tokens. FLOPs не оценивать просто суммой teacher/student token counts.
Если baseline threshold не достигнут, отношение скоростей не число:
показать right-censored observation; нижнюю границу давать только при
сопоставимом бюджете/учтённых pilots. Основные данные — curves, не один ratio.

## Где ухудшается: общая диагностическая панель

Не использовать full-probe старого composition_diagnostics без адаптации:
его training gradient включает KL+0.1CE и naive norms, а здесь нужны именно
CE gradients и stable norms. return_hidden_states уже есть в модели.
Панель состоит из первых8 validation windows; выбор фиксируется до обучения.
Диагностические backward passes не обновляют Adam/weights и считаются отдельно.

Старт,1024,4096,8192,16384,24576,32768:

- По каждому decoder layer: hidden RMS/maxabs, residual contribution RMS,
  relative contribution error и cosine against original Qwen. Сравнивать
  вклад блока h_out-h_in, а не только полный hidden с общим residual.
- На одинаковом Qwen входе блока и на реальном hybrid входе: ошибка Mamba
  относительно Qwen mixer; это различает local approximation и input shift.
  Qwen и hybrid streams не считать совпадающими после предыдущих замен.
- CE gradient по каждому in_proj component: gate/value/B/C/dt/decay/trap/angle;
  отдельно out_proj, norm, MLP и retained attention. FP32 max-scaled norm,
  norm/sqrt(parameter_count), zero fraction, finite count, clip retained fraction.
- Фактический post-Adam update RMS/max/relative norm по тем же группам,
  FP32 parameter displacement от старта и доля реально изменившихся effective
  BF16 kernel entries. Для111 смотреть A/B coordinates И effective kernels.
  Нулевые защищённые/замороженные группы отмечать expected-frozen.
- Качество по одному и тому же окну: CE по позициям0–63/64–127/128–191/192–254
  предсказаний (первый target token исключён); per-window NLL и верхний хвост.
  Это проверяет накопление ошибки по позиции в пределах context256,
  не доказывает long-context поведение или длину документа.
- BF16 diagnostics сравнивать с FP32-forward на тех же effective weights
  только на2 фиксированных validation windows при memory feasibility.
  Не менять train math. Если FP32 probe не помещается, отмечать unavailable,
  не уменьшать production dimensions и не считать tiny результат эквивалентом.

Cause-oriented probe: на milestones4096/8192/16384 по validation выбрать
2 decoder layers с наибольшим ухудшением contribution error от их старта.
На фиксированной панели временно вернуть ОДИН такой block к собственному
начальному subtree, остальные current; измерить изменение CE. Затем соседний
контрольный layer index=min(j+1,27), исключая совпадение; никаких train updates.
Улучшение показывает, что изменение блока вредило текущей композиции,
но не доказывает единственную причину и не предлагает deployable treatment.
Для111 временно обнулить corrections этого блока. Если блок не менялся,
это expected-null control. Правильность rollback требует идентичности NLL
при замене current subtree на копию current и untouched checkpoint после probe.

Никаких full logits/activations для28 слоёв сразу в файл/Git. Только агрегаты
и per-window losses, streaming по слоям/окнам. Teacher residency последовательно;
добавленная диагностика не должна молча съесть HBM full-decoder шага.

## Девять часов, checkpoints и ограничения

Default8.8h ВЕСЬ вызов,90min внутри на save/upload,12min внешний setup запас.
До7.3h restore/compile/pilot/train/diagnostics. Если setup длиннее12min, уменьшать
лимит. Не гарантировать final32768 в одной сессии; fixed horizon может требовать
continuation. Каждые4096 окон/seed и при deadline сохранять parameters/Adam/
LR-search phase/selected LR/data cursor/evaluation cursors/source hashes.
Checkpoint111 небольшой,112/113 full ~16.37GB; immutable history расходует HF.
Source hashes, initial model parity, data SHA, memory allocation и compiler
memory checks до обучения. Новые prefixes exp111/112/113-recovery-efficiency;
отдельные notifications start/failure/deadline/saved/pending.

112/113 нельзя запускать прежним108 main: он начинает из EXP098 и не содержит
LR pilots/новой диагностики. Точечный --experiment override не меняет этот факт.
Новые entry points должны быть подготовлены и проверены отдельно.

Registered conclusion: final quality gap to Qwen и crossing-cost curves по
обоим seed. Candidate practical win:7.5 достигнут обоими seed быстрее контроля
по учтённым training tokens, fresh locked Wiki/PG19 не хуже control более0.1
на matched selected-path budget. Pipeline win заявлять только с известной
полной подготовительной стоимостью. Пока все hypotheses, не результаты.

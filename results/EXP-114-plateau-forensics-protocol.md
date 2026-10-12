# EXP-114: обследование плато восстановленного Mamba-3/Qwen

PREPARED locally, 12 октября 2026. Пользователь разрешил один подробный
диагностический эксперимент. Один аккаунт, TPU v5e8, сессия до9 часов.
Runner и запуск: `scripts/m3q_plateau_forensics_114_campaign.py`,
`results/EXP-114-kaggle-launch.md`. TPU-результатов EXP-114 нет.

### Предварительные уточнения после локальной реализации, ДО TPU

- Patient revision `f839c2a898c3e9eff9907ea3cf1e022fe16f3e09`, оба32768
  manifests и580 binary paths проверены без скачивания model weights.
  `EXP-114-source-pin.json` содержит exact contracts/chunk SHA/reference NLL.
- Все исходно предложенные offsets поместились; `EXP-114-data-preflight.json`
  содержит реальные token SHA и article provenance. Точных повторов окон нет.
  Между B и C Wiki есть одна общая статья: token-disjoint НЕ означает
  article-independent. PG19 также contiguous book slice; выводы описательные.
- CPU tiny BF16 control-flow меняет округление: выключенный rollback/boundary
  имеет delta NLL около0.000132 относительно обычного executable. Поэтому
  intervention effects измеряются относительно mask0 В ТОМ ЖЕ executable.
  Canonical identity background отдельно сохраняется по каждому окну;
  зарегистрированный предел его maxabs0.01 NLL, иначе модуль invalid/stop.
  PRIMARY practical signal0.1 остаётся прежним, минимум10x данного предела.
  FP32 functional scope проверки строгие; original32768 checkpoint replay
  остаётся<=1e-4. Это pre-TPU amendment, не изменение gate по TPU результату.
- State reset сравнивается с теми же выбранными слоями и period0 (normal carry)
  в том же executable: даже выбор эквивалентной recurrence ветви в BF16 меняет
  округление. Этот control background относительно canonical также публикуется
  и ограничен0.01; period0 имеет delta0 по определению, не считается улучшением.
- Readout улучшение считается относительно identity readout того же пути,
  его canonical background также ограничен0.01 и публикуется отдельно.
- Dynamics time series сохраняются32-token bins с mean/min/max, включая
  extrema, а не полные per-token states. Token losses и spikes не сглаживаются.
  Это ограничивает report size и HF payload; bin resolution указывается в JSON.
- FIT features вычисляются последовательно on demand, без полного feature
  cache. Это дополнительные frozen forward passes, их время учитывается.
- Итоговые CSV и графики можно строить офлайн из полного JSON: runner сохраняет
  сырые token losses, intervention tables и агрегированные layer/dynamics probes.
  Выполнение всех модулей за одну9h сессию НЕ подтверждено локально.

## 1. Вопрос и основания

Нужно выяснить, какая часть оставшегося разрыва качества связана с локальными
блоками, изменением входов для следующих слоёв, рекуррентной динамикой,
выходным представлением и направлением фактических обновлений.
Цель — получить проверяемое основание для следующего лечения, а не очередной
перебор LR, batch, clipping или KD. Обследование может установить несколько
механизмов одновременно; отрицательная проба не доказывает их отсутствия.

Проверенные пользовательские результаты:

- EXP-111 completed: обе MAIN32768, durable, pending[]. Fresh Wiki NLL
  7.345073/7.309848 против Qwen 3.135366 на том же диапазоне; PG19
  9.934455/9.978602 против Qwen 3.085908. Существенный разрыв остаётся.
- EXP-113 пока partial26662/24576, durable. LR1e-4; сильные колебания.
  На seed456/8192 откат блока8 к собственному старту уменьшал NLL панели
  на17.150178; соседний блок9 ухудшал её на1.227560.
- В EXP-111 откат блока26 при8192/16384 сильно ухудшал обе seeds:
  изменение этого блока необходимо текущей композиции. Большая ошибка
  относительно Qwen не равна вредности блока.
- Эти rollback результаты получены на8 selection-validation окнах с выбором
  слоёв по той же диагностике. Они мотивируют обследование, но не являются
  независимым подтверждением общей причины плато.
- EXP-112 пользователь уже запустил; результата пока нет. EXP-114 не зависит
  от его завершения. EXP-110 сейчас не продлеваем.

Свидетельства: полученные полные JSON EXP-111 SHA256
`a85d7dcbc9e4f57a1e06346a72474b646779456c8bb0dba53654a9c3aaa9ba8d`,
EXP-113 `20e444606445617607f6d89e733ee548adab8d91306bdf6434fb323d8db4fbd1`.
Это хеши пользовательских файлов, не хеши model checkpoints.

## 2. Исходные состояния и сравнения

Обязательные пациенты: EXP-111 MAIN final32768, seeds123/456. Восстановить
реальные FP32 correction coordinates, Adam и исходную BF16 базу EXP-072;
не подставлять EXP-098, random или заново обученный аналог.
Checkpoint prefix: `experiments/exp111-recovery-efficiency/checkpoints`,
slots `123/MAIN`, `456/MAIN`. В полученном JSON payload каждого slot85782792 bytes.

База: Qwen3-1.7B-Base,24 Mamba-3 MIMO +4 GQA на[6,13,20,27], context256,
источник/tokenizer revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
Исходные EXP069/072 pins: `results/EXP-111-113-source-pin.json`.
Protected INOUT rank32, остальное frozen; dt/decay/trapezoid/angle columns protected.
Защита весов не делает динамику неизменной: входы для этих проекций меняются.

Перед реализацией закрепить отдельную HF revision и manifests/chunk SHA
для обоих MAIN32768. Не читать изменяемый latest во время эксперимента.
Сохранить исходные contract hashes, tree hashes эффективной модели и optimizer,
проверить count32768. При отсутствии бинарного endpoint — остановка, не fallback.
Canonical replay NLL на прежних панелях должен совпасть с сохранённым final
по среднему и per-window в пределах1e-4; проверка выполняется до новых probes.
Если расхождение связано с lowering, записать и разобрать его, а не ослаблять
порог по наблюдённому результату.

Референсы: оригинальный Qwen на тех же текстах и неизменённый собственный старт
EXP-072 ONPOLICY. Использовать одинаковый evaluator/precision/layout.
EXP-113 и112 сейчас НЕ добавляются как обязательные источники: это изменило бы
бюджет и зависимость от незавершённых запусков. Позже можно зарегистрировать
отдельную репликацию обследования, не смешивая checkpoint stages.

## 3. Данные: обнаружение, проверка и закрытая оценка

Все нижние диапазоны прошли token-capacity/SHA preflight без запуска модели.
Полный data manifest публикуется в Git вместе с runner.
Tokenizer, Wiki/PG19 revisions те же, что111–113.

| Роль | Источник | Token offset | Окна x256 | Использование |
|---|---|---:|---:|---|
| A discovery | Wiki103 validation |122880|64| прежняя панель111; первые16 для широкого sweep |
| B verification | Wiki103 validation |180224|32| проверка заранее выбранных вмешательств |
| C locked Wiki | Wiki103 validation |196608|64| единственная финальная оценка выбранных вмешательств |
| C locked PG19 | PG19 test |458752|128| единственная финальная оценка переноса |
| FIT | Wiki103 train |73924608|4096| обучение только диагностической readout-поправки |
| T-seen | Wiki103 train |65536000|64| действительно использованные111 окна, без отбора по loss |
| T-new | FIT первые64 |73924608|64| ещё не использованный111 train-фрагмент, до FIT |

FIT начинается сразу после32768 окон EXP-111 и с ними не пересекается.
T-new является частью FIT: после подгонки readout это уже training evaluation.
T-seen и T-new не объявлять случайной выборкой всего обучающего корпуса.

Preflight: фактическая capacity, token SHA, dtype, shapes, непересечение
диапазонов в пределах одного split и поиск прежнего использования в протоколах.
Проверять text/document provenance и дубликаты: разные token offsets не
гарантируют независимые статьи. Если один документ пересекает панели,
пометить зависимость; не выдавать окна за независимые репликации.
При нехватке capacity зарегистрировать замену ДО просмотра NLL.
Не использовать modulo, повтор окон или диапазоны прежнего locked111 как новый C.

A используется для ranking и выбора. B проверяет результат этого выбора,
но не даёт повторно искать лучший слой или alpha после провала.
C не участвует ни в ранжировании, ни в выборе гиперпараметров/endpoint.
Все selected IDs/alphas/readout endpoint зафиксировать и выгрузить до C.
Внешние модельные seeds123/456 показывать отдельно.

## 4. Этап I: профиль ошибки и траектория через весь decoder

Замороженные111/Qwen/ONPOLICY: canonical NLL на A/B, T-seen/T-new.
Для A первые16 окон собрать по каждому target:
NLL student и Qwen, excess NLL, true-token rank (с оговорённым правилом ties),
top1/top5 accuracy, entropy, true-token logit и margin к лучшему другому токену.
По окнам: mean/median/p90/max, доля total excess loss в худших10% targets.
Знаменатель excess loss может иметь отрицательные элементы: отдельно positive
excess и signed sum, не прятать отрицательные под abs.

Позиционные диапазоны prediction indices0–63/64–127/128–191/192–254.
Первые targets не называть «исключёнными»: index0 предсказывает input token1,
а input token0 не является target. Никакой off-by-one смены относительно111.
Срезы по частотности истинного токена считаются по FIT, а не по C.
Срезы по сложности Qwen: NLL<2,2–5,>=5 на A; это описательные категории.

По28 слоям и отдельно двум sublayers mixer/MLP:
hidden RMS/max/finite, residual contribution RMS, cosine/relative error,
дрейф входа относительно Qwen, norm до/после RMSNorm, amplification ratio.
Добавить final RMSNorm и нормированное представление перед tied readout.
Показывать распределения по окнам, а не только одно среднее по слоям/seeds.

Matched-input разложение выполняется на одинаковых токенах и boundary inputs:
F_S(h_S)-F_Q(h_Q) = [F_S(h_Q)-F_Q(h_Q)] + [F_S(h_S)-F_S(h_Q)].
Первое — локальное различие операции на teacher input, второе — эффект смены
входа для student operation. Записать norms и cross term: нормы слагаемых
не складываются, эти величины не являются аддитивными процентами причин.
Дополнительно F_S(h_S)-F_Q(h_S) сравнивает операции на реальном входе student.
Mamba и Qwen сравниваются по функции mixer, без residual identity; MLP отдельно.

Новый вопрос относительно070/111: что осталось ПОСЛЕ восстановления до плато,
а не повторное доказательство input shift до recovery.

## 5. Этап II: полный sweep функциональных вмешательств

Все вмешательства temporary/read-only; optimizer и исходные arrays не меняются.
На A первых16 окнах, для каждой seed:

1. **Rollback всех28 блоков** к собственному ONPOLICY старту. Frozen blocks
   EXP111 должны давать expected-null. Вредное изменение отличается от
   необходимой компенсации upstream ошибок. Не выбирать только top2 по MSE.
2. **Teacher mixer replacement всех24 Mamba позиций.** Только mixer становится
   оригинальным Qwen attention, на текущем student input; MLP/norm/suffix current.
   Нельзя заменять Mamba-packed weights на Q/K/V по совпавшим shapes.
   Это диагностическая гибридная функция, не production Mamba.
3. **Teacher boundary rescue всех28 выходов блоков.** Подать h_Q после блока j
   в неизменённый student suffix j+1..27 и исходный final norm/readout.
   Показывает, помогает ли исправление upstream stream данной границы.
   Не локализует причину в одном блоке и может создавать чужое для suffix распределение.

Записать per-window paired delta NLL=intervention-current, baseline и итоговую
NLL, время, intervention type/layer, identity lowering floor.
Для каждой пары(seed, intervention family) выбрать на A лучший слой по средней
delta; tie — меньший индекс. Итого не более6 одиночных кандидатов.
На B выполнить ТОЛЬКО эти кандидаты, непосредственный сосед (j+1 либо j-1 у27)
и identity control. IDs не менять по B. На C оценить все6 выбранных кандидатов
с identity: включить провалы B, не отфильтровывать неудобную seed.

Взаимодействие: на A взять два лучших teacher-mixer слоя отдельно по seed;
измерить оба одиночных и совместное вмешательство, повторить на B и C.
Interaction = delta_pair-delta_first-delta_second, с правильным знаком:
отрицательный — совместное улучшение сильнее суммы, положительный — слабее.
Это ограниченная проверка композиции, не exhaustive28x28 search.

Teacher rescue использует больше attention и teacher activations. Даже крупное
улучшение не доказывает существование столь же хорошего deployable Mamba treatment.
Отрицательная replacement-проба не оправдывает слой: suffix мог научиться
компенсировать именно его ошибку. Указывать это в интерпретации таблицы.

## 6. Этап III: что происходит внутри рекуррентной динамики

На тех же16 A окнах, во всех24 Mamba:

- raw/transformed dt и decay; alpha=exp(decay*dt), trapezoid sigmoid,
  gamma и beta, gate/value/B/C RMS и extrema;
- state RMS/max по времени, отношение state injection к carry и final readout;
  skip contribution отдельно, cosine skip/state output и их взаимная компенсация;
- по MIMO rank: энергия до rank reduction и covariance/participation ratio;
  cancellation ratio суммы относительно суммы norms. Это utilisation proxy,
  а не доказательство числа семантически полезных каналов;
- angle increment pi*angle_step*dt, cumulative phase, finite flags;
- доли alpha<1e-3,>0.99,>1 и gate |z|>10, распределения p01/p50/p99.
  Пороги описательные. Decay/alpha trace характеризует данную траекторию,
  не измеряет истинную информационную ёмкость модели.

Сравнить plateau111 и собственный ONPOLICY на одинаковом реальном входе111:
защищённые колонки должны иметь идентичные weights, а input-conditioned traces
могут различаться при сравнении естественных полных streams.
Не путать изменение входа с обновлением protected weight.

Для двух выбранных teacher-mixer слоёв из этапаII: diagnostic state reset каждые
32/128 tokens, с явным reset ssm/previous_k/previous_v/angle, baseline normal carry.
Сравнение на A/B, последняя32-target область окна; C только зафиксированные IDs.
Reset вмешивается одновременно в память и фазу: это чувствительность к данному
вмешательству, а не чистый тест «память плохая». Без нормального carry parity
этот модуль считается invalid.

Дополнительная context-проба:32/64/128/256 input tokens с одним концом окна,
считать одни и те же последние31 next-token targets у student и Qwen.
Короткие варианты перезапускают state и positions; сохранять exact token IDs.
Сравнение показывает доступную пользу контекста до256, не long-context способность.

## 7. Этап IV: может ли оптимизатор двигать качество в нужную сторону?

В plateau checkpoint есть Adam. Его читаем, не сбрасываем и не продвигаем.
Градиенты CE по trainable coordinates на8 T-new окнах и8 A окнах, при одних
и тех же weights; gradients по FP32 effective kernels отдельно, включая frozen.
Сначала усреднить unclipped gradients внутри каждой панели. Сохранить и
индивидуальные gradients statistics, чтобы cancellation не выдать за нули.

По layer/component и всей модели: стабильная raw norm, norm/sqrt(n), zero/finite,
train/validation gradient cosine, effective-vs-coordinate discrepancy,
clip retained fraction, Adam moments norms. Effective gradient frozen группы
не равен доступному направлению optimizer и не доказывает нужду её открывать.

Из ОДНОГО averaged T-new gradient и существующего Adam state получить shadow
proposal как в111 (LR3e-4, stable global clip1, прежние betas/eps).
Это диагностический averaged batch8 proposal, НЕ следующий реальный batch1 step111.
Сравнить Adam направление с тем же clipped gradient без moments:
gradient direction нормируется до той же coordinate L2 длины, что Adam update.
Линии alpha0/0.1/1 для обоих направлений; оценка T-new/A, затем B/C без
повторного выбора направления. Итого6 включая дублированные identity.
Показывать linear prediction g dot delta и фактическую paired delta CE.

Записать proposed coordinate update и effective BF16 kernel changed fraction,
delta norm/max по компонентам. Если после BF16 projection направления становятся
нулевыми — это измеренный quantization stall данного proposal, не универсальный
диагноз. Если train улучшается, а A/B ухудшаются — локальный mismatch направления.
Если Adam хуже equal-length gradient, это сигнал проверить moments дальше,
а не доказательство пользы их сброса в длительном обучении.
Ни один shadow update не принимается в основной пациент; after tree/Adam digest
должны совпасть с before.

## 8. Этап V: проблема представления перед выдачей ответа?

Основная модель, input embeddings и tied vocabulary matrix полностью frozen.
Диагностическая поправка ставится ПОСЛЕ final RMSNorm и только на output path;
она не изменяет input embeddings. Именно это отделяет несовместимость readout
от разрушения входного словаря. Сохранять название «readout alignment probe».

Контроли: identity и temperature logits/T, T в{0.5,1,2,4,8}, выбор одного T
по mean A NLL двух seeds. Это проверка калибровки уверенности, не полноты знаний.

Обучаемый probe: h'=h+hAB, A2048x32,B32x2048,131072 FP32 params/seed.
A Normal(0,0.01), B0; init seeds114123/114456, identity при старте.
Adam LR1e-3,128 warmup, betas0.9/0.999, eps1e-8, stable global clip1, decay0,
batch1/context256, true-token CE, FIT4096 уникальных окон/seed.
Endpoints1024/4096; выбор ОДНОГО общего endpoint по средней A NLL двух seeds.
При failure любого seed selection не объявлять положительным двухseed результатом.
Head logits canonical BF16 vocabulary values; A/B FP32 masters, BF16 effective
forward, exact init parity. Полное vocabulary-matmul учитывать в compute.

Cached final-normalized features из frozen пациента разрешены с shape/token/tree
SHA и storage accounting; teacher labels/logits для обучения не используются.
В RAM максимум один seed cache; logits и полные hidden всех слоёв не сохранять.
На B проверить identity/temperature/selected rank32; на C один финальный проход
каждого. FIT losses/validation curves, updates/targets/input tokens публикуются.
Общий FIT budget2,097,152 input tokens /2,088,960 CE targets,8192 probe updates.
Это цена диагностической подгонки, а не main recovery tokens.

Если небольшой readout probe улучшает закрытые Wiki И PG19, это свидетельство,
что часть разрыва доступна исправлению на выходе без изменения decoder.
Это не доказательство сохранения всех знаний и не чистое восстановление Qwen.
Если не помогает, конкретный rank32/LR/budget мог быть недостаточен: нельзя
объявлять, что информация безвозвратно потеряна или Mamba лишена ёмкости.
Новая точка постановки и plateau source отличаются от прежних head gains094.

## 9. Этап VI: численный фон и контроли достоверности

Для всех механизмов:

- current->copy-current identity и alpha0 в одном executable: error<=1e-4;
  BF16 background относительно canonical пути отдельно, max per-window<=0.01,
  иначе invalid/stop. См. pre-TPU amendment в начале;
- hooked forward с tracing OFF/ON должен воспроизводить canonical unhooked
  output до вмешательства. Собирать не более одного layer trace одновременно;
- expected-null rollback для frozen blocks, начальная readout correction0;
- hashes patient parameters/Adam до и после, отсутствие accepted updates;
- BF16 vs FP32-forward на первых2 A окнах с одинаковыми effective values,
  включая embedding/readout и все MIMO factors. Это другая математика, отдельная
  NLL, не замена canonical reference. Если HBM не хватает — unavailable;
- FP32-vs-BF16 эффект сравнить с lowering floor; на2 окнах нет global precision
  заключения. NaN/Inf не заменять нулями; operational/numerical failure разделить.

## 10. Primary outcome и правила вывода

Primary outcome — reproducible intervention table и локализация bottleneck,
а не обязательный gate «выиграть NLL». Отрицательное обследование валидно.
Модули могут завершаться отдельно; отчёт обязан показывать missing/invalid.

Рабочий критерий PRIORITY_SIGNAL для каждого выбранного типа вмешательства:
на B и обоих C доменах mean delta NLL<=-0.1 у ОБОИХ seeds относительно своего
current baseline, identity checks passed. Имена выбранных слоёв могут различаться
по seeds — это проверка правила локализации, не одного универсального индекса.
Кроме средних показать per-window дельты/долю улучшений и чувствительность к
кластерам соседних8 окон. Cluster bootstrap только описательный; не выдавать
две model seeds или tokens за достаточную статистику универсальности.
Сигнал в одной seed/домене называется CONDITIONAL_SIGNAL, не подтверждённой причиной.
Пороги служат выбору направления следующего эксперимента, не заявлению superiority.

| Наблюдение | Следующий адресный вопрос | Что ещё НЕ доказано |
|---|---|---|
| Teacher mixer rescue одного участка устойчиво помогает | локальное восстановление функции на реальных входах | deployable Mamba может повторить rescue |
| Boundary rescue помогает, mixer rescue слаб | распределённый input/representation shift и suffix совместимость | единственный виноватый слой |
| Rollback помогает, teacher rescue не требуется | вредное отклонение конкретных updates | длительный freeze/rollback будет лучше |
| Совместный rescue имеет большой interaction | коадаптация блоков | pairwise обучение решит задачу;085–088 уже ограничили этот вывод |
| Readout alignment переносится на C обоих доменов | выходная совместимость/ограниченная адаптация readout | все знания сохранены |
| Temperature помогает почти как rank32 | калибровка уверенности | архитектурная ошибка устранена |
| Adam shadow хуже equal-length gradient | состояние optimizer/направление update | нужно немедленно сбрасывать Adam |
| Ненулевые coordinates почти не меняют BF16 функцию | effective precision/materialization | FP32 длительное recovery улучшит качество |
| Train-seen низкий loss, свежие панели высокие | генерализация и корпус | решение — просто ещё train tokens |
| Ни одна проверка не помогает | непроверенные distributed/capacity/objective механизмы | Mamba3 принципиально слабее attention |

Выводы раскладывать: наблюдаемая ассоциация -> контролируемое локальное
вмешательство -> независимая проверка -> пока непроверенное лечение.
EXP114 не подтверждает 14B, MLA, inference speed или initializer compute advantage.

## 11. Девять часов и сохранение

Default max-wall-hours8.8 ВКЛЮЧАЯ restore/compile/diagnostics/save; reserve60min
внутри, внешний setup allowance12min. Диагностическое состояние невелико,
поэтому reserve отличается от90min больших full-decoder111–113 states.
TARGET, не измеренная гарантия: до1h load/parity,2h sweep+verification,
1.5h dynamics/context,1h optimizer/precision,1.5h FIT/readout,0.8h final C/save.
Реальный ETA определяется по warm compile и throughput каждого модуля.
Не растягивать законченные проверки до9h; при большом времени полный план
продолжается той же ячейкой, без сокращения panels после просмотра результатов.

Порядок обязательных модулей: source/data/parity -> I -> II A/B -> V FIT/A/B
-> IV -> III -> VI -> фиксирование choices -> C всех готовых модулей.
При deadline сохранить task cursor и statuses. Реализация начинает C после всех
предшествующих модулей и durable фиксации choices; partial продолжается с cursor.
Не сравнивать incomplete module с completed как FAIL.
До выполнения кандидата оценить стоимость целиком на полном prespecified panel.
Не обрезать дорогое вмешательство по «плохому» loss и не продолжать только winner.

HF namespace `experiments/exp114-plateau-forensics`, stem
`extent-m3q-plateau-114`. max-new-hf-gb1 на вызов: только JSON/CSV, небольшие probe
coordinates/Adam/cursors, source manifests и provenance. Пациенты не изменяются
и повторно в HF не загружаются. Не выгружать feature cache, teacher logits или
новые полные16GB checkpoint copies. Старые HF artifacts не удалять.

Resumable results после каждого завершённого intervention/candidate;
readout params/Adam/data cursor каждые256 updates, remote bundled не чаще10min
и в конце. При kill возможен откат к durable probe cursor; все repeated costs
сохраняются. Проверять SHA каждого restored chunk и patient model provenance.
Telegram started/module completed/deadline/finished-saved/failed/pending с явным
completed_modules и next task. «Finished» означает сохранённый отчёт, а не PASS.

## 12. Артефакты и обязательная подготовка runner

Результат: machine-readable полный JSON и короткая сводка с разделением
MEASURED/DERIVED/HYPOTHESIS/UNAVAILABLE, sources/data/code hashes,
per-module engineering wall/forward/backward/FIT costs, pending/durable.
Таблицы: layer_profile, intervention_matrix, component_dynamics,
optimizer_shadow, readout_fit, final_verification. Статические научные графики:
layer/subcomponent heatmap; A/B/C paired rescue deltas; context/position error;
raw-coordinate/effective-BF16 update; readout learning curve и quality gap.
Не сглаживать spikes, не объединять seeds и не строить причинный waterfall
из неаддитивных intervention deltas.

Локальные preflight перед ячейкой:
1. Малый compiled hooked/unhooked forward и teacher-mixer/boundary/state reset,
   точные scopes, identities, tied head и causal target indexing.
2. Запрет мутации пациента, layout/sharding8 virtual CPU checks.
3. Подлинный patient/source manifest pin, binary-source existence и data SHA/capacity.
4. Readout FIT update, checkpoint/resume точного следующего шага и cache SHA.
5. Aggregation missing/failed, deterministic A-only selection и запрет C leakage.
6. Budget/429/quota/pending/durable/notification mocks.
7. Compiler-memory guard и последовательная residency; реальные1.7B TPU HBM,
   throughput и пригодность hooks локальные tiny tests НЕ подтверждают.

Новый runner независимый: `scripts/m3q_plateau_forensics_114_campaign.py`.
Старые111–113 contracts/runner не изменены:112 уже работает,
113 продолжает свой registered endpoint.
Запуск только import main в notebook kernel, без второго JAX subprocess.

Локальная проверка реализации:18 tests passed на8 virtual CPU devices, включая
legacy efficiency tests и весь tiny диагностический цикл с cloud resume.
CSV и12 файлов экспорта графиков проверены на synthetic tiny JSON (не результаты
исследования); --plan-only проверяет source/data/code contract без модели.

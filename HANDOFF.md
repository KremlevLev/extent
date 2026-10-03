# Extent / Mamba-3 in the Qwen: передача проекта в новый чат

**Обновление 3 октября 2026 после передачи:** EXP-097 завершён на TPU;
зарегистрированный диагностический gate PASS. Полный JSON и разбор:
`results/EXP-097-numerical-stability.json`, `results/EXP-097-completed-summary.md`.
Четыре NAIVE-ветви остановились из-за переполнения нормы при конечных градиентах;
восемь SAFE завершены. Protected rank32 CE при LR3e-4 — перспективный кандидат,
не конечный подтверждённый метод. Пользователь разрешил три новые кампании
EXP-098–100 под три отдельных восьмичасовых TPU v5e8-сессии:
`results/EXP-098-100-three-account-protocol.md` и отдельные протоколы.
Ниже исходная передача сохранена исторически: её «у EXP-097 ещё нет результатов»
уже перекрыто этим обновлением и последними записями технического отчёта.

Состояние на 3 октября 2026. Это инструкция для нового исследовательского/кодового
ассистента, а не статья и не новый результат эксперимента.
Последний кодовый коммит перед созданием этой инструкции:
`8b2eef3 feat: diagnose gradient norm overflow in EXP-097`.

## 0. Что пользователь просит от нового ассистента

Продолжить уже существующий исследовательский проект Extent. Не начинать заново,
не выдавать его за готовую модель и не повторять десятки закрытых экспериментов.
Сначала прочитать перечисленные ниже файлы, понять результаты и текущий протокол,
затем помогать с анализом новых результатов и реализацией следующих экспериментов.

Пользователь делает эксперименты на TPU в Kaggle, присылает небольшие сводки и
полные JSON/текстовые логи. Ассистент пишет и проверяет код локально, документирует
результаты, коммитит и отправляет изменения в приватный GitHub. Локально доступного
TPU у ассистента нет: CPU-проверку нельзя называть проверкой на реальном TPU.

Язык общения — русский, простой и прямой. Объяснять, что проверяем и почему это
может помочь, без лавины английских терминов и без чрезмерно оптимистичных обещаний.
Пользователь устал от двух месяцев исследований и мелких запусков. Это не повод
приукрашивать отрицательные результаты, но важно показывать конкретные выводы.

## 1. Где всё хранится

- Приватный код: `https://github.com/KremlevLev/extent`, основная ветка `main`.
- HF dataset: `https://huggingface.co/datasets/levkremlev888/extent`.
  Он публичный; там лежат большие состояния, чекпоинты, манифесты и сводки.
- Основной научный журнал: `extent technical report.md` в корне репозитория.
- Компактные протоколы, разборы и зарегистрированные критерии: `results/`.
- Основной актуальный Python package: `extent/`; запуски: `scripts/`; тесты: `tests/`.
- README — полезная карта проекта, но некоторые слова «next» и верхняя сводка
  устарели. Последние записи журнала и отдельные протоколы важнее.
- `kaggle operations.md` — эксплуатационная документация, не текст статьи.
  Раннее упоминание 10 часов не отменяет текущего требования пользователя:
  для EXP-098–100 рассчитывать на 8 часов одного вызова, включая 45 минут
  запаса на сохранение; учитывать время верхних ячеек отдельно.
- Старое имя проекта — `singularity`; новое — `extent`. Старые логи с прежним именем
  относятся к тому же проекту. Не переименовывать обратно.
- На старом ноутбуке checkout был в
  `C:\Users\Utest\Documents\Codex\2026-08-15\referenced-chatgpt-conversation-this-is-an\extent`.
  На новом пути будут другими; никогда не зашивать старый путь в код.

Старые пользовательские вложения из Downloads не обязательно имеются на новом ПК
или в Git. Не утверждать, что прочитал отсутствующий исходный JSON. Репозиторий
содержит документированные разборы и хеши; HF — часть исходных артефактов.
Секретов в этом документе нет и быть не должно.

## 2. Как читать репозиторий, чтобы восстановить контекст

### Первый проход: цель и актуальное состояние

1. Прочитать этот `HANDOFF.md` полностью.
2. Проверить `git status`, ветку, remote и последние коммиты. Не стирать чужие изменения.
3. Прочитать начало `extent technical report.md`: цель, точные источники,
   архитектуру, границы утверждений и правила фиксации результатов.
4. Прочитать `results/EXP-094-096-first-results.md` полностью, включая full-file addendum.
5. Прочитать `results/EXP-094-096-parallel-recovery-protocol.md`.
6. Прочитать `results/EXP-097-numerical-stability-protocol.md` полностью.
7. Прочитать в техрепорте EXP-061, EXP-062/062B, EXP-064, EXP-070–072,
   затем последние разделы EXP-089–097. Эти записи объясняют текущие решения.

### Второй проход: история механизма

Прочитать существующие `results/*summary.md` и `results/*protocol.md` для:

- EXP-071/072 — последовательное восстановление на входах уже изменённой модели;
- EXP-081/082 — принятие локальных обновлений по поведению собранной модели и провал репликации;
- EXP-085/086/087/088 — парные/совместные предложения и downstream-чувствительность;
- EXP-089/090 — длинный горизонт и заморозка dt;
- EXP-091/092/093 — защита скопированных весов, глобальная интерполяция, коэффициенты ветвей.

Использовать `rg --files results` для точных имён. Не выдумывать название файла,
если оно отличается. Для полной хронологии прочитать весь журнал по частям;
его ранняя дата «Last updated» устарела, свежие записи находятся в конце.

### Третий проход: текущая реализация

Перед изменением EXP-097 прочитать:

- `scripts/m3q_numerical_stability_campaign.py` — новый эксперимент;
- `scripts/m3q_subspace_engine.py` — общий resumable двигатель EXP-094–097;
- `extent/stable_gradient_clip.py` — устойчивое вычисление нормы/клиппинга;
- `extent/recovery_subspace.py` — обучаемые поправки к замороженной модели;
- `extent/hf_artifact_sync.py`, `extent/hf_checkpoint_sync.py` — отправка на HF;
- `extent/campaign_checkpoint.py`, `extent/endpoint_checkpoint.py` — состояния и provenance;
- `extent/notifications.py` — Telegram;
- `tests/test_stable_gradient_clip.py`, `tests/test_recovery_subspace.py`,
  `tests/test_subspace_campaign_workflow.py` и относящиеся к загрузке/сохранению тесты.

Для архитектуры и восстановления:

- `extent/qwen_source.py`, `extent/qwen3_teacher.py`, `extent/qwen3_parity.py`;
- `extent/config.py`, `extent/model.py`, `extent/layers/`;
- `extent/mamba3_transplant.py`, `extent/initialization.py`, `extent/hybrid_transplant.py`;
- `extent/rorope_bkv_conversion.py`, `extent/rorope.py`, `extent/mla_conversion.py`;
- `extent/sequential_recovery.py`, `extent/full_model_distillation.py`;
- `extent/sharding.py`, `extent/optimizer.py`, `extent/train_step.py`;
- `extent/teacher_activation_cache.py`, `extent/offline_distillation.py`,
  `extent/streamed_lm_eval.py`, `extent/artifact_summary.py`.

В корне есть более ранние `model/`, `modeling/`, `training/` и `config1.py`.
Не считать их автоматически основной реализацией: проследить реальные imports
актуальных campaign scripts. Legacy `tests/test_qwen_base.py` скачивает большую
модель/запускает работу на import и исключён из обычного offline pytest.

## 3. Научная задача — не просто собрать гибрид

Итоговая цель: превратить предобученную Qwen3 в преимущественно Mamba-3 MIMO модель
без обучения с нуля, сохранить знания и восстановить качество с ограниченным
числом токенов/TPU-часов. Сохранённое GQA-внимание потом сжать в MLA-подобный cache.
Нужны реальные преимущества по качеству/памяти/скорости, а не только конфиг.

Пользователь хочет сильную научную работу, arXiv и потенциально ICLR-подобную
конференцию. Рабочее название метода — «Mamba-3 in the Qwen», сокращение M3Q.
Сильного подтверждённого конечного метода пока нет. Не обещать принятие статьи.

Научный вклад должен быть в переносе и восстановлении архитектуры: новый механизм,
объяснение, сильные контролируемые сравнения, воспроизводимость и экономия compute.
Само сочетание SSM и attention не является достаточной новизной.

После восстановления пользователь хочет reasoning SFT, возможно на примерах
рассуждений Claude, и большое контекстное окно. Это будущая отдельная стадия;
не смешивать улучшение от SFT с восстановлением после замены архитектуры.

## 4. Точные модели и архитектуры

### 14B — окончательная целевая модель

Источник именно `Qwen/Qwen3-14B`, НЕ Qwen2.5, НЕ автоматически `Qwen3-14B-Base`.
Immutable revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`.

- 40 слоёв, hidden 5120, intermediate 17408;
- 40 Q heads, 8 KV heads, head_dim 128, per-head Q/K RMSNorm;
- vocab 151936, RoPE theta 1000000, опубликованные max positions 40960;
- untied embeddings; источник содержит 443 тензора;
- teacher mapping: 14,768,307,200 параметров, coverage 443/443;
- direct mapping: 203 тензора / 12,251,714,560 параметров.

Цель строго 34 Mamba + 6 attention = 85%/15%.
Сохранённое attention на zero-based `[5,12,19,25,32,39]`.
Пользователь прямо отверг 25% attention как полноразмерный baseline.
Ошибочные ранние 48-слойные схемы/12 attention не являются текущим контрактом.

Production attention выбран по EXP-019: RoRoPE fold1 + BKV activation PCA,
rank448 latent cache + один128-элементный rotary key. На сохранённом attention
cache сокращается с2048 до576 элементов/token, то есть71.875%.
Для шести слоёв:12288→3456 элементов/token. Это не измерение полного HBM,
не отношение всей модели и не обещание ускорения decode.
Ранний MaxText-style MLA остался scaffold/ablation, не production attention.
Extent не нужно объявлять готовым drop-in upstream MaxText decoder.

Production Mamba первоначально выбран `INIT-K-balanced-qkvo-lift`.
Это инженерный выбранный builder, а не утверждение о победе метода в full-model recovery.

### 1.7B — текущий исследовательский стенд

Источник `Qwen/Qwen3-1.7B-Base` revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.

- 28 слоёв, hidden2048, intermediate6144;
- 16 Q heads,8 KV heads,head_dim128, vocab151936;
- tied embeddings, max positions32768;310 source tensors;
- hybrid:24 Mamba +4 неизменённых GQA на `[6,13,20,27]`.

Это около85.7% Mamba, дискретная малая аналогия целевых85%.
Сохранённое внимание сейчас GQA намеренно: изолируем перенос Mamba от MLA.
Не добавлять MLA в текущий recovery-протокол незаметно.
14B — проверка масштабирования лучшего метода позже, не каждодневный стенд.

## 5. Что уже действительно проверено

### Корректность и инженерная выполнимость

- JAX Qwen decoder layer сверялся с official PyTorch; layer0 FP32 на2T4:
  max_abs≈4.05e-6, relative_l2≈9.86e-8. Это layer parity, не full-model quality.
- Mamba-3 JAX scan сверялся с независимой Torch reference и shapes/names.
  Это корректность конкретной reference-реализации, не доказательство быстрого kernel.
- Малые BF16 sharded train steps прошли на TPU v5e1 и v5e8 с конечными градиентами.
- EXP-062 собрал полный14.766B hybrid:34 Mamba/6 выбранных attention,
  443 source tensors,203 direct mappings, реальный sharded Lion, конечный forward8/32/128.
- Реальные weights и Lion momentum по3.4513GiB/device, вместе6.9026GiB/device.
  Не включены gradients, activations, temporaries, executable buffers.
- EXP-062B backward context8 прошёл: конечные loss/grads, но raw norm≈8.28e8.
  Это numerical/HBM feasibility, не восстановленная модель, не качественное обучение.

Reference recurrence — `lax.scan`, FP32 recurrent state. Для убедительных speed claims
нужны оптимизированные kernels (например Pallas), корректный streaming decode,
сопоставимый Transformer baseline и реальные измерения.

### Перенос матриц и exact lift

Прямое копирование Q/K/V/O в Mamba не оказалось универсально хорошим.
Изучали prior reuse, scale matching, readout calibration, разные MIMO allocations,
linear/complex bridge, RoPE преобразования и несколько objectives.

Exact balanced lift раскладывает исходную линейную операцию по MIMO-каналам,
сохраняя её в соответствующем подвыражении. Он НЕ делает softmax attention
математически эквивалентным всей Mamba-3 recurrence.

EXP-056–061 дал реальный позитивный результат на локальном восстановлении и
частичных композициях. EXP-061:4/8/12 production-aligned замен,3paired seeds;
exact выигрывает3/3 seeds на каждой глубине; mean exact-random NLL
−0.206008/−0.122781/−0.164937. Все upper bootstrap bounds ниже нуля.
На12 слоях это≈3.12% total-NLL improvement и14.51% уменьшения excess NLL
над исходным Qwen относительно random. Не говорить «модель стала лучше на15%»
без указания метрики, знаменателя и условий.

Hidden-state L2 мог становиться хуже, хотя NLL улучшался. Локальный MSE и NLL
собранной модели не взаимозаменяемые конечные критерии.

### Почему локальная победа не перенеслась на полную модель

В full-model Qwen3-1.7B сравнениях random мог обгонять exact.
Подтверждение длинным горизонтом EXP-089: в завершённой paired seed exact
проиграл random после25.17M training tokens; зарегистрированный gate стал невозможен.
Нельзя описывать ранние layerwise успехи как подтверждённую full-model superiority.

Ключевой механизм: ранние замены меняют residual streams для последующих слоёв.
Блок, обученный на входах чистого Qwen, получает другие входы в собранном hybrid.
Это проверяли не только рассуждением, но matched-input/композиционными опытами.

### Последовательное восстановление — важный подтверждённый эффект

EXP-071: последовательно добавляем Mamba, каждый новый блок учим на входах,
которые реально создаёт уже изменённый prefix. ONPOLICY лучше teacher-input обучения.

EXP-072-v2 подтвердил на свежем тексте, всех24 Mamba positions и большем бюджете:
ONPOLICY final NLL10.545254/10.266002 против22.346362/22.037049 у TEACHER,
дваpaired seeds123/456, gate проходит. Разница≈−11.8NLL в каждой seed.
Снижение total NLL≈53%, excess NLL≈61–62% относительно TEACHER-условия.
Исходный Qwen≈3.0675 на том же оценочном наборе: восстановление далеко не полное.
Особенно заметен провал последних замен:depth20≈6.9–7.0 →depth24≈10.3–10.5.

Это улучшение процедуры восстановления относительно matched TEACHER, НЕ
победа над random при равном полном compute. Затраты на подготовку warm start
должны входить в окончательное сравнение метода.

Completed EXP-072-v2 ONPOLICY checkpoints — текущие исходные модели поздних опытов.
Восстанавливать их непосредственно с HF по проверенным контрактам и hashes.
Не просить пользователя заново обучать всё или передавать большие файлы вручную.

### Что не стало надёжным решением после EXP-072

- EXP-073–080: обычное совместное обучение, варианты защиты backbone/оптимизатора,
  depth segments и второй coordinate sweep не дали надёжного восстановления.
- EXP-081: assembled-model trust interpolation улучшил start в обеих seeds,
  но эффект seed456 мал. EXP-082 с8replications не подтвердил matched-control gate.
  Отбор на WikiText не гарантировал улучшение PG-19.
- EXP-083–088: dual-domain/robust selection, paired lookahead/joint pair/downstream
  не превратились в подтверждённый конечный метод. Прочитать отдельные summaries;
  не свести все опыты к «вообще ничего не работает» и не выбрать лучший seed.
- EXP-090: ранняя заморозка dt до8.39M tokens; endpoint25.17M.
  Seed123 небольшое улучшение excess NLL−0.224476/KL−0.282565;
  seed456 ухудшение+2.374740/+3.372097. Two-seed gate FAIL.
- EXP-091: ONPOLICY warm start + post-Lion backbone updates×0.03 противPLAIN.
  Seed123 уже опроверг gate; оба варианта ухудшались от общего старта.
  Не тратить сессию только на завершение mathematically impossible gate.
- EXP-092: straight whole-model interpolation черезEXP-091 направления.
  Все3available branches выбралиalpha0. Малые положительные alpha тоже ухудшалиNLL.
- EXP-093: обучаем только bounded output gains, без изменения остального.
  Per-layer выигрывает global gain в обеихseeds, но ухудшает start seed456.
  Gate FAIL. Start/пер-layer NLL13.261727→12.451787 и9.161675→10.170573.

Не сравнивать числа между экспериментами без проверки одинаковых текстов,
offsets, checkpoints и того, absolute это NLL или excess NLL.

## 6. Последние законченные EXP-094/095/096

Три параллельных кампании проверяли не новые инициализации, а маленькие обучаемые
FP32 поправки к уже собранным замороженным BF16 ONPOLICY hybrids.
База: 24 Mamba + 4 GQA; seeds 123/456; batch 1; context 256; Adam 3e-4; clip 1.
Поправки начинают с нулевого изменения функции. Low-rank A×B прибавляется к kernel
и по контракту может быть свёрнут для inference. Реальный экспорт надо проверять.

- EXP-094: OUT rank 8, INOUT rank 8, MLP readout rank 8, head gains; всё CE.
- EXP-095: одинаковый INOUT rank 8, цели CE / teacher KD / self-anchor / delta-then-KD.
- EXP-096: INOUT CE rank 8 / rank 32 / rank 64 / protected rank 32.

Protected запрещает поправки к in_proj columns, производящим dt/raw_a/trap/angle.
Но входы от предыдущих слоёв меняются: динамика не целиком «заморожена».

Общие тексты и порядок; 8192 steps на траекторию; горизонты 2048→4096→8192.
Validation диагностический; locked test только на final и start.
Сводки строятся автоматически; полный JSON остаётся для аудита.

Результаты:

- Shared INOUT-rank8 CE повторён byte-equivalent в трёх кампаниях. Это дублированный
  контроль, не три независимых репликации. Seed 123 завершён: NLL 12.968006→7.349788.
  Seed 456 падает на attempted step 4228; last-good step 4227.
- OUT rank 8 падает в обеих seeds. MLP readout завершён, но NLL ≈19.87/19.14.
  Head gains дают смешанный слабый эффект.
- Teacher KD и self-anchor в seed 123 завершены с NLL 7.983263/8.881876, хуже CE;
  в seed 456 падают. Delta-then-KD падает на attempted steps 91/44.
- Ordinary rank32 CE завершён в обеих seeds:
  NLL 12.968006→7.643549 и 12.276975→9.342761.
- Rank 64 падает в обеих seeds.
- Protected32 seed 123 падает на attempted step 4025; seed 456 завершён с NLL
  7.315458, лучше ordinary32 seed 456 (9.342761). Это выигрыш одной seed, не PASS.
- Все три кампании имеют branch_failure и scientific_gate=false.

Новая реальная зацепка: адаптация небольшого input/output subspace способна
улучшать whole-model NLL в обеих seeds при rank 32. Но это не полное восстановление,
не победа инициализации и не устойчивая траектория. Даже успешные rank32 validation
curves сильно скачут; последние нормы ≈1.66e7/4.15e10.

Полные JSON прочитаны. Старый guard объединяет loss, градиенты, naive norm,
новые параметры и Adam state. Из FAIL нельзя узнать, какая величина испортилась.
Перед некоторыми failures norms ≈1.64e17/6.37e17. Это повод проверить переполнение
sum-of-squares нормы, НЕ доказательство этой причины. В FP32 квадрат большого
конечного градиента может переполниться при ещё конечных gradient elements.

Полный разбор и SHA256: `results/EXP-094-096-first-results.md`.

## 7. Текущий следующий эксперимент: EXP-097

EXP-097 реализован, локально проверен, закоммичен и отправлен в GitHub.
В передаваемом контексте реальных TPU-результатов ещё НЕТ.
Если пользователь пришлёт новую сводку, сначала установить её свежесть и статус.

Цель: отделить переполнение вычисления нормы от настоящего forward/backward/Adam
срыва и проверить устойчивый clipping / пониженный LR. Это не новый initializer.

12 траекторий: seeds 123/456 × шесть arms:

| Arm | Rank | Поправка к dynamics projections | Norm/clip | LR |
|---|---:|---|---|---:|
| NAIVE-R8 | 8 | разрешена | старый sum-of-squares | 3e-4 |
| SAFE-R8 | 8 | разрешена | max-scaled | 3e-4 |
| SAFE-R8-LOWLR | 8 | разрешена | max-scaled | 3e-5 |
| NAIVE-PROTECTED-R32 | 32 | запрещена | старый | 3e-4 |
| SAFE-PROTECTED-R32 | 32 | запрещена | max-scaled | 3e-4 |
| SAFE-PROTECTED-R32-LOWLR | 32 | запрещена | max-scaled | 3e-5 |

Все стартуют от неизменённых EXP-072-v2 ONPOLICY. Не использовать optimizer states
проваленных EXP-094–096. Base BF16 frozen; coordinates/moments FP32;
CE; batch 1; sequence 256; clip 1.

Оценка повторяет EXP-094–096: это reused evaluation, не fresh confirmation.
Train WikiText103 offset 8,388,608, 8192 windows;
validation offset 49,152 / 32 windows; test offset 32,768 / 64 windows.
Горизонты 2048/4096/8192; validation каждые 1024; test на start и final 8192.
Максимум 98,304 student steps / 25,165,824 input tokens.

Stable norm: m=max|g|, s=sqrt(sum((g/m)^2)); clipping через (g/m)/s без g²
и без обязательного формирования m×s. Zero case обработан. NaN/Inf не заменяются
nan_to_num и не маскируются. Авторитетная диагностика — log10(norm);
display norm может насыщаться на 1e38, это явно обозначено.

Каждый step вычисляет NAIVE и SAFE proposals на ОДНОМ gradient и ОДНОМ Adam state.
Неиспользуемый shadow proposal не обновляет optimizer. Отдельные trajectories сами
по себе не доказывают norm-overflow causality: rounding может раньше менять путь.
Norm-only event: finite forward/loss/gradient elements + nonfinite naive norm +
finite safe proposal.

Диагностика разделяет forward activations/logits, loss, gradient elements, norm,
current parameters/moments и proposal parameters/moments. Пишутся layer maxima и
coordinate gradient health. До первого norm-only update сохраняется event checkpoint;
при failure сохраняется last-good state. Исходный Qwen отдельно оценивается на том
же locked test для остаточного quality gap, не используется для выбора LR или steps.

Зарегистрированный gate: все 12 arms terminal; хотя бы один norm-only event;
SAFE-R8 final test NLL лучше start в обеих seeds. Controls могут numerically fail
как наблюдаемый диагностический исход, но это не successful recovery.
Protected и LOWLR — заранее заданные вторичные сравнения.

Default cap 8.25 часа, включая 30 минут резерва; это не обещание runtime.
Local atomic binary save каждые 256 steps; bundled HF не чаще одного commit / 10 min,
плюс final. HF 429 → cooldown 1 h; обучение локально продолжается.
Внешний VM kill может потерять unsynced tail.

HF namespace: `experiments/exp097-numerical-stability`.
Summary stem: `extent-m3q-numerical-stability`.

У общего engine есть optional hooks для EXP-097; legacy contracts EXP-094–096
сохранены. Legacy engine digest:
`c3e17745b12e2d1bbab225ead5db3db705343327c1caa5241e10a8512dfb36e6`.
Не менять его произвольно и не обходить resume checks.

Локально проверены нормы до 1e38, zero/nonfinite handling, matched proposal overflow,
compiled steps всех шести arms, восемь virtual CPU shards, binary optimizer resume,
event checkpoints и 429/interruption workflows. В предыдущем ответе отмечены
32 targeted CPU tests и шесть sharded step tests. Это НЕ real TPU throughput
и НЕ наблюдённый numerical win.

Последняя ячейка пользователя, когда setup уже выше:

```python
from scripts.m3q_numerical_stability_campaign import main as run_exp097
result = run_exp097([])
```

Попросить small summary + JSON, а не передачу checkpoint через пользовательский ПК.
При budget-partial повтор той же ячейки возобновляет совместимые HF states.

## 8. Правила Kaggle/Colab и сохранения

Основной accelerator: Kaggle TPU v5e8; mesh data/fsdp/tensor = 1/4/2.
Colab v5e1 подходит для single-device correctness, не для multi-device sharding
или полного 14B. На двух T4 старый BF16 compute давал NaN; FP32 compute проходил.
Использовать аппаратный auto dtype и не переносить GPU-specific failure на TPU
без проверки.

TPU эксклюзивен одному Python process. После jax.devices() в notebook kernel
не запускать `!python`, `!pytest` или subprocess с JAX: возникают busy TPU/PID/libtpu.
Campaign запускается через import main([]) в kernel. Pytest при необходимости —
pytest.main([...]) в том же процессе. Установка dependencies должна предшествовать
инициализации JAX; после его замены pip может потребоваться restart kernel.

Пользователь часто запускает Save Version и уходит. Не полагаться на интерактивные
переменные, ручное сохранение или живую сессию через несколько часов.
Реально теряется локальный /kaggle/working после окончания сессии.
Durable HF save отличается от существующего локального checkpoint.
Нельзя гарантировать callback при внешнем kill: описывать recovery granularity честно.

Ранее уже были ошибки: missing JSON keys; NumPy conversion внутри jit;
FrozenDict/dict mask mismatch; NumPy scalar JSON serialization; donation alias
teacher/student; недостаточный размер токенизированного корпуса; disk full; HF429.
Эти пути нужно проверять локально ДО многочасовой очереди TPU.

Большие Qwen shards/cache можно хранить в RAM-backed /dev/shm при достаточной RAM.
Не забивать примерно 20 GB working disk полным 14B checkpoint.
RAM budget проверяется; наличие 300 GB на одной сессии не универсальная гарантия.

HF uploads должны быть пакетными. Repository commit limit уже ломал запуски:
128 commits/hour. Не делать отдельный remote commit на каждый metric/JSON/step.
Несколько аккаунтов вместе нагружают один и тот же HF repository.

Секреты и провайдеры:

- EXTENT_DEPLOY_KEY_B64 — read-only Git deploy key в Base64;
- HF_TOKEN — авторизация HF;
- EXTENT_HF_CHECKPOINT_REPO — если используется соответствующим runner;
- TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID;
- Kaggle: UserSecretsClient; Colab: google.colab.userdata. Это разные механизмы.

Секреты не печатать и не коммитить. Telegram должен сообщать start, end,
budget-partial и caught failure. NOTIFIER_ENABLED позволяет отключить его.
Никакой notifier не запускается до фактического старта kernel в очереди.

Пользователь не хочет manual uploads больших файлов, Google Drive или перенос
checkpoint через ноутбук. Всё напрямую HF↔runtime. Скачать нужно только небольшие
summary/JSON. HF dataset накопил примерно 500 GB; пользователь хочет периодический
retention audit, а не ограничение использования HF. Ничего не удалять без отдельного
одобрения точных целей. Сохранять зависимости, важные endpoints, configs/hashes/raw
metrics. Удаление current files не обязательно освобождает исторические HF versions.

## 9. Как проектировать следующие эксперименты

Пользователь ждёт TPU часами и хочет 5–8 часов полезной работы в пределах девятичасовой
сессии, не отдельную 15-minute пробу. Разрешены несколько гипотез/arms в одной
кампании. Не тратить compute только ради длительности: нужны comparators,
balanced horizons и остановка при уже доказанном провале зарегистрированного gate.
Runtime estimate брать из measured throughput, не объявлять 8192 steps восьми часами.
Длинный научный run можно продолжать несколькими resumable sessions.

До запуска:

- plan-only и проверки без скачивания моделей;
- реальные compiled tiny steps, все arms и objectives;
- восемь virtual CPU devices для sharding, если возможно;
- serialization/resume model+optimizer+data cursor;
- aggregation с отсутствующими/failed/partial branches;
- cloud execution без интерактивных запросов и случайных переменных из прошлой ячейки;
- HF429, transient network error, auth failure, budget stop;
- token capacity guards, наличие checkpoint dependencies, memory/disk budget.

Не скрывать structural exception под «научным отрицательным результатом».
Не заменять missing metrics нулём ради завершения сводки.

До наблюдения outcomes нужен protocol в results: source pin, methods, paired seeds,
data hashes и offsets, train/validation/test domains, steps/tokens, optimizer,
primary metric/gate, secondary exploratory metrics, budget и resume contract.
Test не использовать для выбора alpha/LR/checkpoint.
Если test уже обсудили и принимали решения по нему, следующая confirmatory phase
требует нового test либо честной отметки reused evaluation.

Equal steps/tokens ≠ equal FLOPs. Учитывать layerwise preparation, offline cache
и teacher passes. Две seeds сами по себе не делают результат conference-level.
Нужны свежие домены, больше seeds, длинные контексты, сильные предыдущие методы
и сравнения при сопоставимом compute.

Related work: Mamba in the Llama, MOHAWK, MHA2MLA, SSM hybrids, обсуждавшиеся
Apple attention-to-Mamba и attention-transfer papers. Некоторые названия/даты
пришли от другой LLM: проверить originals и bibliography в техрепорте перед
ссылкой или утверждением новизны. Не говорить «linear attention = softmax = Mamba3»
без точных условий. Mamba-aware linear/complex/MIMO bridge уже пробовали;
новый вариант должен объяснять отличие от EXP-065/066 и предыдущих мостов.

После EXP-097 дальнейшее решение зависит от результата:

- Norm-only event подтверждён: оценить реальный эффект stable clipping,
  затем проверить перспективный рецепт на свежем наборе.
- Forward/gradient-element failure: работать с layer diagnostics;
  не обещать, что clipping всё исправит.
- LOWLR улучшил устойчивость: отдельно измерять progress и remaining teacher gap.
- Всё устойчиво, но качество плохое: диагностировать capacity/objective/input shift.
- До перехода на14B нужен устойчивый whole-model1.7B trend и random control
  при сопоставимых ПОЛНЫХ затратах, включая warm-start preparation.

Не обещать число токенов полного восстановления8B/14B: это не измерено.
Пользователь допускал3–6месяцев TPU;25–50B tokens были лишь предположением.
Без реальной скорости и recovery curves это не обоснованный прогноз.

## 10. Документирование и взаимодействие

Каждый результат добавлять в `extent technical report.md`: дата, источник,
hash при наличии, MEASURED/DERIVED/TARGET/HYPOTHESIS, решение и ограничения.
Отдельный компактный `results/EXP-XXX-...summary.md`; raw JSON оставлять аудируемым.
Агрегатор должен проверяемо вычислять выводы, а не просто выбрасывать поля.

Не менять gate постфактум ради PASS. Лучший intermediate не заменяет конечный
registered endpoint. Различать operational failure, numerical failure,
budget partial и scientific gate FAIL. Partial иногда уже делает gate невозможным;
тогда не обязательно тратить квоту на оставшиеся controls.

Перед новой гипотезой прочитать историю и проверить, не тестировали ли именно её.
Не предлагать пользователю заново собирать прошлые кеши, если есть durable HF source.

После изменений сообщать:

1. Что изменено и какую гипотезу проверяет.
2. Что проверено локально; что не проверено на TPU.
3. Название и hash коммита; успешен ли push.
4. Одну готовую финальную ячейку для Kaggle.
5. Какую небольшую сводку прислать.

Setup clone/install/secrets у пользователя обычно уже выше в ноутбуке.
Не повторять весь setup, когда он просит только эксперимент.
При сохраняющейся авторизации коммитить и отправлять код, протокол и журнал;
чистый Git и local commit не означают успешный push — проверять.

В прошлом были отдельные просьбы о backdated commits ради GitHub streak.
Это не научное требование и не повод искажать даты измерений или provenance.

## 11. Первый ответ нового чата

После чтения файлов коротко сверить понимание:

1. Цель — надёжный Qwen→Mamba3 recovery method; потом MLA,14B и efficiency.
2. Уже работало: exact lift в частичных композициях; ONPOLICY против TEACHER
   full-depth; recent rank32 corrections улучшают NLL, но нестабильны.
3. Не подтверждено: полное восстановление, whole-model initializer superiority,
   long context, inference speed, tokens-to-recovery для14B.
4. Текущий подготовленный код — EXP-097; запросить результат только если его нет.
5. Не предлагать снова random-vs-copy на одном слое, Qwen2.5 вместоQwen3,
   полгода обучения14B без малого подтверждения или только план без runner.

Уважать амбицию пользователя, оставаясь исследователем:
факты → гипотеза → контролируемый эксперимент → измерение → обновление вывода.

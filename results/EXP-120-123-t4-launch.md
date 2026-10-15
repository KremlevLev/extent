# EXP120–123: масштаб активаций и end-to-end восстановление

15 октября2026. PREPARED: новый streamed backward проверяется инженерными CPU
тестами; физического1.7B GPU training результата этих четырёх экспериментов нет.
Ноутбук `notebooks/extent-t4-long.ipynb`. Один EXP на отдельную Kaggle T4x2
сессию, обе GPU работают независимо. Интернет и прежние Secrets включены.
Никакого предварительного nvidia-smi/проверки марки GPU; LD_LIBRARY_PATH
сохраняется для доступа к библиотекам драйвера. Математические source controls
обязательны. Для новой сессии использовать свежий clone из первой ячейки.

## Почему именно эти проверки

EXP116–119 завершены, оригинальные пользовательские JSON архивированы как
EXP-116/117/118/119-session-1.json, SHA/статусы в EXP-116-119-session-1-analysis.json.
Patient111 воспроизвёлся: GPU7.26410 против TPU7.26461, per-window error<0.0043.
116: резкие скачки масштаба, особенно0/3/11/16/26; 24/24 detailed recurrence
traces rejected по зарегистрированному absolute guard, не интерпретировать их.
117: на одинаковых tail127 targets контекст128→1024 почти не меняет гибрид
(7.24384→7.24391), Qwen3.43932→2.51573; reset mean effects<0.001.
118: compute FP32 mean NLL effects<0.0005, FP16 nonfinite. Это inference-only
intervention со stored BF16 weights/readout, не проверка FP32 retraining.
119: все18 fits1024 durable; локальная teacher-mixer цель на hybrid inputs не
дала substantial full NLL gain, layer26 treatments ухудшали NLL примерно7.2→16.1.

Protected rank32 + stable gradient-norm calculation/clipping полезен для
восстановления. Это **не доказательство**, что rank32 оптимален, и **не
нормализация активаций**. Причина плато пока неизвестна. Сейчас проверяем
causal effects output scale и возможность end-to-end adaptation, а не объявляем
huge activations окончательно установленной причиной.

## Объём и время

Общий лимит11.5h **от вызова runner**, включает загрузку и45min reserve.
Если setup уже занял>30min от12h Kaggle-сессии, уменьшить wall по остатку.
EXP120:461 условий ×64 A-windows =29,504 полных forward evaluations,
разделённых между двумя GPU, плюс проверки выбранных условий. Это сотни условий,
не повтор тех же8 windows десятки раз. В пределах одного условия A64 фиксированы.
На основании measured source replay116 (~134sec/64 windows/GPU) это существенно
больше прежнего короткого профиля; точная скорость нового engine ещё неизвестна.

EXP121–123: **65,536 full-model updates на GPU**, а не1024 local fits.
Это16,777,216 training input-token exposures/arm,65,536×255 targets; четыре
прохода16,384 окон. Плановый горизонт может требовать нескольких12h сессий.
Не обещать физическое завершение всего горизонта в одной сессии. Runner работает
до horizon/registered deadline/storage limit/failure, не заполняет квоту sleep.
Продолжение — та же ячейка. При completed без pending saves повтор не обучает заново.

Первые1/2/4/8/16 updates и затем каждые32 пишут health, после первого — фактические
seconds/VRAM; это позволит честно измерить скорость полного backward на T4.

## Общий пациент и данные

Source — immutable111 seed123 MAIN32768, pin114 revision
f839c2a898c3e9eff9907ea3cf1e022fe16f3e09. Source base digest, все chunk SHA,
manifest,64-window GPU/TPU source NLL gate<=0.03 сохраняются. Новый gain1/R32
hook проверяется против исходного **того же GPU** max per-window NLL<=0.01.
Нет ослабления старых accepted trace criteria или миграции старых experiments.

Data EXACTLY existing pinned panels: A validation122880/64, B180224/32,
C196608/64; FIT Wiki103train74973184/16384, sequence256. Все четыре SHA проверяются
против исходных114/115 manifests. A/B/C reused validation, не новый locked test.
FIT overlap115 (и первые128 окон119) явно маркирован. Same coprime permutation
`(step*7919+120123)%16384` для всех arms,4 epochs. Никакого выбора по B/C.
Single seed123 — исследовательский screening, winner проверять новыми seeds/data.

## EXP120 — causal scale sweep, без обучения

Умножение Mamba output перед residual addition на фиксированный gain. LayerNorm,
MLP, четыре GQA, веса самого Mamba неизменны. Только уменьшающие gains:
0.75/0.5/0.25/0.1/0.03/0.01/0.003/0.001/1e-4/1e-5/1e-6/1e-8/0.
По одному для каждого из24 Mamba layers и5 групп:
all, spikes[0,1,3,11,16,25,26], early<8, middle8..19, late>=20.
Также21 пара изspikes × gains0.5/0.1/0.01/0. Итого461 условий, gain1 identity
отдельным контролем. Порядок от mild к strong, broad round-robin coverage.

Каждый condition A64, paired delta относительно hook control. Каждый worker
выбирает своих top3 по A64, проверяет B32/C64, профиль слоёв и context128..1024.
Nonfinite condition сохраняется как invalid и не останавливает весь grid;
он не участвует в выборе top3. Memory/transport/source failures не скрываются.
Если sweep прерван work deadline, reserve используется для top3 среди **уже
измеренных**, coverage/partial явно сохранены; не заявлять optimality по всему
grid. Следующая сессия продолжает недостающие условия и обновляет selection.
Уменьшение активаций без улучшения NLL не успех. Полезно также ухудшение:
оно проверяет роль ранее выученных компенсаций. Same-target context127
сравнивается с source того же GPU. Больший контекст должен приносить пользу,
а не только уменьшить масштабы.

```python
from scripts.m3q_t4_scale_campaign import main as run_t4_long
result = run_t4_long([
    "--experiment", "120",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "2",
])
```

## EXP121 — научатся ли масштабы через итоговую NLL

GPU0 SCALE-CE: все веса111 frozen,24 trainable log-gains Mamba output.
Начальные gain1; gain=exp(clip(raw,-30,3)), положительный scalar на блок.
GPU1 SCALE-CE-PENALTY: те же24 scalars +0.01×mean squared positive
`log(RMS(gain*MambaOut))-stop_gradient(log(RMS(residual)))` по24 Mamba слоям.
Penalty не может улучшаться градиентом через искусственное увеличение denominator.
Mean берётся только по Mamba; GQA не штрафуется.

Цель — full-model next-token CE. Это24 параметра, но backward идёт через весь
decoder: local attention matching не используется. Сравниваем CE и CE+штраф:
если только штраф снижает RMS без улучшения NLL, считать гипотезу недостаточной.
Raw gain LR.01, Adam b1=.9/b2=.999 eps1e-8,256-update warmup, stable global clip1.

```python
from scripts.m3q_t4_scale_campaign import main as run_t4_long
result = run_t4_long([
    "--experiment", "121",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "2",
])
```

## EXP122 — rank32 control против rank32 + scales

GPU0 R32-CE: существующие trained111 rank32 INOUT coordinates открыты,
исходные bases frozen, protection dt/rawA/trap/angle columns сохранена.
GPU1 R32-SCALE-CE: те же coordinates +24 gains. Оба стартуют одной моделью,
одинаковыми данными/порядком/LR. LoRA LR3e-4, gains.01, warmup256.
**Свежий Adam в обоих arms**, source111 moments не сохраняются/не заявляются.
Это чистое сравнение добавления scales при одинаковом reset optimizer.
Параметров не rank256: намеренно сохраняем рабочий rank32 для контроля причины.

```python
from scripts.m3q_t4_scale_campaign import main as run_t4_long
result = run_t4_long([
    "--experiment", "122",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "2",
])
```

## EXP123 — end-to-end logit distillation вместо локальной цели119

Оба GPU: trained rank32 +24 gains, те же optimizer/data/start.
GPU0:0.9CE+0.1×T²KL, GPU1:0.5CE+0.5×T²KL, T2.
KL exact full-vocabulary teacher-to-student. Teacher — исходный dense Qwen,
весь текст проходит через teacher; это не attention target на hybrid layer input.
Frozen teacher tensors, shared vocabulary/norm, никакой top-k аппроксимации.
Сравнить с CE-only arm EXP122 GPU1 при одинаковом additional-update cursor и
отдельно wall cost: teacher forward/head work дополнительный и учитывается.

```python
from scripts.m3q_t4_scale_campaign import main as run_t4_long
result = run_t4_long([
    "--experiment", "123",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "2",
])
```

## Как считается gradient и что измеряется

Frozen source BF16 на host, один decoder block на GPU. Forward сохраняет inputs
слоёв на host; head CE/KL по32-position tiles. Backward recomputes каждый блок
и передаёт hidden cotangent к предыдущему через JAX VJP. CE/KL нормализуются
на все255 targets **до** backward cast. Градиенты trainable q собираются,
клиппируются совместно и обновляются одним Adam step. Это end-to-end chain rule,
не truncated/detached blockwise training. Память/скорость реального1.7B backward
на T4 ещё должны быть измерены первым запуском.

На tiny FP32: все6 training recipes streamed gradients сравниваются с полным
autodiff той же функции, включая KL и penalty. BF16 gain1 control; exact
next optimizer update после chunked restore; nonfinite proposal rejection;
контролируемый deadline/resume без повтора завершённых updates.

Evaluation steps0/128/512/1024/2048/4096/8192/16384/32768/49152/65536, selection
по A16; после обновлений также B32 monitoring для сравнения arms на одинаковом
update cursor. B используется повторно как validation, не blinded confirmation.
Best parameters сохраняются отдельно от current+optimizer. В reserve:
last+best на A64/B32/C64; best context128/256/512/1024 на тех же tail127
targets; layer profile на первом явном A-window. Есть raw/scaled mixer RMS,
residual/MLP RMS, fraction nonzero MLP additions lost in BF16 addition.
Inspect-executable относительно ordinary layer output проверяется по relative
max discrepancy<=0.02; rejected profile rows не использовать как accepted math.
Это новый отдельно зарегистрированный criterion; rejected traces116 не переименованы.

Practical exploratory signal: paired mean NLL improvement хотя бы0.1,
с согласованным знаком по A/B/C, а также улучшение context response. Это не
статистический confidence threshold и не формальная multi-seed confirmation.
Tiny сдвиги0.001 не считать пробитием плато. При отсутствии эффекта не выбирать
условие только за меньшие RMS/penalty/training loss.

## Сохранение, лимиты и файлы

Local current+Adam+best snapshots128 updates; remote4096 updates или90min,
branch-end/clean deadline. При внезапном kill local-only updates теряются:
report cursor сверяется с checkpoint, future metrics обрезаются, attempted
cost не откатывается. Best step должен быть<=restored cursor. Same cell resumes.
GPU0/GPU1 checkpoints независимы. Storage cap2GB **суммарно на invocation**, по1GB
на worker; source/full-model weights не загружаются обратно в HF. Если очередной
checkpoint не помещается или sync отказал, training останавливается с явным
storage_partial/pending flag вместо бесконечного накопления несохранённых updates.
После cap остановки последняя работа может быть только локальной — durable_step
является единственной границей гарантированно сохранённого прогресса.

Reports local после каждой probe/health row; HF раз10min и в конце. Telegram
start/worker-end/session-end/error с pending/report-sync flags. Можно добавить
`"--no-telegram"`. Нельзя одновременно запускать две копии одного EXP/prefix.

С каждого notebook скачать **один** `extent-m3q-t4-120.json` (соответственно
121/122/123). Он объединяет workers. Output dir прежний `extent-results`.
Все stdout logs и worker reports сохраняются дополнительно, отдельно скачивать
их для обычного анализа не требуется. HF paths `experiments/exp120-t4/worker-N`
и аналогично121–123. У всех workers completed обучение заканчивается раньше
лимита, если весь65536 horizon уже выполнен; никаких пустых ожиданий.

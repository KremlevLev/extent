# EXP116–119: отдельные Kaggle T4 x2 сессии

Подготовлено 15 октября 2026. **Результатов GPU пока нет.**
Ноутбук: `notebooks/extent-t4.ipynb`. Настройки GPU T4 x2 + Internet,
те же deploy-key/HF secrets. Выполнить все setup-ячейки, затем одну из ячеек ниже.
Каждый эксперимент занимает обе T4 независимыми задачами; не вставлять все
четыре запуска последовательно в одну сессию. Максимум 11.5 часов от runner,
из них 45 минут сохранение; при долгом setup уменьшить wall по остатку 12h.
Сессия может закончиться раньше, если все задачи выполнены. Первый облачный
run измерит практическую скорость; обещания заполнить ровно 12h нет.

## EXP116 — где накапливается ошибка

Все 28 слоёв, 8 A-windows на слой: drift входов, ошибка mixer/MLP на teacher
и hybrid inputs, разделение local error/input-shift effect, усиление residual.
Для 24 Mamba также distributions dt/decay/gates/B/C, injection/carry/readout,
rank participation/cancellation по времени. Dynamics принимаются только при
local trace max-abs <=1e-4; rejected traces сохраняются как отказ, не как диагноз.
Карты делят слои пополам. Это подробный frozen-model профиль, не обучение.

```python
from scripts.m3q_t4_campaign import main as run_t4
result = run_t4([
    "--experiment", "116",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "2",
])
```

## EXP117 — контекст и память

128/256/512/1024 контекст, одни и те же последние 127 target positions для
каждого из 8 contiguous 1024-token B-windows. Сравнить обычный гибрид и Qwen;
затем сбрасывать состояние всех Mamba каждые32/64/128 позиции. Сначала
same-executable reset-period0 control, max per-window NLL difference <=0.01.
Сравнение reset с traced normal carry, а не с другим executable. При отказе
control reset-эффекты не вычисляются. GQA всё ещё видит весь контекст;
reset — вмешательство в recurrent state Mamba, не удаление текста.

```python
from scripts.m3q_t4_campaign import main as run_t4
result = run_t4([
    "--experiment", "117",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "2",
])
```

## EXP118 — влияет ли численная точность на плато

BF16 control против FP32-all / FP16-all и FP32 только в ранних/средних/поздних
Mamba-блоках (включая MLP в выбранном блоке). Статические BF16 параметры и
FP32 recurrence сохраняются; tied vocabulary head остаётся BF16. Это compute
precision intervention, не новая инициализация/обучение. 8 A/B/C-windows:
paired NLL, finiteness; nonfinite FP16 не считается качественным результатом.

```python
from scripts.m3q_t4_campaign import main as run_t4
result = run_t4([
    "--experiment", "118",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "2",
])
```

## EXP119 — локальная дистилляция + увеличение обучаемой ёмкости

Фиксированные слои0/12/26 покрывают начало/середину/конец, выбор не использует
результаты116. По6 методов на слой, seed123: новые ADD32/64/128/256 поверх
уже folded EXP111 rank32, FREE256 снимает protection, ALL-MAMBA открывает весь
один mixer. ADD32 здесь **добавочная** поправка, не total rank32 EXP115.
Все начинают с точного исходного слоя; старый восстановленный слой заморожен
в ADD/FREE. ALL-MAMBA открывает его FP32 masters. Цель — воспроизвести Qwen
attention на нормализованных **реальных hybrid-prefix inputs**, FP32 relative
MSE; Adam3e-4 b1=.9/b2=.999, stable clip1. 1024 updates на128 train windows
(8 проходов); input-token cost и повторное обучение после rollback учитываются.

После каждого fit — вставка одного слоя в полную28-layer модель, NLL на8 A/B/C
windows. Улучшение MSE без улучшения NLL не победа. Стартовые ранги имеют общий
случайный prefix, нулевую B. Это отличается от end-to-end CE EXP115, не дубль.
Single seed и reused validation — exploratory; лучшие кандидаты затем
подтверждать на новых данных/сидах и TPU full-model recovery.

```python
from scripts.m3q_t4_campaign import main as run_t4
result = run_t4([
    "--experiment", "119",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "2",
])
```

## Общие контроль и сохранение

Patient: immutable EXP111 seed123 MAIN32768, revision
f839c2a898c3e9eff9907ea3cf1e022fe16f3e09. Проверить все source SHA, digest
ONPOLICY base, exact manifest и каждый parameter chunk. Adam111 не скачивать.
Хост хранит BF16 student/teacher, GPU получает по одному блоку и small activation
cache. Перед probes проверить все64 A-windows против сохранённого TPU replay,
max per-window NLL difference <=0.03. Отказ останавливает эксперимент.
BF16 fusion boundaries GPU/TPU не обязаны дать bitwise match; каждый treatment
сравнивать со своим GPU baseline. До patient download notebook проверяет
tiny-model streamed vs monolithic token NLL на обеих картах, tolerance0.01.

A/B/C — повторно используемые EXP114 validation. FIT первые128 windows EXP115
train74973184. Не называть C новым locked test. В117 на одинаковых targets
сравниваются одинаковые contiguous окна; длины не выбирать по итоговому NLL.

Local reports после каждой задачи; HF раз в10min и в конце. Fit snapshots128
updates локально, удалённо при branch-end/штатном deadline. Abrupt cloud kill
до sync может потерять local fit updates; durable cursor сохраняется отдельно,
reported future history обрезается при restore, cost не откатывается. Quota
cap2GB **всего на вызов**, по1GB/worker: только новые fit states + reports,
никаких больших full-model копий в HF и удаления старых файлов. Cap/sync
failures видны в pending_fit_states/report_sync_error; completed tasks не
означают, что каждый checkpoint durable. Telegram started/worker-end/session-end.
Output JSON и summary, оба worker JSON и GPU logs сохранять вместе.

Отключение Telegram: добавить `"--no-telegram"`. Для одной карты: `"--workers","1"`;
setup notebook всё равно проверяет T4 x2. Возобновление — та же launch-ячейка.
На одном HF prefix нельзя одновременно запускать две копии одного EXP.

CUDA12 установка по [официальным инструкциям JAX](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).

# EXP-114: запуск подробного обследования плато

Один аккаунт, TPU v5e8,9h. Обычные setup/clone/install/secrets уже выше.
Нужен актуальный main с EXP114 files и зарегистрированными source/data manifests.
Не запускать EXP114 одновременно с другим JAX workload в том же runtime.
Реальный TPU HBM/throughput этого диагностического runner пока не измерен.

```python
%cd /kaggle/working/extent

from scripts.m3q_plateau_forensics_114_campaign import main as run_exp114

result = run_exp114([
    "--max-wall-hours", "8.8",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "1",
])
```

8.8h включает source download/restore, token packing, compile, диагностику,
readout FIT и upload. Рабочий deadline7.8h,1h внутри для сохранения.
12min внешний allowance на setup. Если setup дольше, вычесть превышение из8.8.
Не обещаем завершения всех модулей в одном вызове: deadline_partial штатен.
Для продолжения — та же ячейка, с сохранённым task cursor и readout Adam.
Main patients EXP111 не обучаются и повторно в HF не выгружаются.

Проверяем: обе MAIN32768 EXP111, полный layer sweep, вклад upstream input shift,
teacher mixer/boundary/rollback rescues, MIMO recurrence, направления Adam,
BF16 materialization, output-only rank32 alignment и проверку на других текстах.
Для readout FIT платим4096*256*2=2,097,152 input tokens и frozen forward passes.

HF namespace `experiments/exp114-plateau-forensics`.
Сохранить/прислать `output/extent-m3q-plateau-114.json` и
`output/extent-m3q-plateau-114-summary.txt`.
JSON содержит intermediate completed probes даже при partial; отсутствие
отдельного модуля не заменяется нулями. Telegram сообщает start, завершение
модулей, final status и pending. Внешний kill может не вызвать callback.

`--max-new-hf-gb 1` — cumulative upload payload cap этого вызова, не квота HF.
Только отчёт и малые readout states; source model checkpoint copies не пишутся.
Никакие старые HF файлы автоматически не удаляются. При429 соблюдается cooldown,
при quota/network error pending отмечается явно. Готовый локальный файл и durable
HF файл различаются; statuscompleted не означает успешный scientific gate.

Если исходные Python111 зависимости или patient32768 endpoint не совпадают с
immutable pin, runner останавливается до scientific probes; не подставляет latest
или заново обученную модель. BF16 выключенные hooks оцениваются в том же пути,
canonical background записывается отдельно; background>0.01 NLL — stop.
Original endpoint replay сохраняет1e-4 threshold.

Discovery A намеренно reused. B/C Wiki token-disjoint, но одна статья частично
общая; PG19 — новая область того же test manifest, независимость книг не доказана.
Полный provenance в data manifest; нет заявления о независимых article replicates.

Перед запуском исходные pins/data SHA прошли CPU/read-only HF preflight.
Tiny CPU/virtual8 проверки — инженерные, не реальный1.7B TPU smoke.

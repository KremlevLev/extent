# EXP-111–113: локальная проверка, 11 октября2026

PREPARED, НЕ TPU evidence. JAX0.6.2/Python3.12, восемь виртуальных CPU устройств.
Ни одной новой TPU training сессии не запускалось; full1.7B memory/throughput
и registered scientific gates ещё не измерены.

- `tests/test_efficiency_recovery.py`, `tests/test_layer_clip_recovery.py`,
  `tests/test_batch_recovery.py`:30 passed в155.97s. Проверены compiled BF16
  adapter/full-decoder CE, protected columns, frozen vocabulary, точный следующий
  шаг после binary Adam restore; существующие106–110 contracts не изменились.
- После финальных diagnostic memory guards и software-version resume guard:
  повторены read-only CE/FP32/rollback probe и три mocked campaign scenarios.
  Отдельный mock full-decoder сценарий проходит все шесть pilots, validation-only
  выбор LR и charged MAIN replay;16 updates при сокращённом тестовом горизонте,
  в cloud уходят только два MAIN states, завершённые pilots metrics-only.
- Потеря VM: локальные файлы уничтожаются только в test temporary directory;
  cloud-only resume восстанавливает Adam/cursor. JSON, опережающий durable binary,
  откатывается; deferred milestone probe не объявляется завершённым и повторяется
  без лишнего training update. Счётчик затрат сохраняет выполненные попытки.
- Payload cap проверен ДО checkpoint upload; превышение оставляет pending и
  уведомление. Старые generic store tests покрывают429/cooldown, transient/auth
  failures, corruption/SHA и independent remote sample verification. Cloud и
  Telegram в этих тестах MOCKED, не проверка живых write permissions.
- Реальный tokenizer/data preflight: pinned Wiki/PG19, все capacity guards и
  token SHA прошли; train32768x256, Wiki selection64x256, locked holdout64x256,
  reproduction64x256, PG19 test128x256. SHA: EXP-111-113-data-preflight.json.
- Live read-only HF source preflight:48 EXP069 manifests +48 EXP072 manifests
  и96 referenced payload paths на pinned9c031f6e615d7afb0133f5611ace78f8cc5cc5fc;
  contracts/declared SHA совпали, referenced payloads2,053,833,984 bytes.
  Binary weights не скачивались локально; payload SHA проверится runtime restore.
- Все три entry points проходят plan-only без модели/TPU allocation;
  archived v1 contracts и launch-preflight JSON.8.8h total/7.3h training-side
  deadline/90min sync reserve, caps2/35/35 decimal GB per invocation.

BF16 diagnostic forward с gradient/captured states может округляться иначе,
чем canonical evaluation: эти NLL раздельны. Тест доказывает равенство canonical
NLL обычному evaluate, одинаковые initial/current contribution metrics при
identical weights, null rollback и неизменность исходного tree digest.
FP32 probe использует те же представимые значения, FP32 math и tied-readout dtype;
это tiny CPU correctness, а не full-model FP32 TPU feasibility.

HF audit read-only: current logical tree1.87155TB, API usedStorage4.08611TB,
account remaining quota UNKNOWN. Существующие remote files/history не удалялись
и не переписывались. Новые training states ещё не публиковались.

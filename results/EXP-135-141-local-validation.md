# EXP135–141: локальная инженерная проверка

17 октября2026. Это НЕ научный GPU/TPU-результат и не замер скорости T4.

- Заключительный набор до последней небольшой правки статусов:19 проверок passed за63.50с.
  Новые проверки охватывают реальные compiled updates на двух виртуальных CPU-устройствах,
  сравнение с одним устройством, BF16 full-chain gradients, живые градиенты dense-динамики,
  совпадение начальных функций R32/dense, свёртку130, чистый QKVO mapper из teacher tensors,
  послойный профиль, рецепты и путь вызова фактического runner.
- После добавления отдельного случая отклонённого первого обновления и уточнения статусов:
 3 проверки passed за6.45с (две ветки реального orchestration caller и планы).
  Принятое продолжение сохраняет счётчик3434→3436 и переданные Adam-счётчики; отклонённое
  остаётся3434, не меняет параметры/оптимизатор, сохраняет failed_stages.
- В тесте first_update_health записан ДО любого принятого обновления.
- source130 audit: все487 частей114,979,740B проверены SHA256; восстановлен НАСТОЯЩИЙ
  payload в настоящий optimizer template; текущий parameter SHA совпал, best3072,
  все6 Adam/schedule counters3434. Доказательства: EXP-135-source-audit.json.
- Сверены геометрия source111 и YAML-конфигурация нового runner: совпадают.
- Все7 самостоятельных ячеек разобраны AST;ID соответствуют файлам. Планы сформированы
  без облачного выполнения, не требуют токена/ускорителя для plan-only.
- Старые130–132 core/runner/контракты не редактировались; исходный HFprefix130 только читается.
  Общий TPU helper получил health-диагностику и параметр block_fn с прежним поведением по умолчанию.

Ранее полный набор tests/test_competitor_tpu.py проходил15 проверок на8 виртуальных CPU
за56мин39с; после появления новых follow-up helpers заново проверено релевантное подмножество,
а не повторялось дорогое1024-token испытание без изменения его математической схемы.

Не проверено локально: реальная CUDA/NCCL/T4-скорость, GPU-память модели1.7B на физическом
устройстве, TPU-компиляция1.7B, прохождение source replay.03 на физическом устройстве,
первый реальный градиент новых веток, достижение32768 за одну сессию и новое качество.
Раннер проверяет источник и первое предложение на месте; при ошибке не принимает обновление.


## Заключительная проверка

После добавления legacy-gradient replay и явной передачи frozen base в evaluators:
**23 проверки passed за94.38с**. После расширения потолка до131072:
**12 follow-up проверок passed за81.97с**. Среди них реальный legacy streamed130
против full-compiled градиента на tiny-модели, дополнительный выход градиента,
обработка больших значений без переполнения при сравнении, и реальные ветви orchestration
для принятого/отклонённого обновления. Никаких градиентов7M-элементного дерева не сериализуется
на каждом шаге: дополнительный gradient output135 используется только для первого сравнения
и остаётся на устройствах в остальном цикле. Базовые1.7B-веса передаются явными аргументами
compiled evaluators, не замыкаются как большие константы.

Семь plan-only контрактов иAST ячеек пересозданы для текущего кода. Первая цель32768,
потолок131072, работа до лимита времени; отдельное HF-сохранение135 на32768.
Для прекращения строго на32k до первого запуска выбрать --max-steps 32768.
135 cap6GB;136–138 по4GB;13924GB;1404GB;14140GB. Суммарный верхний предел новых
загрузок86GB, не оценка обязательного расхода. Репозиторий HF не очищался.


## EXP135 startup failure: wall budget validation (2026-10-17)

User cloud log stopped at 111.5 seconds in main(), before session creation or
training: the inherited EXP108 helper rejected 11.5 hours (hard limit 9).
The supplied GPU cells for 135-138 were all affected; TPU 139-141 at 8.5 hours
pass that particular check. Dependency warnings were not the fatal exception.
Earlier tests mocked session creation/run and missed the real CLI budget guard.

The follow-up runner now owns a hardware-aware budget helper: <=12h GPU,
<=9h TPU, >=30min reserve strictly below the total wall time. Both main() and
the actual session constructor use it; older experiment limits are unchanged.
17 budget regression checks passed (0.96s), including real plan-only parent
and child entrypoints for all seven experiments, real session constructors
for GPU/TPU with remote I/O mocked, invalid limits and nonfinite inputs.
This does not establish full-scale accelerator memory, throughput or training.

The supplied failure happened before any new contract/report/checkpoint could
be created, so EXP135 can rerun its existing pull-and-reload launch cell.
Do not interrupt already-running jobs. Code hashes change with this fix;
any job that created state on the previous revision must resume that exact
revision, rather than bypassing its contract check. The engineering plan was
regenerated for the fixed code. Scientific recipe/source/optimizer unchanged.

Full follow-up suite collected before adding the two session-constructor cases:
27 passed in 59.13s. Both added cases then passed in the 17-case budget subset.

# EXP177: запуск после GPU-проверки

На сервере, при запуске около19:00 МСК. Используется существующий `torch176-env`; предварительно обновить checkout до коммита с `engine_torch177.py`. `tomorrow 09:00` корректно при запуске вечером.

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
git pull --ff-only
source ../torch176-env/bin/activate
set -a
source ../engine-core.env
set +a
unset LD_LIBRARY_PATH
RUN_DIR="$PWD/../extent177"
mkdir -p "$RUN_DIR"
END_AT="$(TZ=Europe/Moscow date -d 'tomorrow 09:00' --iso-8601=seconds)"
nohup python -u -m engine_torch177 \
  --work-dir "$RUN_DIR" \
  --until "$END_AT" \
  --corpus "$PWD/../engine-core/data-pack-edu-600m" \
  --c4 "$PWD/../engine-core/data-pack" \
  --save-reserve-minutes 25 \
  --memory-gib 90 \
  --clear-stop \
  >> "$RUN_DIR/runtime.log" 2>&1 < /dev/null &
echo "PID=$!; deadline=$END_AT"
tail -n 40 -f "$RUN_DIR/runtime.log"
```

Ручной просмотр: `tail -f /data/workdir/_biometric-old/biometric-linking-guarded-v4/extent177/runtime.log`; Telegram `/check`. Не передавать токены в чат или логи. `Ctrl+C` внутри `tail` не останавливает обучение.

Мягкая остановка: `touch /data/workdir/_biometric-old/biometric-linking-guarded-v4/extent177/STOP`. Быстрая пауза без HF-выгрузки: `touch /data/workdir/_biometric-old/biometric-linking-guarded-v4/extent177/FAST_STOP`; состояние остаётся только в RAM. Повторная команда с `--clear-stop` возобновляет план из сохранённого checkpoint; если RAM потеряна до HF-выгрузки, нельзя обещать восстановление последних шагов. Для нового независимого опыта выбрать новую рабочую папку.

Тестированная физическая GPU-производительность финальной версии отсутствует; preflight отбирает integer batch1..8, ограничивается выделенным временем и изолирует OOM. Отдельные старые GPU-пробы до запрета пользователя не доказывают работоспособность полного runner. См. [протокол](EXP-177-night-protocol.md).

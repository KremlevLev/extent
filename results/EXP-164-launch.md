# EXP164 launch and resume

Wait until the prior training process exits and releases server.lock. Use the existing environment and work directory. No new dependencies are needed beyond requirements-blackwell/status already present.

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
git pull --ff-only
source ../engine-core-env/bin/activate
set -a
source ../engine-core.env
set +a
unset LD_LIBRARY_PATH
RUN_DIR="$PWD/../engine-core"
HOURS=14
nohup python -u -m engine_launch --runner full \
  --work-dir "$RUN_DIR" --max-wall-hours "$HOURS" \
  --save-reserve-minutes 18 --clear-stop \
  >> "$RUN_DIR/runtime-full.log" 2>&1 &
tail -n 40 -f "$RUN_DIR/runtime-full.log"
```

Change HOURS per invocation. The initial launch restores EXP163 FULL-KD CURRENT157684; later invocations restore EXP164 q/Adam/data state. It does not reset moments or replay the consumed Edu training prefix. The planner always runs before accepting updates. Bot registration and the single listener are automatic with the same Telegram credentials. `/check` shows the new report, selected batch and explicit tokens. Bot polling logs live under `/tmp/engine-status-service-<token hash>/listener.log`; the token itself is not printed. `--no-telegram` explicitly disables auto-start and notifications.

Soft pause (self-contained path, works in a new terminal):

```bash
touch /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core/STOP
```

Wait for final save/upload and actual process exit; rerun the launch block to resume. Ctrl+C while tailing stops only tail. Never use a global Python kill or alter the face service.

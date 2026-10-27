# EXP166 launch

Wait for failed165 process exit. Keep its source report/state. The default full route now starts166 from CURRENT157685; later calls resume166. Leave-free budget is computed by the launcher, so the long inline Python shell block is no longer needed.

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
git pull --ff-only
source ../engine-core-env/bin/activate
set -a
source ../engine-core.env
set +a
unset LD_LIBRARY_PATH
RUN_DIR="$PWD/../engine-core"
HOURS=12
nohup python -u -m engine_launch --runner full \
  --work-dir "$RUN_DIR" --max-wall-hours "$HOURS" \
  --leave-free-gib 5 --save-reserve-minutes 18 --clear-stop \
  >> "$RUN_DIR/runtime-full-stream.log" 2>&1 &
tail -n 40 -f "$RUN_DIR/runtime-full-stream.log"
```

Set HOURS to available time. The planner measures all8 streamed batches; do not interpret compilation as accepted training. /check automatically reads runtime-full-stream.json. Send that JSON, not the historical runtime-full-next.json.

Soft stop from any terminal:

```bash
touch /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core/STOP
```

Wait for final save/upload and actual process exit before resuming with the same launch command. Ctrl+C while tailing only stops tail. Explicit full164/full165 routes remain historical; do not use them for this correction.

# EXP162 launch — existing Blackwell server

Use existing nested checkout and existing JAX environment. EXP161 must have
finished and saved; C4 CURRENT154852, no HF pending. User selects duration with
HOURS below. No source resets or parent state deletion. Do not stop other services.

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
git pull --ff-only
source ../engine-core-env/bin/activate
set -a
source ../engine-core.env
set +a
unset LD_LIBRARY_PATH

HOURS=21
RUN_DIR="$PWD/../engine-core"
nohup python -u -m engine_compare \
  --work-dir "$RUN_DIR" \
  --max-wall-hours "$HOURS" \
  --save-reserve-minutes 18 \
  --clear-stop \
  > "$RUN_DIR/runtime-compare.log" 2>&1 &

tail -n 40 -f "$RUN_DIR/runtime-compare.log"
```

Ctrl+C stops log viewing only. Launcher automatically prepares bounded Edu pack
on CPU if missing (up to3h inside selected wall limit), preserving its journal
if stopped. Existing100M C4 pack/caches reused. The first source shard download
can be sizeable; only one shard retained temporarily. No pip/JAX/CUDA upgrade.
If preparation stops, rerun same command to resume; no GPU updates before a
complete SHA-checked corpus and immutable parent plan.

Optional prepare Edu BEFORE starting the timed experiment:

```bash
python -u -m scripts.prepare_edu_corpus \
  --output "$RUN_DIR/data-pack-edu" \
  --c4-dir "$RUN_DIR/data-pack" \
  --train-tokens 100000000 --max-hours 3
```

This step is CPU packing/download, not model training. Do not run another packer
on the same directory; build.lock prevents collisions.

## Telegram /check

The existing listener does NOT magically discover a new report. Add a compare
entry to your external status-sessions.json and restart ONLY engine_status:

```json
{
  "sessions": [
    {"id":"compare", "work_dir":"/data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core", "report":"results/runtime-compare.json"},
    {"id":"previous", "work_dir":"/data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core", "report":"results/runtime-b6.json"}
  ]
}
```

Use `pgrep -af 'python.*-m engine_status'`, inspect the PID and `kill PID` ONLY
for this listener (not engine_compare). Start the listener once:

```bash
nohup ../engine-core-env/bin/python -u -m engine_status \
  --work-dir "$RUN_DIR" --config "$RUN_DIR/status-sessions.json" \
  >> "$RUN_DIR/status-bot.log" 2>&1 &
```

Then /check compare or /check previous. NLL E and BEST C4 D supported in updated
listener. Old161 still labels BEST Wiki A. A held shared lock for previous report
is a WORK-directory process check, not evidence previous161 is training again.

Return runtime-compare.json for analysis. No need to download checkpoint chunks.
End state may be completed (data limit) or deadline_partial (time limit); both
require checking pending_slots and per-branch durable/finalization fields.

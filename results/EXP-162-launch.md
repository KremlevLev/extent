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

## Optional process title and repeated daytime pauses

To request the process title VLLM:EngineCore, install requirements-label.txt in
the user's existing venv and launch `python -u -m engine_launch` with exactly the
same engine_compare arguments. This is a cosmetic process title; runtime remains
the same JAX experiment, not vLLM. Wrapper stays in the SAME PID, forwarding args
and signals without spawning a separate trainer. It does NOT modify pinned
engine_compare/scientific sources or invalidate existing EXP162 checkpoints.

```bash
python -m pip install -r requirements-label.txt
nohup python -u -m engine_launch \
  --work-dir "$RUN_DIR" --max-wall-hours "$HOURS" \
  --save-reserve-minutes 18 --clear-stop \
  >> "$RUN_DIR/runtime-compare.log" 2>&1 &
```

The first log line prints requested title and nvml_process_name (actual driver
query before CUDA context creation, if available). setproctitle changes Linux
process title; exact NVIDIA display is driver/platform dependent and has NOT
been verified on the user's Blackwell here. SPT_NOENV protects /proc/environ.
Installing this small optional helper does not update JAX/CUDA dependencies.
Already-running processes keep their existing title; change it on next resume.

Soft pause any time, including CPU packing:

```bash
touch "$RUN_DIR/STOP"
tail -n 40 -f "$RUN_DIR/runtime-compare.log"
```

During training the runner stops at a safe check, saves both CURRENT/Adam/cursors
locally and attempts HF sync, then final diagnostics within remaining reserve.
This is NOT instantaneous GPU release; wait for the process to EXIT. A momentary
0% GPU utilization is not completion. Final runtime-compare-status.json is written
and the shared server.lock released at launcher exit. HF pending may remain on
network failure; locally saved progress is still independently resumable.

Resume using the SAME launch command, work dir, corpora, source plan and code.
--clear-stop removes the request file; own EXP162 checkpoints take precedence
over immutable source154852. Parameters, Adam, own BEST and distinct C4/Edu
cursors resume; data does not cycle and new tokens accumulate across sessions.
Each invocation's HOURS is a NEW wall budget, not a reset of model/data progress.
2–3 soft pauses/day are supported; repeated loading/audits/validation cost time.
No need to git-pull/reinstall on every pause. Never delete states/runtime-compare
or corpus-switch/source.json to resume. Do not use kill -9 or pkill python.

References: https://github.com/dvarrazzo/py-setproctitle and
https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlSystemQueries.html .

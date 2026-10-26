# EXP163 — Blackwell14h methods comparison

If162 is running, request soft stop and WAIT FOR PROCESS EXIT before launching163.
Do not kill other GPU services; shared server.lock rejects competing launches.
Use the SAME source work dir containing saved EXP161 CURRENT154852. Paused162
states are retained but not used as source. No JAX/CUDA upgrade needed.

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
nohup python -u -m engine_launch --runner methods \
  --work-dir "$RUN_DIR" --max-wall-hours "$HOURS" \
  --save-reserve-minutes 18 --clear-stop \
  >> "$RUN_DIR/runtime-methods.log" 2>&1 &

tail -n 40 -f "$RUN_DIR/runtime-methods.log"
```

Optional label helper is the same already installed requirements-label.txt. To
launch without process-title helper, use `python -u -m engine_methods` with the
same arguments. All methods use FineWeb-Edu; if100M pack incomplete it is prepared
on CPU within selected hours before weights load. Data preparation journal resumes.

Soft pause, then wait for save/finalization and actual process exit:

```bash
touch "$RUN_DIR/STOP"
```

Resume with same launch command/HOURS changed as needed. No reinstall/pull needed
between pauses. Same data/source/code contracts required; q/Adam/cursors resume.

## Telegram listener

Replace/add the current session entry in WORK/status-sessions.json:

```json
{"sessions":[{"id":"blackwell","work_dir":"/data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core","report":"results/runtime-methods.json"}]}
```

Restart ONLY the existing engine_status listener with --config pointing to this
file. Inspect `pgrep -af '[p]ython.*-m engine_status'`, kill ONLY that listener PID,
then start once using the same token env:

```bash
nohup ../engine-core-env/bin/python -u -m engine_status \
  --work-dir "$RUN_DIR" --config "$RUN_DIR/status-sessions.json" \
  >> "$RUN_DIR/status-bot.log" 2>&1 &
```

/check shows three methods, measured NLL steps, token costs, own BEST D and HF
pending states. /check does not control training. runtime-methods.json appears
after source/corpus preparation; no report during CPU packing is not a crash.

## Budget caveats

14h includes preparation/loading/compilation/evaluation and18min reserve. First
16 accepted updates/method measure speed; logs then report a common adaptive
target. It is extended if time remains. Full decoder may be much slower than
rank256, especially if it needs accumulated batch6. Do not expect last run's
~100M inputs in19h to apply unchanged. Compare common token endpoints.

Full-state HF payload is large: about29GB for all methods. Full states are sent
ONLY at exit/soft pause, not every4h; small reports continue during work.32GB
default session upload cap can leave pending after network failure or unexpected
payload growth. Check durable_step and pending_slots before assuming a remote
copy exists. Larger optional --max-new-hf-gb is explicit; no automatic unlimited
quota consumption. One current generation per arm is retained locally, atomically
replaced every30min and at exit. Frequent soft pauses still upload large states;
they are resumable but are not free network/storage operations.

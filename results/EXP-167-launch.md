# EXP-167 launch: one Blackwell session, three matched LR arms

First stop previous training and wait for its actual process exit and successful
save. Do not run166 and167 concurrently in the same work directory.

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
git pull --ff-only
source ../engine-core-env/bin/activate
set -a
source ../engine-core.env
set +a
unset LD_LIBRARY_PATH

RUN_DIR="$PWD/../engine-core"
HOURS=9

nohup python -u -m engine_launch --runner lr \
  --work-dir "$RUN_DIR" \
  --max-wall-hours "$HOURS" \
  --pilot-updates 4096 \
  --leave-free-gib 5 \
  --save-reserve-minutes 18 \
  --clear-stop \
  >> "$RUN_DIR/runtime-lr.log" 2>&1 &

tail -n 40 -f "$RUN_DIR/runtime-lr.log"
```

Set HOURS to the allowed TOTAL duration, including all three arms, preparation,
diagnostics and saves. Each arm4096updates/8.39M inputs; all arms25.17M inputs.
This is a deliberately bounded comparison and can finish before HOURS. It does
not automatically spend the remainder continuing a guessed winner. Source fixed
to received166CURRENT181145; if that donor pointer has advanced, fail rather than
silently compare different sources. Default disk cap220GB/free reserve20GB;
declared upload cap96GB total (~66GB payload across3arms), not repeated hourly
uploads. No dependency upgrade: preserve existing166 math runtime.

Status: existing Telegram /check follows the active arm; console heartbeat5min.
When finished, send only **results/runtime-lr.json** for the comparison. Per-arm
details remain runtime-lr-x1.json, runtime-lr-x3.json, runtime-lr-x10.json.

Soft stop from any shell:

```bash
touch /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core/STOP
```

Resume with the SAME command and unchanged --pilot-updates, a new HOURS budget,
and --clear-stop after actual exit. Complete arms are skipped; partial arm resumes
its own q/Adam/data before the next arm. No extra bot/token required.

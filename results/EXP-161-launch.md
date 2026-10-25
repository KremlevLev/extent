# Neutral single-GPU batch6 continuation

Existing dependencies and environment suffice. Parent160 must be stopped after a
normal checkpoint/save; the completed capacity probes did not change its state.
Do NOT remove the existing work directory or parent states.

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
git pull --ff-only
source ../engine-core-env/bin/activate
set -a; source ../engine-core.env; set +a
unset LD_LIBRARY_PATH

RUN_DIR="$PWD/../engine-core"
nohup python -u -m engine_batch \
  --max-wall-hours 21 \
  --clear-stop \
  --work-dir "$RUN_DIR" \
  > "$RUN_DIR/runtime-b6.log" 2>&1 &
```

Change21 to actual hours available FROM this invocation. The parameter includes
initial preparation/compile/evaluation and18min final reserve. No pip upgrades,
new checkout or another full copy of the dataset/model needed.

```bash
tail -f ../engine-core/runtime-b6.log
```

At startup expect parent/current/Adam restoration and initial NLL reference
checks, then `engine-core batch6: gradient audit` and, after8 accepted updates per
branch, throughput. `compiled-batch6` is the fast path; `accumulated-batch6` is an
explicit safe fallback with same effective batch but potentially lower throughput.
Startup is not a new full sweep over batch sizes; the batch is FIXED at6.

Each new update consumes6 windows,1536inputs/1530targets; source pin, global and
data-window counters are explicit. Original weights and Adam retained. Own files:
- results/runtime-b6.json — result to send.
- runtime-b6-status.json — launcher status.
- batch6/source.json — immutable parent plan (do not edit on resume).
- states/runtime-b6 — bounded trainable checkpoints, separate from parent160.

Stop with `touch ../engine-core/STOP`, wait for save/process exit. Resume with the
SAME command, changing only hours as needed. Own161 CURRENT/Adam/data resumes,
not a new fork from original160 each time. Start and finish notifications only.
Pending HF states in the final notice require keeping local files.

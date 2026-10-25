# Device execution upgrade for the running single-GPU EXP160

25October2026. Observed server baseline:~0.8095s/update,~316 input positions/s,
~17.1GiB own GPU allocation, batch1/context256. This is an actual user-supplied
measurement of the previous worker, not of this upgrade. One nvidia-smi sample
does not identify the bottleneck or guarantee a particular speedup.

New optional neutral entry point: `python -m engine_runtime --engine auto`.
Existing `engine_worker` remains unchanged. NO original160/155 mathematical
source or server-control file is edited, because the current checkpoint records
their hashes and must remain resumable.

## Candidates and gates

- legacy: unchanged host-activation, per-call synchronized execution.
- resident: same block/head tile order and precision, device activation storage,
  asynchronous per-layer launches, one explicit synchronization per complete step.
- compiled: whole teacher/student objective, gradient and Adam step compiled on
  device. It keeps block/head AD intermediates rather than re-materializing them.
  This uses additional GPU memory to avoid host transfers, launch overhead and
  recomputation. It does NOT change batch, tokenization, context, loss or optimizer.

Before accepting any first proposal, compare current-q/current-Adam gradients
against legacy on the same real next batch. Resident has tight relativeL2<=1e-5
and CE<=1e-5. Whole-compiled uses the existing registered compiled/streamed BF16
gate (relativeL2<=.15, cosine>=.98, CE<=.03, all proposal stages finite). This is
bounded numerical agreement, NOT bitwise or long-trajectory identity. Fusion
can change rounding. No threshold is relaxed on a failed comparison.

After warmup, time three synchronized unaccepted proposals per valid backend;
auto chooses the fastest median. A memory guard/OOM or failed gradient gate
excludes a candidate. Arbitrary non-memory implementation errors are not hidden
as OOM. Legacy stays a valid fallback, so speedup is measured rather than promised.
`--engine resident` avoids whole-graph compilation; `--engine compiled` prefers
compiled only if valid, otherwise safe fallback. `--engine legacy` launches the
unchanged old worker. No memory is reserved just to make nvidia-smi look busy.

The audit reuses SAME input q/Adam/batch without accepting trial updates. Only
one returned proposal advances optimizer/data. `engine_gate` records checks,
chosen engine, timing and extra teacher/student audit token presentations. There
can be twelve gradient evaluations for the first audited proposal, not twelve
training updates. First8 accepted updates provide the regular throughput report.

The process-local factory adapter is explicit: old parameter/data/source contract
remains unchanged, and `execution_history` adds new backend/adapter source SHAs
and timestamp. Each resume audits actual next-step gradients again. No historical
contract hash is suppressed, edited or waived. Both branches record their own
execution checks; examine them before interpreting between-branch timing.

## Switch the live server job without losing progress

From the user's CURRENT checkout:

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
touch ../engine-core/STOP
```

Wait for the old worker's final notification/log message and process exit BEFORE
starting the next one. If HF pending>0, retain the local states and report. Do not
kill -9 or delete the work directory. Then:

```bash
git pull --ff-only
source ../engine-core-env/bin/activate
set -a; source ../engine-core.env; set +a
unset LD_LIBRARY_PATH
RUN_DIR="$PWD/../engine-core"
nohup python -u -m engine_runtime --engine auto \
  --max-wall-hours 21 --clear-stop --work-dir "$RUN_DIR" \
  >> "$RUN_DIR/runtime.log" 2>&1 &
```

The21h is a NEW invocation allowance; replace with the actual remaining night
budget. Corpus/base are reused, both branch q/Adam/current cursors resume.
`tail -f ../engine-core/runtime.log` shows the audit/selected execution and timing.
STOP, bounds, HF checkpoint cadence, quiet notifications and unrelated GPU
process handling remain as in the original server worker. Default work directory
for this new entry point is beside its checkout, not top-level /data/workdir.

## Evidence

Tiny CPU tests compare resident gradients AND proposed parameters/Adam against
legacy, verify whole-compiled gate and second optimizer update, simulate memory
fallback, and verify explicit adapter provenance/restoration. This is not a
physical Blackwell speed, peak memory or actual running-server resume test.
No server job was stopped or restarted by this local preparation.

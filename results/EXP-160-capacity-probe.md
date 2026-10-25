# Bounded current-state batch capacity probe

New neutral CLI: `python -m engine_capacity --max-minutes 5`.
Default candidates1/2/4/8/16. Uses the current saved one-GPU server160 model,
both frozen branches/resident teacher allocation as in the real experiment,
and the audited whole-step compiled objective. Actual probes use unaccepted
proposals on one reported branch's SAME q/Adam, consecutive C4 inputs, context256.
The current profiled branch is recorded; this is not a quality comparison.
Custom batches can be ANY integer1..16, including6: `--batches 6`.
For fair memory-capacity diagnosis, test larger candidates in a fresh invocation:
the allocator pool/fragmentation from earlier candidates can cause an OOM even
when the same batch fits in a fresh process. Memory guard details are recorded
in memory_diagnostic; do not interpret every memory_limit as physical96GB exhaustion.

## Bounds and behavior

- Default5minutes TOTAL, including imports, restore, source replay/initialization,
  XLA compile, warmup and synchronized timing. Parent watchdog terminates only
  its own isolated probe process group if native compilation exceeds deadline;
  up to5seconds shutdown grace and final notification overhead may follow.
- Three timed proposals after compile/warmup (at least2 needed to rank a batch).
  Record median and aggregate input tokens/s, next-token targets/s, teacher
  input positions separately, losses/finite flags and allocator memory reports.
- Incremental atomic results survive timeout during the next compilation. Select
  fastest FINITE fully timed candidate; within3% of best prefer smaller batch.
  OOM/memory guard stops ascent. It may not finish all sizes within the limit.
  If no candidate completes, report no_completed_measurement, no invented winner.
- Compiled executable cache is shared under WORK/compile-cache to amortize
  subsequent profiling; no teacher targets or model states are stored there.
  20GB free disk reserve checked before launch. One previous JSON retained.
- Own GPU allocator memory and its cumulative high-water mark include preceding
  initialization/probes; they are NOT isolated NVML peaks or proof of100% FLOPs.

## Model safety and scope

The parent acquires the SAME server.lock; it refuses to profile alongside a live
training job. Worker reads local checkpoints with hub=None, so it cannot fetch or
overwrite training-state chunks. The original report is deep-copied in memory,
checkpoint/report synchronization is disabled, and no proposal is accepted.
Original global/data cursors and Adam remain unchanged. Source immutable model
cache may be reused/refilled; original learned states are read-only.

The result is a capacity recommendation, NOT an automatic change to the running
experiment. Larger effective batch changes tokens per Adam update; applying it
requires a separately versioned batch/data-cursor contract. The current batch1
experiment must not silently skip/duplicate tokens or reset optimizer for a speed
number. No long-training quality win is inferred from this benchmark.

## Run at the user's existing nested location

First request the active job's normal save/stop, then WAIT for its process exit:

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
touch ../engine-core/STOP
```

After exit, update and launch the bounded probe (no dependency changes):

```bash
git pull --ff-only
source ../engine-core-env/bin/activate
set -a; source ../engine-core.env; set +a
unset LD_LIBRARY_PATH
RUN_DIR="$PWD/../engine-core"
nohup python -u -m engine_capacity --work-dir "$RUN_DIR" --max-minutes 5 \
  > "$RUN_DIR/capacity-launch.log" 2>&1 &
```

Only one final Telegram notification. `--no-telegram` disables it. Existing
STOP from training is not removed: the read-only probe is separately timeboxed.
For a shorter grid use `--batches 1,2,4`.
`--max-minutes` can be changed; timeout may leave only a partial recommendation.

Files in the existing work directory:
- results/runtime-capacity.json: measurements and recommended_batch.
- runtime-capacity.log: actual worker errors/restore/compile progress.
- capacity-launch.log: final controller output when launched with nohup.

After the probe, the old batch1 run can resume with its existing engine_runtime
command and --clear-stop, unchanged q/Adam/data. No training is started by the
probe itself. Keep original work directory/states even if profiling fails.

## Engineering evidence

CPU tests exercise true batch1/2 compiled proposals, finite mean-loss consistency,
zero mutation of q/Adam, recommendation/near-tie policy, parent compile timeout
with partial results, live-job lock rejection and read-only worker wiring. This
does not measure Blackwell throughput or prove a batched training recipe.

# EXP165 corrected launch

The failed164 process must have exited. Keep its states and results. Update repo; use the same environment and GPU cap. The default full route now starts165 from durable164 CURRENT157685; subsequent invocations resume165.

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
git pull --ff-only
source ../engine-core-env/bin/activate
set -a
source ../engine-core.env
set +a
unset LD_LIBRARY_PATH XLA_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_ALLOCATOR TF_GPU_ALLOCATOR
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
RUN_DIR="$PWD/../engine-core"
HOURS=13
nohup python -u -m engine_launch --runner full \
  --work-dir "$RUN_DIR" --max-wall-hours "$HOURS" \
  --save-reserve-minutes 18 --clear-stop \
  >> "$RUN_DIR/runtime-full-next.log" 2>&1 &
tail -n 40 -f "$RUN_DIR/runtime-full-next.log"
```

Set HOURS to the available time. GPU fraction is an allocator budget with headroom, not a strict NVML process quota. Mandatory batch1..8 profiling repeats. Bot /check automatically switches to runtime-full-next.json. To soft-stop from any terminal:

```bash
touch /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core/STOP
```

Wait for final save/upload and process exit before resume. No global Python kill.

# EXP143-146: launch routing after source-container repair

143/144: use EXP-143-fixed-launch.md / EXP-144-fixed-launch.md instead.
The old cell below is retained for145/146 ONLY; do not interrupt current runs.

Download transport fixed after reported cloud failures. Use the EXISTING T4 notebook.
Each account gets one experiment, two independent one-GPU workers.
Parent Telegram only: start, progress at4h/8h, final; at mostone additional
failure alert, hard maximum5 messages/invocation. Child events stay in logs/JSON.
Notebook setup messages are outside this runner's cap.

```python
import importlib
import os
import subprocess
import sys
import time

EXP = 145  #145 or146 ONLY;143/144 require the separate fixed runner.
repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()
# This limit INCLUDES download waits; keep the saving reserve free.
WALL_HOURS = 11.7
RESERVE_MINUTES = 30
os.environ["EXTENT_HF_DOWNLOAD_DEADLINE_UNIX"] = str(
    time.time() + WALL_HOURS * 3600 - RESERVE_MINUTES * 60
)
os.environ["EXTENT_HF_SAVE_DEADLINE_UNIX"] = str(time.time() + WALL_HOURS * 3600 - 60)
os.environ["EXTENT_HF_RETRY_STATE_DIR"] = "/kaggle/working/extent-hf-download"
from extent.hf_download_retry import install_download_retries
install_download_retries()
for module in (
    "extent.reset_long_patient",
    "extent.reset_long_recovery",
    "extent.reset_precision_probe",
    "extent.reset_quiet",
    "scripts.m3q_reset_long_campaign",
):
    if module in sys.modules:
        importlib.reload(sys.modules[module])
from scripts.m3q_reset_long_campaign import main

result = main([
    "--experiment", str(EXP),
    "--max-wall-hours", str(WALL_HOURS),
    "--save-reserve-minutes", str(RESERVE_MINUTES),
    "--max-steps", "131072",
    "--max-new-hf-gb", "4",
])
```

11.7h INCLUDES30min reserve;18min outside runner for notebook setup. Startup,
source restore, compilation and evaluations consume runner time. Train until work
deadline (11.2h), not until an arbitrary small step limit.131072 is a large ceiling
in NEW updates; not a promise to reach it in12h. After8 new accepted updates the
worker log/JSON reports recent throughput and ETA, without a Telegram message.

Download ONE aggregate: extent-results/extent-m3q-long-ID.json. Child exceptions
and log tails are included there and printed in Kaggle output. arm_failed is a
visible parent failure, even if the child exited0. deadline_partial is a saved
time stop. Same cell resumes OWN long-v1 state, not a new start on every session.
Source143/144 Adam/data clocks are inherited;145 deliberately resets Adam BOTH
arms. Old experiments and their files/contracts unchanged. Local128-update saves,
remote120min+exit.4GB new HF cap TOTAL/session (2GB perworker), no base model upload.
The last unsynced tail can be lost if the cloud VM is abruptly destroyed.

BF16 training remains unchanged, FP32 correction/Adam/recurrence as before.
Worker0 records a short synthetic BF16/FP16/FP32 GEMM-forward benchmark with JIT
warmup and block_until_ready. Its ratio is NOT full-model/gradient speedup and does
not switch training dtype. Benchmark failure is recorded and does not reject
otherwise valid training. No physical new-run GPU result exists yet.

18 October download fix: HF429 waits Retry-After (or300s if absent) plus random
1..60s. Other transient HTTP/network failures use exponential backoff. Retry
until success or the work deadline, with waits split into <=30s. Both workers
on ONE VM share request pacing (2s minimum) and a cooldown; separate Kaggle VMs
use independent jitter, not a fictitious cross-VM lock. This may take longer to
restore sources. Partial local files remain reusable in the same runtime.
401/403/404 and integrity errors keep their original semantics. Per-retry events
are printed only in worker logs, without Telegram. No experiment contract, pinned
source, numerical kernel or optimizer was changed. Resume uses the same long-v1
prefix. This is a transport fix, not a new experiment or a scientific result.
If resuming inside an already used VM, set WALL_HOURS to the remaining runtime
budget (including the >=30min reserve), not to a fresh11.7h.

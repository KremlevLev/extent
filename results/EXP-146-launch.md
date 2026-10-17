# EXP-146: Prepared model seed456: full versusprojection reset

# EXP143-146: four long T4x2 sessions

PREPARED; no new cloud training launched. Use the EXISTING T4 notebook.
Each account gets one experiment, two independent one-GPU workers.
Parent Telegram only: start, progress at4h/8h, final; at mostone additional
failure alert, hard maximum5 messages/invocation. Child events stay in logs/JSON.
Notebook setup messages are outside this runner's cap.

```python
import importlib
import os
import subprocess
import sys

EXP = 146
repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()
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
    "--max-wall-hours", "11.7",
    "--save-reserve-minutes", "30",
    "--max-steps", "131072",
    "--max-new-hf-gb", "4",
])
```

11.7h INCLUDES30min reserve;18min outside runner for notebook setup. Startup,
source restore, compilation and evaluations consume runner time. Train until work
deadline (11.2h), not until an arbitrary small step limit.131072 is a large ceiling
in NEW updates; not a promise to reach it in12h. After8 new accepted updates the
worker log/JSON reports recent throughput and ETA, without a Telegram message.

Download ONE aggregate: extent-results/extent-m3q-long-146.json. Child exceptions
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

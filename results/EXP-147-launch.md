# EXP-147:7h T4x2 session

GPU0:145 current24965, originalLR; GPU1:same source, ALL LR groups/3.
Use the EXISTING T4 notebook; replace only the last cell. Account1 EXP147,
account2 EXP148. Each parent starts two independent one-GPU workers.

```python
import importlib
import os
import subprocess
import sys
import time

EXP = 147
WALL_HOURS = 6.8  # Total runner budget, including saving; account has7h.
RESERVE_MINUTES = 30
repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()
os.environ["EXTENT_HF_DOWNLOAD_DEADLINE_UNIX"] = str(
    time.time() + WALL_HOURS * 3600 - RESERVE_MINUTES * 60
)
os.environ["EXTENT_HF_SAVE_DEADLINE_UNIX"] = str(time.time() + WALL_HOURS * 3600 - 60)
os.environ["EXTENT_HF_RETRY_STATE_DIR"] = "/kaggle/working/extent-hf-download"
from extent.hf_download_retry import install_download_retries
install_download_retries()
for name in ("extent.fold_continue_recovery", "scripts.m3q_fold_continue_campaign"):
    if name in sys.modules:
        importlib.reload(sys.modules[name])
from scripts.m3q_fold_continue_campaign import main
result = main([
    "--experiment", str(EXP),
    "--max-wall-hours", str(WALL_HOURS),
    "--save-reserve-minutes", str(RESERVE_MINUTES),
    "--max-steps", "65536",
    "--max-new-hf-gb", "4",
])
```

Total runner6.8h INCLUDING30min reserve;12min outside for notebook setup.
Source loading, JIT, evaluation consume work budget6.3h.65536 is a HIGH ceiling
in ADDITIONAL updates, not a short stop target or promise to reach it.
Existing throughput145~2.8s/update suggests several thousand more updates;
actual count depends on download/JIT costs. Recent throughput/ETA is logged
once8 new updates are accepted. At8192 goal, continue within remaining budget.

All source tensor chunks checked SHA; all160 source A/B/C windows must replay
within.03 BEFORE first update. Exact continuations restore145 Adam count12379
with GLOBAL data cursor24965; do not reset optimizer/warmup or fold again.
Control deliberately starts freshAdam0, old136 factors/data cursor12586.
Empty frozen GQA entries6/13/20/27 retained. No scientific change to running143/144.

HF retries honor429 + jitter with shared local cooldown; source waits obey
work deadline, save-verification reads use separate wall-end-60s deadline.
Local saves every128 accepted NEW updates, HF every120min+exit.4GB cap TOTAL
session(2GB/worker); frozen base reconstructed, never uploaded. Separate prefix
experiments/exp147-fold-continue-v1/worker-W; same cell resumes OWN state.
Telegram parent-only: normally start/4h/final, max5 including failure.

Download ONE aggregate: extent-results/extent-m3q-fold-continue-147.json.
Do not use old145 launcher, which still contains the failed control path.
EXP148 arms start from DIFFERENT checkpoints: compare its control to archived145
at matched new-step counts, its LR/10 arm to147 continuations at matched new-step
counts. Do not rank both148 arms by their last NLL as an ablation.

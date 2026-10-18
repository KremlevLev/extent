# EXP-143: corrected source restoration

Use the EXISTING T4 notebook. Applies to143/144 only. Leave running145/146 alone.
Two independent1GPU workers;11.7h includes30min save reserve. Set WALL_HOURS to
the remaining budget if reusing a runtime. Same scientific sources, recipes,
Adam moments/counts and data cursor. The earlier143/144 attempts accepted ZERO
updates and produced no new NLL measurement.

```python
import importlib
import os
import subprocess
import sys
import time

EXP = 143  # Repaired runner accepts143 or144 ONLY; leave145/146 running.
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
    "extent.reset_long_fixed_sources",
    "extent.reset_long_patient",
    "extent.reset_long_recovery",
    "extent.reset_precision_probe",
    "extent.reset_quiet",
    "scripts.m3q_reset_long_fixed_campaign",
):
    if module in sys.modules:
        importlib.reload(sys.modules[module])
from scripts.m3q_reset_long_fixed_campaign import main

result = main([
    "--experiment", str(EXP),
    "--max-wall-hours", str(WALL_HOURS),
    "--save-reserve-minutes", str(RESERVE_MINUTES),
    "--max-steps", "131072",
    "--max-new-hf-gb", "4",
])
```

Expected first signals: source_replay.passed=true in worker JSON, then accepted
updates/step above source12586+12051(143) or11965(144). Source replay and all
integrity/finite/HBM checks stay active. A successful download alone is not a
successful experiment. Exact physical T4 recovery remains to be verified.

Download ONE aggregate: extent-results/extent-m3q-long-143.json.
Worker files use extent-m3q-long-fixed-143-worker-W.json; remote prefix
experiments/exp143-long-fixed-v2/worker-W. Old long-v1 states remain intact.
The same new cell resumes its OWN fixed-v2 checkpoints. Missing EMPTY frozen
attention entries are reconstructed before initializing the restore template;
all tensor leaves and optimizer state come from the immutable original source.
Missing trainable Mamba entries are an error. Existing145/146 code/contracts are
unchanged. New CPU tests exercise real save/restore, NLL, profiles, streamed
updates and own-checkpoint resume; CPU checks do not prove full-size GPU success.

HF download retry remains enabled:429 Retry-After or300s fallback plus jitter,
shared pacing/cooldown for two workers on one VM; separate VMs use jitter only.
Waits stop at the work deadline to preserve saving time. Retry events log only.
Parent Telegram stays capped at5 messages, normally start,4h,8h,final.

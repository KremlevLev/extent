# EXP-142: full-decoder recovery from EXP-130, TPUv5e8

PREPARED: no new cloud result. Existing TPU notebook; replace only its last cell.
Source is EXP130 current3434, not best3072. New Adam and new training cursor0.

```python
import importlib
import os
import subprocess
import sys

repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()
for module in (
    "extent.full_decoder_reset",
    "scripts.m3q_full_decoder_142_campaign",
):
    if module in sys.modules:
        importlib.reload(sys.modules[module])
from scripts.m3q_full_decoder_142_campaign import main

result = main([
    "--experiment", "142",
    "--max-wall-hours", "8.5",
    "--save-reserve-minutes", "60",
    "--max-steps", "131072",
    "--max-new-hf-gb", "80",
])
```

8.5h includes1h save/final-evaluation reserve;30min outside runner for notebook setup.
The ceiling131072 does not promise this many steps in one session. Training continues
until the work deadline; source setup/evaluation/compilation also consume that budget.
After8 accepted updates the runner reports throughput and an approximate ETA.

Download ONE report: `/kaggle/working/extent-results/extent-m3q-reset-142.json`.
Remote prefix: `experiments/exp142-reset`. Same cell resumes exact saved142 weights,
Adam and data cursor under the same code/runtime contract. `deadline_partial` is a
wall stop, not convergence. `arm_failed` means a proposal was rejected; inspect health.
Local checkpoint every512 updates; remote every4h and exit. Sudden cloud termination
before sync loses unsynced work; this is the cost of reducing large HF uploads.
80GB is the invocation upload cap, not required allocation or a promise to use it.
Do not rerun a TPU setup or install GPU dependencies inside a running TPU kernel.

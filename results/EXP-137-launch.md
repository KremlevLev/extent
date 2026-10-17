# EXP135-138: corrected T4 GPU execution, streamed-v2

Replace the LAST cell in the existing T4 notebook. No new notebook/setup required.
Fresh cloud session recommended after a failed worker. TPU cells139-142 unchanged.
Choose the correct EXP on each account. Code preserves recipes/initial states;
changes numerical execution, records bounded parity, not identical trajectories.
Old failed reports/state are read-only and not reused as successful training.

```python
import importlib
import os
import subprocess
import sys

EXP = 137
repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
importlib.invalidate_caches()
for module in ("extent.reset_streamed_gpu", "scripts.m3q_reset_t4_streamed_campaign"):
    if module in sys.modules:
        importlib.reload(sys.modules[module])
from scripts.m3q_reset_t4_streamed_campaign import main

result = main([
    "--experiment", str(EXP),
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-steps", "131072",
    "--max-new-hf-gb", "6" if EXP == 135 else "4",
])
```

135 uses BOTH T4 for model-parallel layer executables and ONE optimizer trajectory.
136-138 use two independent single-GPU workers, as before. Hidden boundaries live
in host RAM between backward layers, preserving the full end-to-end chain rule.
No full28-layer reverse executable. Existing GPU memory guard remains enabled.
The new135 first-step gradient must also pass its legacy replay gate before update.

11.5h includes1h reserve; outside30min for setup. Firstgoal32768, ceiling131072;
no assurance these fit one session. After8 accepted steps: measured throughput/ETA.
Same new cell resumes its OWN streamed-v2 weights/Adam/cursor. It does NOT import
failed v1 checkpoints or any unsupported hypothetical progress. If an old job is
still running, do not start its duplicate. Checkpoints and caps unchanged.

One file per experiment: extent-m3q-reset-streamed-v2-ID.json in extent-results.
HF prefix experiments/expID-reset-streamed-v2. On child failure, the parent now
prints the original report exception and last16k log characters into Kaggle output;
these are also in the aggregate JSON. No need to download many worker files.
Full1.7B physical GPU memory/throughput remain to be confirmed by the actual run.

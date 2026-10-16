# EXP133: Continue126 best: LR and MLP

PREPARED. Full protocol: [EXP-130-134-protocol.md](EXP-130-134-protocol.md).

Existing TPU notebook LAST cell. git_env from deploy-key clone cell.
Runner wall3.6h INCLUDES reserve45min; new HF cap4GB total.

```python
import os
import subprocess
import sys

repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)

from scripts.m3q_peer_tpu_campaign import main
result = main([
    "--experiment", "133",
    "--max-wall-hours", "3.6",
    "--save-reserve-minutes", "45",
    "--max-new-hf-gb", "4",
])
```

Send ONE file: /kaggle/working/extent-results/extent-m3q-peer-133.json.
Same cell resumes; compare durable cursors and registered common endpoints.

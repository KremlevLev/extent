# EXP134: Dense Mamba CE versus KL

PREPARED. Full protocol: [EXP-130-134-protocol.md](EXP-130-134-protocol.md).

Existing TPU notebook LAST cell. git_env from deploy-key clone cell.
Runner wall8.5h INCLUDES reserve60min; new HF cap24GB total.
Dense state about8.45GB per paired save, about17GB for two uploads; no frozen full-model copies.

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
    "--experiment", "134",
    "--max-wall-hours", "8.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "24",
])
```

Send ONE file: /kaggle/working/extent-results/extent-m3q-peer-134.json.
Same cell resumes; compare durable cursors and registered common endpoints.

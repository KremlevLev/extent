# EXP-115: seven variants, one seed,9-hour account

Use the existing clone/install/HF/Telegram setup in a fresh notebook kernel.
Update main before importing the runner. Select TPUv5e8, Internet on.
No second JAX subprocess; run in the notebook process.

```python
%cd /kaggle/working/extent
!git pull --ff-only

from scripts.m3q_rank_growth_115_campaign import main as run_exp115

result = run_exp115([
    "--max-wall-hours", "8.8",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "6",
])
```

One seed123. R32/G64/G128/G256/ADD256/FAST256/LADDER256 each4096 updates;
top2 by validation plus mandatory R32 continue to16384. Same durable EXP111
start, old Adam retained, added cohorts warm up separately. Locked Wiki/PG19
after selection, no additional seed training in this run.

8.8h includes one hour for save;12min outside for setup. If setup exceeds that,
reduce wall-hours. Rank256 TPU throughput is unmeasured; deadline_partial can
need continuation. Resume with this same cell, retaining branch moments/cursors.
Reports include discarded pilot/replay costs. Finalized pilots are not rerun when
their state is available. Do not run another experiment in this same runtime.

HF prefix `experiments/exp115-rank-growth`. Source model weights are not copied
back to HF.6GB is a per-invocation new-upload cap, not a claim of remaining quota.
Periodic reports are durable before most model snapshots; an external kill can
lose local updates. Final notifications distinguish saved versus HF pending.

Send these files from `/kaggle/working/output`:

- `extent-m3q-rank-growth-115.json`
- `extent-m3q-rank-growth-115-summary.txt`

If uploads remain pending and the original runtime is alive:

```python
result = run_exp115([
    "--sync-only",
    "--max-wall-hours", "0.9",
    "--save-reserve-minutes", "15",
    "--max-new-hf-gb", "6",
])
```

Choose sync-only wall budget from actual remaining time. It cannot recover files
from an already destroyed VM. No actual TPU115 result is asserted by local tests.

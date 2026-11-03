# EXP171 TPU launch

Use [notebooks/extent-tpu-171.ipynb](../notebooks/extent-tpu-171.ipynb) in a FRESH
TPU v5e-8 session, Internet enabled. Secrets: EXTENT_DEPLOY_KEY_B64,HF_TOKEN,
TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID. The notebook parent must not initialize
JAX/Torch devices. Dedicated requirements-tpu171.txt replaces the old setup;
changing only the final cell of an already initialized169 session is insufficient.

The notebook includes private clone, dependencies/secrets and final launch. Last
cell controls TOTAL_HOURS=9 and PREFLIGHT_ONLY=False. Set True for speed/memory
only with zero accepted training updates. All preparation counts against9h.

If the fresh notebook already cloned the repo and installed requirements-tpu171:

```python
import os, sys, subprocess, time
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0')
os.chdir('/kaggle/working/extent')
TOTAL_HOURS = 9
remaining = TOTAL_HOURS - (time.monotonic() - SESSION_STARTED) / 3600
if remaining < 1:
    raise RuntimeError('Less than one hour remains')
completed = subprocess.run([
    sys.executable, '-u', '-m', 'engine_tpu171',
    '--max-wall-hours', str(remaining),
    '--work-dir', '/dev/shm/extent171',
    '--output-dir', '/kaggle/working/extent-results',
], check=False)
```

Main file to return: **extent-results/extent-m3q-tpu171.json**. It contains selected
batch, measured complete-update speed, probe gates, training trajectory and NLL.
If failure: also return prepare171.log or probe171-bN.log / train171.log matching
last_phase. Logs stay on notebook output, not only ephemeral tmpfs. There is no
need to download model chunks manually.

The whole9h job can resume its own synced state by rerunning this fresh notebook.
It targets100,007,936 accepted inputs; context1024/effective16 is intentionally
different from the GPU170 contract. Read [protocol and limits](EXP-171-protocol.md).

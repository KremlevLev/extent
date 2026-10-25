# EXP160 on one RTX PRO6000 Blackwell96GB (Linux, Python3.12)

Prepared25October2026. User-reported driver595.99.02/CUDA13.2; existing GPU
processes use~3.8GiB. No server connection or physical GPU run has been performed
by this preparation. Do not stop unrelated inference processes or modify /opt/venv.

## One-time setup in /data/workdir

Use a separate checkout and virtual environment. If this checkout already exists there,
update that checkout with `git pull --ff-only` instead of cloning a second copy.

```bash
cd /data/workdir
git clone https://github.com/KremlevLev/extent.git engine-core-src
cd engine-core-src
python3 -m venv /data/workdir/engine-core-env
source /data/workdir/engine-core-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-blackwell.txt
```

Pinned JAX/Flax/Optax match source155. CUDA13 wheels are installed in this venv;
the existing system CUDA toolkit/services are not altered. Official installation:
https://docs.jax.dev/en/latest/installation.html#nvidia-gpu
Source replay remains mandatory on Blackwell; a different GPU backend is not
assumed bitwise identical. Refuse updates if the existing .03 window gate fails.

Copy the empty secrets template OUTSIDE the repo and fill it privately:

```bash
cp config/engine-core.env.example /data/workdir/engine-core.env
chmod 600 /data/workdir/engine-core.env
nano /data/workdir/engine-core.env
set -a
source /data/workdir/engine-core.env
set +a
```

Required HF_TOKEN (read/write existing dataset); Telegram bot token and CHAT_ID.
The server also accepts TELEGRAM_USER_ID as alias. Never paste actual tokens
into Git, command arguments, issue descriptions or logs. Same notifier as Kaggle.

## Run, change hours, and leave terminal

```bash
cd /data/workdir/engine-core-src
source /data/workdir/engine-core-env/bin/activate
set -a; source /data/workdir/engine-core.env; set +a
unset LD_LIBRARY_PATH
python -u -m engine_worker \
  --max-wall-hours 21 \
  --work-dir /data/workdir/engine-core
```

For an SSH terminal, run the command inside `tmux new -s engine-core`, then detach
with Ctrl+B, D. Reattach with `tmux attach -t engine-core`.
`unset LD_LIBRARY_PATH` affects this shell/children, not running services; avoids
mixing system CUDA libraries with venv pip CUDA wheels.

Change21 to the desired TOTAL invocation hours. Includes corpus/source downloads,
compilation, model replay, evaluation, saves and an18min final reserve. Re-running
with another hour limit resumes the same own checkpoint/Adam/data cursors.
Preparation builds the registered100M C4 once if not already in the work dir,
without downloading full C4 or model-weight duplicates for every branch.
Optional: copy the already prepared local corpus into
`/data/workdir/engine-core/data-pack` before launch to skip preparation.

The original kernel is BF16 storage/compute with FP32 head/corrections/states as
before; Blackwell supports native BF16. No precision conversion or full-decoder
intervention is hidden in this port. It remains protectedrank256 output KD.

## What executes

One process, one visible GPU. Original Qwen teacher stays resident on GPU;
both hybrid branches' weights/optimizers stay resident. Current backward engine
still streams saved activations through host; this is a first correctness port,
not a fused Blackwell throughput benchmark. No claim that JAX beats PyTorch.
After each branch's first8 updates, record median update seconds and input
tokens/s (teacher token cost separate). Final report records device memory stats.

Wiki control and C4 intervention take turns, choosing the least advanced branch.
First8 updates per branch then blocks up to256; end counts may differ by up to
one block. Compare common registered endpoints. A single GPU shares compute
between the two branches; it does not train both at full-card speed simultaneously.
No full training job is launched by writing these instructions.

## Stop cleanly in the morning

From another terminal:

```bash
touch /data/workdir/engine-core/STOP
```

Ctrl+C or SIGTERM directed to THIS process also requests the same soft stop.
Allow the save to finish before killing the process/shutting down. A running
download/compile must return to its next check first; SIGKILL/power loss cannot
perform a final upload. Previously saved local state remains resumable.

Resume after a requested stop:

```bash
python -u -m engine_worker --max-wall-hours 10 \
  --work-dir /data/workdir/engine-core --clear-stop
```

## Storage and notifications

One work directory, one shared Qwen/source cache, one400MB C4 train file. No full
model/teacher output caches per step. Adapter+Adam+best snapshots only. Atomic
current generation, obsolete local generations pruned by the existing safe store.
Local save at most once per branch per15min plus final; HF sync every4h plus
final,32GB upload cap for the entire server invocation (flag --max-new-hf-gb).
Default work-directory ceiling80GB and free disk reserve20GB; a budget breach
requests save/stop, never deletes unrelated files. Caches are reused next night,
not mirrored again. Environment checkout/venv are outside this work-directory
counter; no claim that the counter covers the whole server disk.

Only start and finish Telegram notices. Final includes status and pending HF
states; pending>0 means NOT fully synced. Network/quota failure does not erase
local checkpoints. Do not treat a completion message as a capability result.

Report to send:
`/data/workdir/engine-core/results/runtime-report.json`

Own namespace `experiments/exp160-corpus-server-v1`; T4-pair/TPU artifacts are not
overwritten. Horizon1b NEW updates avoids a short arbitrary step stop, but C4
contains390625 windows: exhausting100M C4 positions stops explicitly, never
repeats. Whether this or time is reached first depends on measured throughput.

Morning update: `git pull --ff-only`, activate same venv, launch with chosen hours.
Scientific code/corpus identities are strict; a changed scientific contract
needs an explicit migration/new experiment, not silent optimizer reuse.

Neutral operational names: `python -m engine_worker`, `engine-core` work dir,
`runtime-report.json`, `runtime-status.json`; internal experiment provenance and
HF source identities are retained. No process title spoofing or service changes.

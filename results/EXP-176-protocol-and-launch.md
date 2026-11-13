# EXP176 · Weekend PyTorch full-CE continuation

Prepared14 November2026. The user offered the company RTX PRO6000 server
until Monday16 November09:00 Moscow time. The server has approximately
500GB host RAM and80GB free SSD. Retain at least30GiB SSD headroom.
No physical EXP176 CUDA measurement or training result exists yet.

## Scientific continuation

Continue the selected CE checkpoint from EXP175 at step511. It has
8,372,224 CE-trajectory input presentations after source171, with
128-window validation NLL2.539805. Its full parameters and Adam are pinned
at HF dataset revision4bb81a483398a2a944a519507bb8df6d9c13d0be,
generation step-511-9e08133cf6934ce4a0788601f6fd5101.
[Source descriptor and window references](EXP-176-source-reference.json).

Train all active weights of the seven-attention/21-Mamba hybrid with
next-token CE, constant LR1e-6, AdamW(.9,.95),eps1e-8,weight decay0 and
global clip1. Preserve source175 Adam moments and update count511.
No new warmup, rank change or layer-selection experiment is introduced.
FP32 master parameters and optimizer states, BF16 projection GEMMs and
FP32 state algebra. PyTorch CE runs only the student. Prior JAX CE pilots
also computed teacher outputs for diagnostic KL, so a throughput difference
would include this work reduction.

Input geometry remains context1024/effective16. Continue verified Edu600M
at cursor92,258,304, leaving507,741,696 input positions before rounding
to complete effective updates. No cycling. At a sustained1750 input tok/s,
Friday23:16 through Monday09:00 is57h44m and363.72M positions before setup,
planning, evaluation and saving overhead. PyTorch throughput is unmeasured.
If sustained throughput is above roughly2443tok/s for that entire window,
the remaining corpus can finish before Monday. Exhaustion ends normally.

## Port and engineering gates

The existing torch170 runner implements local alignment. EXP176 uses a
separate composed full-student training path. The bridge maps all active
original-Qwen and Mamba tensors plus both Adam moments, with exact shape
and coverage checks. Unused mixer/attention slots are omitted. Final norm
and the tied embedding/head are included. A CPU regression loads an actual
small JAX chunked checkpoint, checks mapped tensors and moments, and
compares the next Adam update to the bias-corrected formula.

Explicit RMSNorm wrappers preserve the BF16 return boundary with FP32 master
weights. EOS is a real packed token, so input embedding padding suppression
is disabled. The block-recompute function receives each concrete module
explicitly, preserving backward layer ownership.
Bounded first-update weight samples compare physically materialized BF16
storage bits on CPU, independently of accelerator comparison fusion.
These are samples of the new run and do not retroactively validate175's counter.

Before CUDA work, a small pinned-commit HF upload/download verifies transport.
Real CUDA probes compare a learned Mamba layer's outputs and parameter
gradients against the independent recurrence on128 tokens. Source replay
on four windows has mean tolerance0.02 and maximum per-window tolerance0.08.
The training worker also checks the complete128-window initial NLL against
the received175 mean within0.02. This is a numerical portability gate.
It does not establish equality of downstream capabilities.

Disposable workers try physical batches1,2,...,8 and measure three discarded
complete effective updates, with the last two used for throughput. Test
chunked compiled and eager kernels, plus the pinned official MIMO backend
if already installed. Optional fused dependencies are not installed by the
launcher. Keep passed candidates after another process fails. Select the
fastest measured fit. Logs and failed reports remain on SSD. A cached plan
is rechecked in one disposable worker on restart before reuse.

Reports contain full-update wall speed, CUDA forward/backward/clip/optimizer
times, peak allocated/reserved VRAM, validation and accepted input counts.
Evaluate128 Edu windows every512 updates and at the final normal stop.
No downstream task score is measured here. No NLL-based early stop.

## Storage and interruption

Heavy artifacts live in an owner-marked `/dev/shm/extent176-<uuid>` workspace.
Initial source restoration needs160GiB available host RAM and120GiB free
tmpfs. Restart needs80GiB available host RAM. During work, host RAM headroom
below48GiB or owned RAM files above120GiB requests a normal stop.
Model import chunks are discarded after the converted CURRENT is verified.
The PyTorch checkpoint contains only active parameters and optimizer state.
One atomic CURRENT is replaced through one temporary generation.

CUDA allocation is capped at the smaller of90GiB and currently free VRAM
minus5GiB. This is a PyTorch allocator cap, and unrelated processes can
change available memory after measurement.

Local CURRENT is saved after the first accepted update and every30 minutes.
HF uploads use a stable hard-linked snapshot while training can continue
updating CURRENT. Checkpoint payload budget64GiB reserves room for the
final snapshot. This typically permits one intermediate upload after24h
and the final snapshot. The HF API reports repository usedStorage about
4.71TB, so previous assumptions about a1.87TB repository are stale.
Budgets persist across fast pauses and restarts.

`FAST_STOP` finishes the current effective update or interrupts validation,
saves CURRENT in RAM, exits the CUDA worker and cancels any unfinished
periodic upload process. It retains the workspace for restart. It starts
no new final HF upload. The ready notification follows observed worker exit.
Completion time is unmeasured on the actual server.

`STOP` finishes normally, evaluates, saves and uploads. The until-deadline
reserves18 minutes for final evaluation/upload/cleanup. CPU-only periodic
uploads are isolated from the GPU process. A hard wall guard controls
the launcher-owned GPU process group.

After normal completion the latest CURRENT SHA must equal the remotely
verified descriptor before deleting this run's owned RAM workspace.
On failed upload or mismatch, local recovery files remain. Source code,
the reusable venv and small reports/logs stay on SSD. Unknown directories
and older experiments are outside this cleanup. Prior reports are archived,
and resumed reports retain history only through the restored checkpoint.

## Install and launch

Run inside the existing repository on the server.

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
git pull --ff-only

python3 -c 'import shutil; free=shutil.disk_usage(".").free; print("Free SSD GiB", round(free/2**30,1)); assert free >= 50*2**30, "Need50GiB free before creating the CUDA venv"'
python3 -m venv ../torch176-env
source ../torch176-env/bin/activate
python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -r requirements-torch176.txt

set -a
source ../engine-core.env
set +a

RUN_DIR="$PWD/../extent176"
mkdir -p "$RUN_DIR"
nohup python -u -m engine_torch176 \
  --work-dir "$RUN_DIR" \
  --until "2026-11-16T09:00:00+03:00" \
  --memory-gib 90 \
  --min-free-gib 30 \
  --max-new-hf-gib 64 \
  --save-reserve-minutes 18 \
  --clear-stop \
  >> "$RUN_DIR/launcher.log" 2>&1 &

tail -n 60 -f "$RUN_DIR/launcher.log"
```

The preflight message gives measured speed and batch before long training.
To perform only the CUDA check, add `--preflight-only`. That retains the
prepared RAM CURRENT for the normal launch.

Fast pause from any terminal, with no dependence on a previously set variable.

```bash
touch /data/workdir/_biometric-old/biometric-linking-guarded-v4/extent176/FAST_STOP
```

Normal stop with final HF sync and cleanup.

```bash
touch /data/workdir/_biometric-old/biometric-linking-guarded-v4/extent176/STOP
```

Restart with the same command and `--clear-stop`. Auto-restore prefers the
local RAM CURRENT and otherwise checks the canonical EXP176 HF descriptor.
Initial EXP175 import is used when no EXP176 continuation exists.
Automatic `/check` registration preserves the existing session registry.
Only the known listener for this bot is restarted to load the updated formatter.
Telegram failure is reported separately and does not block training.

Send `extent176/results/runtime-torch176.json` after training, and
`runtime-torch176-launch.json` for planner or engineering failures.

## Local verification

32 CPU tests passed. They cover source-tensor and Adam mapping from a real
small JAX checkpoint, the next Adam step, a28-layer backward with recomputation,
BF16 norm boundaries and host storage-bit sampling, atomic saves, immutable
upload snapshots, mismatch/unknown-root cleanup refusal, fast-pause/failure
retention, normal final cleanup, bot formatting and known-listener restart.
Existing torch170 numerical and remote-cleanup tests still pass. CUDA kernel
gates, actual server throughput and multi-hour behavior remain unmeasured.

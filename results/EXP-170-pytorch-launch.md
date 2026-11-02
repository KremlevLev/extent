# EXP170 — clean Q/K/V/O preparation on PyTorch

Implemented 2 November 2026. Local CPU numerical/integration checks are NOT a GPU
throughput result. No training launched by the assistant. This is stage1 hidden
alignment, not a claim of restored language quality or a finished stage2 KD run.

## What is trained

Start from `Qwen/Qwen3-1.7B-Base` pinned at
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. No Patient, EXP069/072, legacy
adapter, optimizer, or recovered student is loaded. Each query retains its own
projection and Q norm; B receives K and K norm, X receives grouped V, output O
is copied. Full H16/G8/P128/N128/R4 geometry, no averaging, cropping or tiling
of feature widths. Extra rank slots repeat the two queries per KV group.
One-hot output rank selects each query's own slot. Every copied projection is
asserted elementwise equal to its donor and donor tensor hashes are recorded.

Random Z is explicitly retained as an unresolved gate-initialization choice.
Zero token-dynamic projections, bias1, simplex input rank factors, one-hot output
factors, full pre-gate RMSNorm and tanh angle dynamics. This preserves projection
ownership, not softmax attention outputs. Attribution:
[arianraje's pinned implementation](https://huggingface.co/arianraje/qwen3-4b-mamba3-hybrid-stage2b-kd-bias1/blob/b946069890837bbe9b5dd75e2d118a7e98d2d193/modeling_qwen3_mamba3.py).

Train ALL parameters of27 replacement mixers, not rank32/256 adapters. Teacher
embeddings,28 original decoder layers and final norm are frozen and shared.
Teacher-prefix block outputs are targets; the student block includes the frozen
original residual/MLP so gradients propagate through that MLP to its new mixer.
One original teacher pass per microbatch, immediate per-layer backward; no stored
hidden-state dataset and no27 full teacher copies. A bank has798,316,128 trainable
coordinates; actual composed model counts are also recorded. The resulting
hybrids are larger than the original1.7B. Do not compare as equal parameter size.

Primary input: existing verified `data-pack-edu-600m`. First100,007,936 input
presentations =1526 updates x16 sequences x4096. No epochs/cycling. Existing pack
sequence256 only concerns file layout; the continuous token stream, including
EOS, is regrouped into4096. Validation uses its separate original validation.bin.
The pure-Edu data choice is not an exact reproduction of the peer's mixed corpus.

AdamW(.9,.95),eps1e-8,wd0, peak1e-3,46-update warmup,cosine to1e-4, globalclip1.
Loss: per-sequence target-energy-normalized full-block MSE, averaged over27
layers. This reduction makes6+6+4 microbatches equal to16, without padding or a
change of effective batch. Mixer-only MSE and hidden scales are logged as well.
100M shared teacher inputs mean about2.7B layer-input presentations, not27 fresh
100M text corpora. Planner and profiler presentations are separate engineering
cost, not accepted training tokens. There is no matched random-control run in
this launcher; it therefore cannot establish same-budget superiority to random.

## GPU execution and measurements

Candidates, same operator/parameters: blockwise SSD in ordinary PyTorch
GEMMs with activation recomputation, the same block compiled with torch.compile,
and OPTIONAL official TileLang/Triton MIMO
with chunk8/16. The first avoids per-token Python iteration and H100-specific
shared-memory requirements. Independent FP32 recurrence and historical JAX
checks cover values and gradients with nonzero rotations and cross-chunk carry.
The actual CUDA path must additionally pass real-donor BF16 output/gradient
gates; it never silently switches to a token-by-token training loop.

The [upstream issue for RTX PRO6000 Blackwell](https://github.com/state-spaces/mamba/issues/1044)
reports an unlaunchable MIMO backward at some shapes. Hence upstream fused
installation is optional, not an assumed performance fix. No architecture is
silently downgraded to SISO or reduced head width.

Mandatory isolated physical batches1,2,...,8 (or first failure/time cap).
Each probe loads original Qwen, initializes the same bank, passes the CUDA
numerical gate, then performs3 DISCARDED complete effective updates: first warmup
and Adam allocation, two timed samples. All27 student backwards, frozen MLP
derivatives and teacher pass are included. Select maximum measured input tok/s,
not largest allocation. The parent captures OOM, process crashes, malformed
results and timeouts; completed fits survive a later failing probe. It does not
claim immunity to host/GPU failure.25min shared planner budget by default.

The mandatory planner is bounded and may not finish all8 candidates. Read the
planner reports for the tested boundary. If no verified fit completes, it stops
before scientific training and retains logs. A shorter context/budget for an
engineering smoke test is a DIFFERENT contract; use a different work directory
and --no-hf (enforced to protect the canonical remote namespace).

Outputs: per-update teacher/student forward/backward/optimizer CUDA times,
whole-effective-update input tok/s, whole-worker accepted tok/s including its
loading/evaluations/local saves, per-process peak allocated/reserved VRAM,
5-second GPU utilization/VRAM/power CSV. Parent planner/HF overhead remains
separate. `--profile` additionally writes a bounded actual-context layer0 CUDA
trace/operator table without updating parameters. Profiler FLOPs omit some
operators; no unmeasured MFU percentage is asserted.

At source/every128 updates/final when time allows: teacher and full composed
gqa4/gqa7 NLL on the SAME8 held-out Edu4096 windows. These are explicitly named
Edu-validation NLL, never historic Wiki A/B/C or peer WikiText2 PPL. gqa4 retains
6/13/20/27; gqa7 retains3/7/11/15/19/23/27. Report last and full trajectory;
local alignment improvement alone does not establish a successful conversion.

## Install into a separate environment

Do not overwrite the JAX environment or other services. From the existing repo:

```bash
git pull --ff-only
python3 -m venv ../torch170-env
source ../torch170-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-torch170.txt
```

The CUDA12.8 wheel is listed in the
[official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/).
The observed driver595 supports this runtime. Physical GPU validation is still
required. Optional fused candidate, only after installing torch:

```bash
python -m pip install -r requirements-torch170-fused.txt --no-build-isolation
```

The base runner works without mamba-ssm, TileLang, JAX or Flax. If mamba-ssm is
absent, it probes eager and compiled PyTorch SSD candidates only. Do not install an
unrelated third-party Qwen modeling file or run `trust_remote_code=True`.

## Launch

Run from the repo directory containing engine_torch170.py. Reuse the existing
corpus in place; `--model-cache` is one shared original-model cache. It may
download a PyTorch safetensors source if the old JAX cache has another layout.
The teacher is NEVER copied into experiment checkpoints.

```bash
source ../torch170-env/bin/activate
set -a
source ../engine-core.env
set +a
unset LD_LIBRARY_PATH

RUN_DIR="$PWD/../torch170-run"
CORPUS="$PWD/../engine-core/data-pack-edu-600m"
MODEL_CACHE="$PWD/../torch-model-cache"
HOURS=14
mkdir -p "$RUN_DIR"

nohup python -u -m engine_torch170 \
  --work-dir "$RUN_DIR" --corpus "$CORPUS" --model-cache "$MODEL_CACHE" \
  --max-wall-hours "$HOURS" --memory-gib 85 \
  --save-reserve-minutes 18 --profile --clear-stop \
  >> "$RUN_DIR/runtime.log" 2>&1 &
tail -n 40 -f "$RUN_DIR/runtime.log"
```

Change HOURS and memory-gib to the actual allocation. Memory-gib caps this
process's CUDA allocator, not total occupancy including other GPU users. Fixed
100M target can require multiple sessions; there is no promised tok/s or ETA.
Add `--preflight-only` for planner/gates only, zero accepted scientific updates.
Repeated normal launch restores its own CURRENT and the exact optimizer/cursor.

## Stops, storage, Telegram

```bash
# Normal: finish current effective update, save, optional final eval, HF sync.
touch "$RUN_DIR/STOP"
# Faster pause: finish update, retain one local CURRENT, skip eval/HF.
touch "$RUN_DIR/FAST_STOP"
```

Wait for the worker to exit; fast-pause-ready.json is written only afterwards.
Writing one ~9GiB mixer+Adam state still takes time. Interruption within the16-
sequence effective update is not falsely acknowledged as an immediate GPU release.
Resume with --clear-stop; no data repetitions among accepted updates. Hard kill
can lose progress since the last hourly local CURRENT.

One atomic CURRENT.pt is overwritten hourly/finally. Temporary replacement can
need two generations on disk. No teacher weights, per-step checkpoints, or layer
activation caches in that state. With HF enabled, normal finish commits CURRENT,
checks remote LFS SHA256 and descriptor readback, then removes only that local
CURRENT. Any failure keeps it. Fast pause always keeps it. HF history can still
grow by one changed ~9GiB state per completed session; it is not pruned remotely.
Remote prefix is experiments/exp170-torch-clean-align-v1; no old HF state deleted.

HF_TOKEN and existing Telegram environment are used. Existing read-only /check
listener auto-registers results/runtime-torch170.json without a duplicate bot.
Only start/end/failure messages. Telemetry is polled locally, not sent repeatedly.
Use --no-hf or --no-telegram explicitly for local engineering checks if desired.

## Validation limits

44 selected checks passed on local CPU PyTorch2.11.0+cpu / transformers4.57.1,
including14 new numerical/integration/fast-stop/status checks,
legacy initializer/bridge regressions and isolated planner/status tests. CPU
AOT capture/backward is tested; native CUDA Inductor codegen remains a preflight
requirement on the actual GPU.

CPU tests cover copied heads, changed-donor detection, recurrent/JAX parity,
nonzero dynamic gradients, cross-chunk output/all-parameter gradient parity,
activation recomputation, uneven microbatch equivalence, an actual tiny Qwen
alignment update, frozen teacher, Adam/cursor resume, contract mismatch, remote
SHA cleanup guard, isolated-worker crashes/timeouts and status integration.
No server GPU kernel compatibility, BF16 convergence, throughput, or100M result
has been observed by the assistant. Inspect preflight outputs before interpreting
this as an efficient or successful conversion.

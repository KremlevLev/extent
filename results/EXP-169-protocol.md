# EXP-169: TPU diagnosis and matched loss pilots

Prepared 1 November 2026. Worker, CPU supervisor and launch notebook implemented.
Local CPU and virtual-device checks are described below. No physical TPU run has
been performed here;6400 input tokens/s is a target, never a measured promise.

## Immutable source

EXP168 FULL-KD CURRENT260386, Adam105534, Edu window838601, seed123. Full decoder
coordinates; embeddings/final norm/tied head frozen. Actual Mamba MIMO rank4 and
adaptation rank256 unchanged. Validate original source code SHA, manifest/chunks,
runtime and Adam/data clocks. No replacement by BEST or older checkpoint.
A-E per-window source replay tolerance .03 must pass before accepted updates.
A5.7670753 B5.8043190 C5.3834536 D5.0731175 E4.8446160.
Pinned FineWeb-Edu600M preserves old100M prefix and validation; no cycling.
385,318,144 source-fresh input tokens remain. Checkpoint/corpus identity is enforced.

## Resource policy

User runtime:500GB host RAM,20GB SSD,8x16GB TPU HBM. Aggregate128GB HBM is not a
single pool. Mesh data1/fsdp4/tensor2; batch13 is valid because data axis size1.
Checkpoint chunks, source/model/HF caches, datasets and compilation caches use
verified tmpfs only. Require128GiB free tmpfs and256GiB effective available RAM,
including cgroup restriction, before downloading. No SSD fallback. If /dev/shm is
small, stop with an actionable small report instead of filling the20GB disk.
Only small reports and bounded probe log tails are copied to Kaggle Output.

## Shared9-hour budget

Notebook starts its timer before clone/dependency installation. Supervisor receives
only the remaining time; final18min reserved for final panels/checkpoint sync.
Reuse verified published corpus packs via extent.tpu_data_169 before queueing;
otherwise CPU preparation has75min cap. Planning has50min total cap, at most25min per engine;
individual disposable probes have10min cap including initialization/compilation.
These are ceilings, not waits. A successful early boundary immediately ends that
engine sweep. Preparation/planning can be shorter; slow source downloads can
prevent a useful training run and are recorded explicitly.

Measure each integer batch1,2,3,... up to first failure or explicit time/trial cap
(max128). Each probe is a separate process owning the whole TPU slice. Warm2 and
time3 synchronized proposals. Restore identical q/Adam/cursor; accept zero updates.
Select fastest finite successful batch, retaining last fit and failed-worker logs.
Compare resident compiled whole-model step versus layer-streamed exact chain rule.
No padding, repeated samples, powers-of-two jumps or assumed TFLOPS multiplier.
Before training, reinitialize fresh worker and audit selected actual-batch gradient
against streamed reference: cosine>=.98,relativeL2<=.15,CE error<=.03. No tolerance
loosening. Teacher remains sharded in the reference too, never moved to one chip.
Record per-chip parameter/Adam/base/teacher resident bytes, device memory stats and
compiled memory analysis where supported. Missing stats mean unknown, not zero.

## Actual diagnostic pairs

* CE-only versus KL-only unaccepted proposals on one identical fresh training
  window: objective, gradient norms/cosines by layer and mixer/MLP/norm groups;
  actual update/weight ratios and unchanged BF16 coordinate-cast fractions.
* Teacher-prefix versus hybrid-prefix block discrepancy, accumulated discrepancy,
  input/output/residual RMS and max values, on one fresh train and one fixed Wiki
  validation window. These observations do not establish a causal plateau culprit.
* Matched training pilots: mixed .5CE+.5KL control versus CE-only, each from the
  same source q/Adam, same physical batch and identical first approximately1M
  fresh Edu tokens (ceil to a complete batch), with same A-E and EN/RU128 panels.
  LR remains1e-5 mixers/MLPs,1e-6 norms,stableclip1; carried Adam, no warmup reset.
  Pilot clocks/tokens/endpoints saved separately. Compare only equal completed
  tails; partial pilot endpoints are not an equal-budget result. If fewer than3h
  remain before a pilot, omit it explicitly. CE-only saves metrics, discards its
  state and is never donated to control.

Then spend the remaining budget continuing mixed control, restoring its own pilot
CURRENT rather than restarting from the source. Billion-update ceiling cannot
prematurely end9h; actual remaining fresh data/time stops it. No forced LR increase:
EXP167 already found1x better than3x/10x. Longer alternative-loss benefit, causal
activation clipping, FP32 trajectory and architectural capacity remain unmeasured.

## Fault handling and durability

CPU-only parent does not initialize JAX/TPU. It captures worker Python errors,
fatal exits/signals, missing/corrupt JSON and bounded timeouts; terminates only
its own child process group. First failed probe is a boundary of unknown cause,
not automatically an OOM. Failure retains previous valid fits; selected batch is
re-audited in a fresh worker. At most two continuation attempts, same contract,
restore own CURRENT/Adam/data. Unsupported memory introspection is reported.

Control keeps one pruned local atomic CURRENT in tmpfs every30min and on exit;
HF sync after control pilot and at finalization. No22GB write for zero-update
preflight; no checkpoints for CE-only pilot. Full sync is deliberately not performed
every few minutes. Host/session loss can lose locally saved work since last HF
sync; a recoverable worker exit can use local CURRENT. No claim that host death,
quota revocation or unrecoverable libtpu faults can be caught100percent.
Quiet Telegram start/end only; no new polling listener competing with the existing
server bot. SIGINT/SIGTERM requests STOP; finalization remains bounded. All failures
produce aggregate CPU report when the parent/host is alive.

## Launch and outputs

Use notebooks/extent-tpu-169.ipynb in a fresh TPU v5e8 session, Internet enabled.
Enable EXTENT_DEPLOY_KEY_B64,HF_TOKEN,TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID secrets.
Do not run the old notebook JAX/device preflight before the CPU supervisor.
9h includes setup. Last cell starts engine_tpu169 with the remaining total budget.

Download extent-results/extent-m3q-tpu169.json (aggregate and matched pilot
endpoints), runtime-tpu169-control.json (full trajectory/diagnostics), and
runtime-tpu169-ce-only.json if that pilot ran. These reports suffice initially;
no checkpoint downloads needed. Probe log tails are included on failure.

## Local evidence

34 local tests passed (169 diagnostics, isolated planner,168 lifecycle and167 regression).
Tests cover subprocess exception/timeout/malformed output containment, integer
batch boundaries and fastest-fit selection; immutable source fork with carried
Adam/data and own-state priority; real tiny-model compiled/reference gradients,
two optimizer updates, CE/KL diagnostics without source mutation; eight virtual
CPU devices with odd physical batch3 and sharded teacher/head evaluation. Virtual
CPU devices establish plumbing/math checks, not physical TPU throughput or OOM
recovery. The first actual TPU session remains the hardware validation.

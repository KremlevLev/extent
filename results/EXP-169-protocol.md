# EXP169 — TPU execution and recovery diagnostics

Design1November2026. **Not launch-ready:** isolated supervisor implemented and
CPU-tested; actual model worker, TPU engineering preflight and notebook launch
cell remain to be implemented. Do not spend a queued TPU session on this file.

## Source and scope

Freeze EXP168 CURRENT260386, full decoder coordinates, carried Adam105534,
Edu cursor838601, seed123. Frozen embeddings/final norm/head, actual MIMO4 and
adaptation rank256 unchanged. Full source manifest/chunks/runtime checked;
per-window A-E source replay <=.03 before any accepted update. No automatic
reset, loose tolerance or fallback to an old source. Source source-control NLL:
A5.7670753,B5.8043190,C5.3834536,D5.0731175,E4.8446160.
Data remains pinned600M Edu with verified original100M prefix and validation;
next tokens start838601*256. Fresh100M pack alone is too short for this source.

## Why peak TFLOPS is not a throughput prediction

Current create_v5e_mesh makes data1/fsdp4/tensor2 on8devices. Thus batch is not
distributed over8 independent data replicas. Large vocab151936, sequence256,
small microbatches, weight/optimizer sharding, collective traffic, Python tile
dispatch and host-staged activation variants can constrain utilization. These
are implementation facts and candidate explanations, not measured TPU attribution.
Compare resident compiled step against batched streamed reference using actual
same-model/same-batch measurements. No assumed5x multiplier in token planning.

## Fault containment

Notebook parent must not initialize TPU/JAX. Run each speculative batch probe in
a disposable subprocess owning the complete8device slice. Source q/Adam/cursor
immutable; no accepted probe updates. Catch normal exception exit, fatal worker
exit/signals, missing/corrupt JSON and timeout in CPU-only parent. Kill ONLY the
probe's process group on timeout. Always retain log/returncode/timing and last
verified fit. A worker failure does not by itself identify an OOM; distinguish
memory evidence, numerical failure, source mismatch and device failure.

Sweep1,2,4,8,16,... until first failed trial or explicit wall/memory/trial ceiling.
Keep last fitting batch, but select fastest measured finite candidate; biggest
batch can be slower. Optional bounded refinement between fitting/failing sizes
only if planning budget remains. Warm2steps, time at least3 synchronized
proposals, report input and next-token-target throughput separately. Probe
duration must include compilation separately; no q/Adam/cursor mutation.

If libtpu becomes unusable after worker failure, validate recovery in a fresh
worker with the known fit before training. If that cannot initialize, persist
CPU report and notify; do not falsely mark training successful. Host/notebook
termination, unrecoverable accelerator faults and quota revocation cannot be
guaranteed catchable. All successful later training must be resumable from a
durable own checkpoint, not restarted from source after accepted updates.

## Numerical and performance preflight

Verify8local TPU devices/process_count1, BF16 compute/FP32 losses and coordinates,
sharding of q and Adam mu/nu, replicated bytes per chip, actual batch layouts.
Require physical dimensions divisible by partition counts. Record compile
memory analysis and device memory stats where available; null is not zero.
Audit selected resident whole-batch gradients against streamed whole-batch
reference at same q/Adam/input. Existing thresholds cosine>=.98, relativeL2<=.15,
CE error<=.03; all gradients/updates/parameters finite. Test at least two accepted
engineering updates in a disposable state, including CURRENT/BEST divergence,
then discard them before the scientific run. Peak speed is not evidence of
correct loss or state recovery. Save/restore own state in a preflight roundtrip.

## Patient diagnostics and hypothesis checks

Measure fixed small train/validation losses at source separately: do not compare
a random training batch with a differently distributed validation average.
Decompose CE/KL gradient norms and cosines globally and per layer; compare mixer,
MLP and norm groups, clipping factors, actual update/weight ratios, BF16 effective
forward changes and nonfinite counts. Compute teacher/student layer output
discrepancy and residual/update RMS on identical input and on hybrid-prefix input.
Large activations alone are not an attribution of the plateau.

Hypotheses: CE/KL gradient conflict; weak effective updates; localized layer
error; train-vs-validation mismatch/domain specialization. Frozen-source
candidate proposals CE-only, mixed CE/KL and KL-only can reveal local direction
and cost, but do not establish long-run superiority. If token training probes
are added, fork same source/Adam and use the SAME fresh token prefix and
effective batch; isolate their costs from the selected continuation. No winner
from unequal tails. No automatic high LR: EXP1671e-5 beat3e-5 and1e-4.

## Nine-hour budget and training

Total9h including setup, diagnostics, compilations and saves. Reserve18min at end.
Target setup+planner+diagnostics<=90min, with individual compile/probe deadlines;
report partial diagnostics rather than consuming the whole session silently.
Remaining work budget uses observed sustained accepted-input throughput,
including periodic evaluation/checkpoint overhead; no TFLOPS-based token promise.
Default main training continues unchanged CE/KL control from source, equal-batch
probes counted separately. Token/data end stops without cycling. Limited-memory
fallback must be an explicit execution choice and re-audited, not hidden.

Measure exclusive wall phases, actual updates and whole-session throughput.
Keep one local atomic CURRENT generation with q/Adam/data; HF periodically and
exit, metrics separately. Any training worker death makes parent restore own
last accepted durable state before retrying; lost unsaved work counted as cost.
Bound retries and disk/HF queues. One existing read-only Telegram listener,
start/completion/actionable error notices only. Parent saves aggregate report
even when no candidate fits; failed hardware preflight must not launch training.

## Implemented evidence

extent/isolated_tpu_planner.py: CPU-only supervisor with process-group timeout,
result validation, monotone batch growth and fastest successful-fit selection.
Four local tests pass, including real failing/timeout subprocesses. No physical
TPU throughput, recovery-from-libtpu-abort or model-worker result measured yet.

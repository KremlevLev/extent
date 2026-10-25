# EXP161 — batch6 continuation of the saved Blackwell EXP160

Prepared25October2026. User measured batch6 throughput1675input tokens/s on
RTX PRO6000Blackwell96GB (three timed proposals,~.917s/update), peak JAX live
allocation65.13GB. Those are capacity-probe results, not this training outcome.
User explicitly selects batch6. Physical training with this runner is unverified.

## Source and optimization

Each branch bootstraps its OWN server160 local CURRENT checkpoint, including
parameters, complete Adam moments/count/schedule, saved BEST and best_step.
CURRENT is not replaced by BEST. No rank growth, fold, dynamics reset, precision
change or LR scaling. Loss remains .5CE+.5forwardKL(Qwen||student), T1, protected
rank256/pregate. Both Wiki control and C4 branch use batch6/context256.

The immutable155 template only reconstructs the validated frozen backbone and
array layouts. It is never an acceptable fallback for missing parent states.
Parent contract/code hashes, report/current manifest agreement, chunk hashes,
tree/dtypes and Adam counts are checked. A missing/changed parent bootstrap
rejects startup; existing own161 checkpoints take priority on resume.

Source parent plan WORK/batch6/source.json is frozen once, records parent manifest
generations/SHAs, own source/Adam/data origins, current NLL references when
available, corpus identity and runtime. Published under
experiments/exp161-batch6-v1/source.json; transient429/network failures retry with
bounded Retry-After-aware policy. Re-launch republishes the same plan if a prior
upload failed, never silently selects a newer parent. Own checkpoints can resume
without re-reading changed/pruned parent chunks once bootstrapped.

Original parent source replay must have passed. Before the first batch6 update,
measure own CURRENT A/B/C/D; compare recorded parent CURRENT per-window references
when available with original .03 bound. If final parent NLL references are absent,
report available=false/passed=null (not a fabricated match), retain verified
source/chunk/clock evidence and record new baseline before learning. A measured
mismatch rejects updates. Single-window evaluation math remains unchanged.

## Separate optimizer and data clocks

For each arm record parent global G0 and consumed input windows W0=G0-118304.
After n accepted batch6 updates:

- Global=G0+n; Adam=parentAdam+n, preserving all counters/moments.
- Consumed windows=W0+6*n; additional input positions=1536*n.
- Next C4 batch uses windows W0+6*n ... W0+6*n+5, no modulo.
- Wiki uses the original GLOBAL window permutation, starting at
  118304+W0+6*n ... +5. Its cyclic reuse is deliberate control behavior.
- Each update has1530 next-token targets, recorded separately from1536 inputs.

No overlapped/skipped windows on own checkpoint resume. An insufficient final
C4 tail of1..5 windows is not repeated/padded or used in a secretly smaller batch;
stop with data_exhausted, leaving that tail unused. Both branches are scheduled
by new161 update counts; their parent prefixes may differ and are recorded.
Registered evaluations use new batch6 update counts; compare token presentations
as well as optimizer steps. Batch1-vs6 quality is a changed optimization recipe,
not an execution-only speed comparison.

## GPU execution and first real update gate

Use the measured whole-compiled no-remat objective with teacher, both branch
weights/optimizers and intermediates on GPU. Allocate batch6 compiled execution
FIRST, avoiding the smaller-shape allocator history that falsely rejected batch4
in the first capacity sweep. Reuse WORK/compile-cache.

Before acceptance, compare actual compiled batch6 gradient to the MEAN of six
single-example gradients at SAME current q/Adam. The six reference proposals are
discarded; apply Adam ONCE to their mean, not six times or to averaged weights.
Use existing compiled/streamed BF16 gate (relativeL2<=.15,cos>=.98,CE<=.03 and
finite proposal stages), not a bitwise/long-trajectory identity claim. If compiled
fails parity or memory, fallback remains batch6 via accumulated gradients,
explicitly reported; it does not silently become batch1 training. All reference
probes are unaccepted and their extra teacher/student inputs are accounted.

## Checkpoints, storage, notifications

Separate files WORK/results/runtime-b6.json and WORK/states/runtime-b6/;
HF prefix experiments/exp161-batch6-v1. Original160 report/states untouched.
Initial fork copies ONLY small trainable state/Adam/best (~.9GB/branch), not full
frozen models or corpus. Cache, C4 data-pack and compilation cache are shared.
Atomic local generations,15min/branch plus exit; HF4h plus exit;32GB invocation
upload cap,80GB owned-work cap and20GB free reserve. Existing source states are
retained so old batch1 remains independently resumable.

Same shared server.lock and WORK/STOP prevent concurrent workers/unsafe reads.
--clear-stop only clears that owned stop marker. SIGINT/SIGTERM requests bounded
final save; SIGKILL/power loss cannot. Chosen wall hours include source planning,
downloads, compile/audit, evaluations and18min save reserve. No short step ceiling;
C4 corpus exhaustion may end a run before wall time. Only start/finish Telegram.

No existing TPU job or unrelated GPU service is stopped by these files.

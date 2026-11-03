# EXP171 — clean Q/K/V/O alignment and throughput on TPU

3 November2026. PREPARED plus CPU engineering checks; no physical TPU result.
This replaces the intended next run, not the mathematics/history of EXP169.

## Source and scientific contract

Original Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1 only.
No Patient reconstruction, EXP069/072 bank, EXP168 parameters, old LoRA or old
Adam. Loader reads the pinned original safetensors and checks source shapes.
Q/K/V/O, Q/K norm and own-head rank selection are inherited exactly; runtime
assertions check all copied projections. Z is explicitly random. Dynamics rows
zero, bias1, simplex MIMO input factors, one-hot own-query readout, full pre-gate
RMSNorm/tanh. H16/G8/P128/N128/inner2048/MIMO4, same geometry as clean170.

Train ALL27 Mamba mixers,798,316,128 coordinates, on original teacher-prefix
block-output targets, with frozen original residual/norm/MLP. One teacher pass
per microbatch; lax.scan contains local value_and_grad for each layer, rather
than backpropagating across the teacher or storing an offline activation bank.
Per-sequence relative MSE averaged over27 layers. Adam(.9,.95),eps1e-8,global
clip1,46-step warmup,peak1e-3,cosine to.1 peak. Moment clock advances once per
effective update. Uneven last microbatch is weighted by count/effective batch.

**Explicit difference from GPU170:** context1024, effective batch16,16,384
inputs/update,6104 updates /100,007,936 input presentations. GPU170 uses context
4096 and65,536 inputs/update; their quality/rates are not directly comparable.
Use a separate no-HF smoke contract for other shapes; canonical HF namespace is
protected against context/batch/horizon overrides. Teacher input100M is shared
by27 layers: this means2.7B layer-input presentations, not2.7B unique text tokens.

The existing verified FineWeb-Edu600M pack is reused in place. Original file SHA,
manifest and corpus identity are checked against the published inventory.
Sequential training cursor, no epochs/repeats among accepted updates. Validation
is the separate validation.bin. The pure-Edu corpus is a declared difference
from the peer's mixed corpus. Head-preserving recipe attribution and unresolved
Z choice: [EXP170 audit](EXP-170-clean-start-audit.md).

## TPU execution

JAX0.11/Flax0.12.8/Optax0.2.8; no Torch, Torch/XLA, Transformers or tokenizer
needed by this runner. USE_TORCH/USE_TF/USE_FLAX=0 prevent optional discovery
from pulling competing runtimes in third-party code. Eight local TPU devices,
one model-sharding axis. Stacked teacher/bank tensors partition an eligible
feature axis, never the layer axis. Host source weights are never re-uploaded
as experiment state. Full blockwise MIMO SSD uses chunk64 with recomputation,
FP32 state/rotation/decay algebra and BF16 dense projections/MLP.

Each disposable worker logs import, original host initialization, TPU discovery,
eight-chip collective smoke, device placement, own-state restore, explicit
gradient compilation, first execution, independent recurrent gradient gate,
Adam compilation and full effective-update timing. phase.json records PID and
timestamp. faulthandler dumps all Python threads every180s; before timeout the
parent requests another stack. This localizes stalls; it does not certify that
all native TPU hangs can recover or symbolize native code.

No single600-second catch-all timeout. Planner has60min shared maximum,20min per
candidate including its source/setup/compile/gates/timing. Integer physical
batches1,2,...,16 (bounded by effective batch), stopping at first failure/time
cap. Three discarded effective updates per successful probe; first warms Adam
allocation, two timed measurements. Report speed includes teacher+27 student
backwards+one global Adam. Probes accept zero scientific updates and write no
scientific checkpoint. Select fastest verified fit; preserve earlier fits after
a later OOM/fatal process exit/timeout. No valid fit means stop with phase/stacks,
not silently train a numerically unverified fallback.

Before accepting training: real-donor64-token blockwise-versus-recurrent output
relative L2<=.03, gradient relative L2<=.15/cos>=.98, finite gradients. Different
BF16 rounding is bounded, not called bitwise equivalence. Compilation/warmup and
engineering token presentations are separate from training. Per-chip allocator
stats are recorded, not multiplied into a fake single128GB device.

## Learning measurements and time/storage

At source and every128 accepted updates when time allows: teacher plus complete
gqa4/gqa7 hybrid NLL on SAME four held-out Edu1024 windows. Original final layer27
is retained. Log loss, per-layer relative MSE, LR, accepted input presentations,
complete-update input tok/s and training-loop throughput including its evaluations
and local saves. This is a small diagnostic panel, not benchmark recovery and
not historical A/B/C/D/E or170's4k panel. A lower local loss need not improve
composed NLL. No quality outcome or6400tok/s is promised.

9h maximum TOTAL from notebook setup. Parent passes one absolute wall deadline;
source/probes/compilation reduce remaining training time.18min save reserve.
100M horizon is explicit: completion can precede the wall limit. All source,
corpus, compilation cache and large state stay on verified tmpfs with128GiB free
and256GiB effective RAM gates.20GB notebook SSD receives logs/reports only.

One own chunked ALIGN CURRENT (mixers+Adam+cursor), overwritten with safe prior
generation retained until manifest commit. Local save every15min/end; obsolete
known local generations are pruned. No teacher copies/BEST states. Fresh cloud
invocation restores its own remote CURRENT, checks exact contract and data clock;
it never quietly starts from an old recovered hybrid. Final HF sync verifies
manifest/sample readback; pending state is explicit. Local RAM-only checkpoint
without completed HF sync will not survive session destruction. HF history is
not pruned by this runner; each final state is roughly9GiB.

STOP finishes the current effective update and saves/syncs. FAST_STOP saves one
local CURRENT and skips HF; a TPU cloud session then has to remain alive to resume
locally. No instantaneous stop guarantee. Start/end/failure Telegram only; no new
getUpdates listener competes with the user's existing bot.

## Validation boundary

CPU checks cover blockwise versus independent recurrence outputs/all parameter
gradients with nonzero angles across chunk boundaries; teacher attention versus
the established Qwen formula and causality; uneven microbatch weighting; stacked
teacher/layer gradient scan, frozen teacher, one Adam clock; phase-timeout capture;
own checkpoint restore and changed-source rejection. Eight virtual CPU devices
also execute real sharded clean gradients and Adam update. These are not physical
TPU compilation/HBM/speed or100M convergence measurements.

# EXP-142: does unlocking the decoder help after the EXP130 reset?

Status: PREPARED, single-seed123 screening, TPUv5e8, one9h session.
Launch: [EXP-142-launch.md](EXP-142-launch.md).

## Hypothesis and comparison

EXP139 compares R32 and all-Mamba dense recovery from folded EXP130 current3434.
EXP142 starts from the identical function but unlocks every decoder parameter.
Primary comparison:142 versus139 RESET130-DENSE at matched *additional* training
updates/tokens. Both reset Adam and train/data cursor to0. Neither inherits the
EXP130 Adam; EXP135 remains the exact-Adam continuation. This is NOT EXP112 resume
or a random-start efficiency control. EXP112's unfinished trajectory stays separate.

Use original source pin EXP-135-source-pin.json: dataset levkremlev888/extent,
revision b85a8fbfb2832ad7cc4f7442defaafb454417e6a, EXP130 worker1 CURRENT3434,
all source q/Adam/best chunks verified. Best3072 is not the start. Existing source
loader folds LoRA and the completed reset into base Mamba; copies learned pregate
weights; removes reset auxiliary fields. Pregate normalization and tanh angles
remain active at strength1 from the first update; no repeated overwrite of dynamics.
Each A/B/C source-window NLL must match the pinned source within0.03 before training.

## Registered training recipe

One arm RESET130-FULL-DECODER. Trainable FP32 masters: all24 Mamba, all28 MLP,
remaining4 GQA, input/post-attention RMSNorm, QK norms, final RMSNorm, added pregate
weights and one TIED input embedding/readout. Both uses of the tied matrix share
one parameter/Adam state and accumulate gradients. No LoRA restriction.
Every original parameter must appear in the trainable tree; unknown base groups fail.

Shared with139: CE, context256, batch1,255 next-token targets, same pinned4,194,304
input-token FIT pool/permutation, seed123, stable global clip1, Adam beta.9/.999,
eps1e-8,256-update warmup, flat thereafter, all-Mamba LR3e-4 including internal
Mamba norm scales, pregate LR1e-4. Newly unlocked MLP/GQA/embedding kernels LR3e-5;
original decoder/GQA/final norm scales LR1e-5. New groups have no learning-rate
counterpart in139; this is a registered two-rate decoder recipe, not an LR sweep.
Global clipping includes the new gradients, so shared Mamba effective updates can
change even when their nominal LR is identical. No claim of an isolated rank effect.
No weight decay, teacher KL or label smoothing. No adaptive rescue/LR pilot mid-run.

Single in-kernel JAX process over all8 TPU devices, model/optimizer sharded.
Tiled full-vocabulary loss, full decoder rematerialization, BF16 compute/storage
views with FP32 trainable masters and Adam. No data-parallel batch increase.
Source reconstruction base stays explicit in compiled function arguments.

## Endpoints and diagnostics

First goal32768 extra updates (8,388,608 input positions), ceiling131072, work until
wall deadline. Same registered evaluation grid as139; source, A64-selected best,
last A/B/C. B/C do not select. Source training/preparation cost is inherited and
must not be advertised as free or counted as only142 input positions.
Compare both matched tokens and measured training/wall time, plus total accepted
and attempted updates. Stop by time is not convergence. Do not extrapolate curves.

Source/4096/8192/16384/32768/65536/98304/131072 and final/best layer profiles:
residual/mixer/MLP scales, absorption and existing Mamba diagnostics. Diagnostic
versus ordinary layer output must agree within2%; reject unsupported mechanism
claims if not. Actual NLL always uses the ordinary full learned decoder.
Final contexts128/256/512/1024 use identical last127 targets on8 windows.

Every proposal reports finite objective/current params/Adam/gradients/updates/new
params/Adam. First real health is persisted before accepting an update; nonfinite
proposal does NOT advance weights, Adam or data cursor. Six group summaries report
parameter counts, gradient max, changed FP32 master entries and changed BF16 views:
Mamba, MLP, GQA, original decoder norms, pregate norms, tied embedding. Small updates
can leave individual BF16 entries unchanged. These counts are descriptive, not proof
that quantization causes or cures a plateau. Throughput/ETA after8 accepted steps;
ETA excludes evaluation and saving and is not a completion promise.

## Time, storage, resumption

Runner8.5h INCLUDES60min reserve; outer30min for notebook startup. Local snapshots
512 updates. Large full-decoder Adam+best checkpoint uploads every4h and exit,
80GB NEW upload cap/invocation, no frozen-base copy and no HF cleanup. Report
actual payload_bytes and parameter group counts. Interrupted VM before remote
sync loses local tail. Local disk and physical HBM are checked by existing
checkpoint/compilation guards, not assumed from small CPU tests.
Outputs experiments/exp142-reset, extent-m3q-reset-142.json. Exact resume contract
includes source pins, runtime, horizon and code hashes. Old135-141 files/contracts
are untouched. Physical1.7B TPU memory, speed and scientific outcome remain pending.

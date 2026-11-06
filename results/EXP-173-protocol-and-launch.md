# EXP173 — composed gqa7 CE + output distillation on TPU

PREPARED6Nov2026, no physical TPU173 result yet. Use notebooks/extent-tpu-173.ipynb
in a fresh TPUv5e8 session,9h including installation/prepare/planner/save. Same
four Kaggle secrets as171. Dedicated requirements-tpu171.txt, no PyTorch/XLA.

## Source and contract

Only gqa7 mask3/7/11/15/19/23/27,21 Mamba replacements. Never train/evaluate gqa4
in this run. Frozen clean171 CURRENT5120 at HF dataset revision
d341d2b8f87592f617c44c92c5d7532669c75755, exact source chunk SHA. Stage1 bank
contains27 mixers;7 attention configuration uses21. No random069 lineage,
no rank adapters, no stage1 Adam. Original Qwen source pinned exactly as171.

Composed full-network backprop through all28 blocks. Update all21 Mamba mixers
(620,912,544 coordinates); original attention/MLP/norms/embeddings/head frozen.
Unused six bank slices remain immutable with zero gradient; current storage
retains27 stacked slices/Adam arrays for layout compatibility, not21-only storage.
This is mixer-only optimization with end-to-end gradients, NOT full-decoder FT.

Loss .5CE + .5 forwardKL(teacher||student), temperature1, exact full151936-vocab
probabilities; teacher sees same ground-truth prefix, not generated rollouts.
No hidden-target term. FP32 loss/state, BF16 dense projections. Teacher forward
stop_gradient; student receives its own hybrid prefix. Original tied head frozen.

Fresh original-Edu600M stream starts at input offset83,886,080 after171 training,
never cycles; target12208 updates x16x1024 =200,015,872 new input presentations.
No training on validation/benchmark questions. Each update16,384 inputs and
16,368 next-token targets. Repeated engineering probes reported separately.
Adam(.9,.95),eps1e-8,stableglobalclip1; peakLR1e-4/warm128/cosine to.1peak.
New optimizer for new objective. Values are hypotheses, not optimality claims.

## Planner and performance

Eight-chip model sharding same as measured171, stacked layer-axis never sharded.
Rematerialized layer scan differentiates composed network. Vocabulary logits
calculated32 token positions/tile with rematerialization: avoids B*T*V materialized
full-batch teacher/student logits; exact KL, not top-k approximation.
No explicit data-parallel or alternative SSD-kernel search in this first run.

Separate process per physical batch1,2,...16 at fixed effective16, weighted uneven
tail;3 complete discarded updates (first warm,2 timed), including teacher,
student, backprop, accumulation, clip/Adam and readiness sync. Compile/setup
excluded from measured samples, included in session budget. Both fresh fits
and failures persist; first failed probe/time cap stops search; fastest earlier
fit selected even if largest fit is slower. Failure not automatically labelledOOM.
20min per process/60min planner cap, preserve work time; not a promise to reach
physical memory boundary. Timeouts/signals handled in CPU supervisor, no TPU
context retained in parent.5 tests of reused isolated planner include real
child MemoryError, missing result and timeout. Native TPU service can still fail;
no100%crash-immunity or predicted6400tok/s claim.

171's~2973tok/s was local block-MSE training.173 includes full-vocabulary teacher
and student output distributions and a composed backward: workloads differ.
TFLOPS ratio alone does not establish tokens/s or MFU. Print physical/effective
batch, input tokens/update, measured full-update speed, accepted tokens, CE/KL/LR;
report whole training loop throughput separately.

## Durability and diagnosis

Large weights/data/own CURRENT/compilation cache only on verified tmpfs; require
128GiB tmpfs free/256GiB available host RAM, as171. NotebookSSD logs/results only.
Phase/PID/time,180s Python stack dumps, SIGUSR1 stack before supervisor timeout.
First-gradient finite check; independent recurrent output/gradient gate on learned
source; initial gqa7 four-window Edu replay within.08 of2.583224. No threshold
relaxation after failure. Evaluation teacher+gqa7 only, same4 monitored1024 windows,
not fresh broad benchmark or proof of improved MMLU.

Own prefix experiments/exp173-gqa7-end-to-end-v1/ALIGN, never overwrite171.
Before training: local CURRENT save+full SHA-checked restore roundtrip and cursor
check. Every15min/final local CURRENT; previous generation retained until atomic
manifest, then prior known local generation removed. Final HF verified manifest
and sampled chunk as existing RecoveryCheckpointStore. Remote history not pruned.
18min save reserve; storage_partial explicit if HF not synchronized. RAM state
alone does NOT survive cloud destruction; residual risk of quota/network outage.
Quiet Telegram start/end only. STOP finishes current update, checkpoint/sync;
FAST_STOP saves local state/no HF, unsuitable before destroying Kaggle session.

## Launch

Import updated173 notebook; fresh TPU session + Internet + same4 secrets.
Last cell TOTAL_HOURS=9; default runner target200M new inputs, train until deadline
or target/exhaustion. No step cap at the old5120/6104 clock. Do NOT use old171
last-cell import. Return extent-results/extent-m3q-tpu173.json; on failure
include matching prepare173/probe173-bN/train173.log. No broad quality claim
until frozen follow-up benchmark against EXP172 baseline.

## Local checks

New tests verify exact tiledCE/KL vs dense values/gradients and detached teacher;
composed27-layer gradient is nonzero on21 Mamba layers and zero on6 retained-bank
slices; changing early Mamba affects final composed hidden states. Includes a real
8-virtual-CPU-device sharded full-composed gradient. Reuses171 source/clock/
checkpoint/layout and isolated supervisor tests. These are not physical TPU
memory/codegen/throughput/convergence measurements.

Validation completed:17 selected CPU tests passed in71.46s (4 new173,
8 existing171,5 isolated planner). Notebook cell syntax verified.

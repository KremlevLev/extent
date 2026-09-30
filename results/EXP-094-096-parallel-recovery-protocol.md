# EXP-094 / 095 / 096: parallel whole-model correction campaigns

Prepared 2026-10-01. Three independent Kaggle TPU v5e-8 accounts may run these
concurrently. Each uses its own HF prefix, RAM state directory and output names.
No cross-account dependency on freshly produced checkpoints.

## Shared immutable protocol

- Pinned Qwen3-1.7B-Base, completed EXP-072-v2 ONPOLICY sources, seeds123/456.
  Production 28-layer hybrid: 24 Mamba-3 MIMO and 4 retained GQA layers.
  This isolates recovery; MLA is not introduced in these experiments.
- All original weights frozen BF16; only zero-change FP32 correction coordinates
  and FP32 Adam moments trained. Low-rank `A B` corrections fold into kernels;
  gains fold into out_proj rows. No inference-time new module is required.
- Adam lr3e-4, global clip1.0, batch1, sequence256, no accumulation.
- WikiText103 pinned by the existing loader; same ordered texts across methods.
  Train offset8,388,608, 8,192 windows (2,097,152 unique tokens per trajectory).
  Validation offset49,152 /32 windows; locked test offset32,768 /64 windows.
  Training is not asserted fresh relative to all historical experiments;
  the test range differs from EXP-093. Token SHA256 and source SHA256 are checked.
- Balanced horizons2,048 ->4,096 ->8,192 for every arm/seed before extending.
  Eight trajectories/campaign, 65,536 total student steps /16,777,216 student
  training tokens if complete. Teacher and anchor passes cost additional compute.
- Validation every1,024 steps is diagnostic. Locked test only at final8,192,
  alongside the unchanged starting model. No checkpoint selection or test tuning.
- Primary gate requires final primary NLL <= control NLL-0.1 and below the
  unchanged start in **both seeds**. All arms must complete. Two seeds are screening
  evidence, not a sufficient statistical basis for a conference-level claim.

## EXP-094: correction subspace

| Arm | Changed kernel groups | Rank |
|---|---|---:|
| OUT-LORA | Mamba out_proj | 8 |
| INOUT-LORA | Mamba in_proj + out_proj | 8 |
| MLP-READOUT-LORA | copied MLP down_proj at replaced layers | 8 |
| HEAD-GAIN | individual Mamba head output-row gains | — |

All use true-token CE. Primary INOUT vs OUT tests whether adapting the input
projections adds recoverability beyond readout correction. Other arms exploratory.

## EXP-095: learning objective

All arms use the same rank8 INOUT coordinates and initialization.

| Arm | Objective |
|---|---|
| CE | true-token CE |
| TEACHER-KD | original Qwen KL(T2) +0.1CE |
| SELF-ANCHOR | CE +0.1KL(T2) to frozen starting hybrid |
| DELTA-THEN-KD | steps1..2048 decoder-delta relative MSE +0.1CE; then teacher KD |

Primary staged objective vs TEACHER-KD. Delta matching uses replaced layers>0:
layer zero excluded because embedding states are not exposed. Each model uses
its own upstream stream, so this is assembled-model contribution matching,
not identical-input local layer distillation. This is a recovery-objective bridge,
not a trained linear-attention-to-Mamba architectural bridge.

## EXP-096: capacity and dynamics protection

INOUT CE arms rank8/rank32/rank64 plus rank32-PROTECTED. Protected corrections
cannot modify the in_proj columns producing dt/raw_a/trap/angle; z/x/B/C and
out_proj remain adaptable. Those projection weights stay exactly unchanged,
but their activations can still shift due to changed preceding layers.
Primary protected32 vs ordinary32; rank scaling exploratory.

Rank8 INOUT has1,786,368 allocated coordinates; rank64 has14,290,944.
Replicated rank64 coordinates+two Adam moments occupy~163.55MiB/device,
excluding gradients/model weights/activations/compiler buffers.
Protected32 allocates7,145,472 coordinates,7,077,888 active.
Actual kernel shape/count is checked against the analytical plan at runtime.

## Execution, persistence and scientific boundaries

- Default wall budget8.25h; training stops30min before that to leave saving and
  reporting room. This is a cap, not a measured promise of runtime. Same last cell
  in a new session resumes exact optimizer/cursor/adapter state from HF.
- Every256 steps local atomic binary checkpoint; remote saves bundled across
  pending branches at most once/10min, plus initial/final commits. Three accounts
  should stay far below128 repository commits/hour unless other jobs also upload.
- 429 defers upload for1h without stopping local training; other transient upload
  errors defer120s. Auth403/401 fails explicitly. External VM termination before
  remote sync can lose the unsynced tail; no code can guarantee that callback.
- Nonfinite updates retain the last finite state, fail only that branch, and allow
  remaining arms to run. Structural/compiler failures are reported, not hidden.
- Checkpoints contain coordinates+Adam only. Latest JSONs remain compact. Source
  provenance/contracts and implementation file hashes prevent incompatible resume.
- Telegram start/final/caught-error notifications. A completed rerun allocates no
  TPU model. `--plan-only` works without Hub/TPU; `--sync-only` retries local upload.
- Shared rank8 CE controls across the three jobs are duplicated implementation
  controls, not six independent samples. Equal tokens do not imply equal compute.
- Results cannot be called full recovery, a new initializer win, or an inference
  speed result. Promising corrections require held-out domains and more seeds;
  they may later motivate an initializer/bridge recipe.

## Last notebook cells (clone/dependencies/secrets already installed)

Local verification (2026-10-01, Windows / JAX0.6.2 CPU): zero-change initialization,
finite compiled updates for every arm/objective, protected-column gradients,
binary Adam roundtrip, final-evaluation resume, and mocked cloud completion /
interruption-resume /429 workflow. Separate eight-virtual-CPU tests compile
sharded CE/protected, teacher-KD and staged-delta updates. Real TPU memory,
throughput, Hub transfers and scientific results are not yet measured.

Account1:
```python
from scripts.m3q_subspace_campaign import main as run_exp094
result = run_exp094([])
```
Account2:
```python
from scripts.m3q_subspace_objective_campaign import main as run_exp095
result = run_exp095([])
```
Account3:
```python
from scripts.m3q_subspace_capacity_campaign import main as run_exp096
result = run_exp096([])
```

Download/send each `extent-m3q-*-summary.md` (or latest-summary.md from HF).
No manual transfer of model/checkpoint files is required.

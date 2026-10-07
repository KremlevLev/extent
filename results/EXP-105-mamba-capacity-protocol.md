# EXP-105: full Mamba capacity with the Qwen backbone frozen

Registered7 October2026 before any EXP105 result. Separate Kaggle account,
one8h TPU v5e8 invocation,6.5h loading/compilation/training +1.5h save reserve.
Runner `scripts/m3q_mamba_capacity_recovery_campaign.py`; shared explicit engine
configuration, no global mutation of EXP104 settings. No TPU result yet.

## Question and evidence

Does the protected rank32 update subspace cause the recovery plateau, or does
effective recovery also require adapting copied Qwen MLP/GQA/norms? EXP104 opens
the entire internal decoder. EXP105 opens every parameter inside the24 Mamba
blocks (including full in/out projections, dynamics, MIMO weights and B/C norms),
while ALL other leaves remain frozen: Qwen MLPs, GQA, decoder norms, vocabulary.
This is a scope ablation of the paper-inspired stage2 strategy, NOT an exact
HedgeMamba replication or a claim that Mamba3 is automatically stronger.

EXP102 is now measured, completed/finite, gateFALSE. Smaller LR improves Wiki
relative to3e-4 by0.05110/0.03290, below0.1. Larger LR improves PG19 versus3e-5
by0.21349/0.07983. LR1e-4/3e-5 Wiki ordering reverses between seeds. Combined with
101/103, this argues against another blind LR/KL/rank sweep; it does not prove
that dense Mamba adaptation will work. Full Mamba removes the rank32 restriction
that RELEASED103 retained, and includes additional recurrent/MIMO parameters.

## Fixed comparison and training

Two existing seeds123/456, fixed098 SAFE-PROTECTED-R32 final16384 at the exact
revision/SHAs in `EXP-101-103-source-manifest.json`. No newer/best102 checkpoint.
ADAPTER-CE: protectedrank32 control, frozen072-v2 base. MAMBA-CE: exact BF16
materialization of the SAME recovered student, FP32 masters for ALL Mamba leaves.
Other model leaves remain source-BF16 and never enter optimizer. No zero/random
fallback, new preprocessing bridge, or reset of recurrent dynamics.

Both CE ONLY, fresh AdamW at transition, beta(.9,.95),eps1e-8, stable global clip1;
peak1e-5, warmup64fromzero, cosine2048 to1e-6, decay.1 for non-bias trainable
weights with>=2dims; no vectors/named biases. All adapter A/B matrices decay.
Context256/batch1,2048 additional windows per trajectory. Horizons512/1024/2048,
seed123 control then Mamba;456 reverse. Total2,097,152 student input tokens.
Compatible resume retains new moments/count/data cursor. Save rejected proposals
without adopting nonfinite values; resource/unfinished runs do not falsify method.

Canonical materialization and BF16 evaluation layout shared with fixed104-v2;
bitwise identical initial weights and per-window NLL equality at1e-4 required.
Never relax the threshold merely to pass. Teacher evaluation only for baseline;
teacher arrays released before training. Source caches on host, masters/moments
sharded; checkpoints in RAM filesystem on Kaggle, atomically chunked/uploaded.

CPU abstract production shapes:256,725,120 trainable parameters,
3,080,701,448 raw checkpoint bytes. Conservative7,888,509,184B/device estimate
includes6GiB POLICY workspace allowance. These are not real TPU measurements.
Runtime checks capacity/compiler footprint, host RAM/filesystem space and logs
actual upload timing. Completion of all four trajectories in8h not guaranteed.

## Data, evaluation and registered interpretation

Deliberately reuse the pre-registered104 token slices and hashes:
Wiki103 trainoffset39,845,888/2048 windows; validation114,688/32;
locked Wiki test294,912/24; PG19 test294,912/64. Pins/token hashes in
`EXP-104-data-preflight.json`; test ends301,056 and fits observed capacity301,829.
Only104 STARTS have been seen; no104 final result informed choice of105.
This makes comparisons matched across campaigns, not independent replication or
fresh unseen data. Wiki locked suffix contains only6144tokens; report limitation.
Validation every256 diagnostic only. Qwen baseline, recovered start and final2048
on both tests, save per-window NLL. No best endpoint/seed/checkpoint selection.

Primary gate: all4 finite/complete; MAMBA-CE Wiki>=0.1 better than its control
AND>=0.1 better than own recovered start in BOTH seeds; PG19 own-start regression
<=0.1 in both. Additional transfer gate requires>=0.1 PG19 control advantage
in both. Report Qwen gap and per-window paired deltas. Shared controls versus104
are consistency checks, not new seeds/evidence. No token-efficiency/random/full
teacher-quality/14B claim without appropriate matched total-compute controls.

If104 wins but105 fails, adapting non-Mamba layers may matter (needs further
scope ablations, not a proof). If105 wins, full Mamba capacity is sufficient for
this short recovery gain; validate longer/fresh tests before14B. If neither wins,
these particular short-budget strategies failed, not the whole paper. If OOM/
unfinished, preserve durable state and report feasibility limitation.

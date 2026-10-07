# EXP-104: paper-inspired full-decoder CE recovery

Status: RUNNER PREPARED, CPU checks and real token preflight; no TPU results.
7 October update: observed v1 parity stop before dense training; v2 canonical
evaluation correction with unchanged1e-4 and exact original checkpoint contract.
See `EXP-102-and-104-update.md` and `EXP-104-105-kaggle-launch.md`. No dense outcome.
One experiment, one Kaggle TPU v5e8 session up to8h. Launch and limitations in
`EXP-104-kaggle-launch.md`. EXP102 full results still unavailable.

## Hypothesis and literature basis

Attention to Mamba: A Recipe for Cross-Architecture Distillation,
https://arxiv.org/html/2604.14191v1 , sections3.2/4 and appendixA.2.
The important stage2 recipe is true-token CE, nearly all internal parameters
trainable, input/output embeddings frozen, AdamW beta=(0.9,0.95), weight decay0.1,
clip1, warmup + cosine decay to0.1 peak, LR in1e-5 range. It does NOT mean adding
teacher KL throughout this second stage. Published default uses1B+9B tokens.

Extent adopts this STAGE2 training strategy to an already recovered Mamba3 MIMO
hybrid. No new Hedgehog bridge, normalization denominator, identity dynamics,
or reset of transferred recurrent parameters. No claim to reproduce their
HedgeMamba architecture or their10B-token result. Prior054/055 bridge adaptations
failed and are not silently reopened. Our prepared warm start is different.

Question: can allowing the full decoder to adapt escape the adapter plateau?
101 late KL yields tiny mixed Wiki differences; stronger KL improves PG19 but
worsens Wiki. 103 opening input dynamics within the same rank32 subspace fails.
Neither experiment opens MLPs, GQA blocks, all Mamba kernels and layer norms.

## Fixed source and comparison

Two original source seeds123/456. SAME pinned098 SAFE-PROTECTED-R32 final16384
checkpoints from `EXP-101-103-source-manifest.json`, not best101 or unverified102.
Original frozen072-v2 base layer SHAs must match. Fixed source prevents endpoint
selection on latest tests and provides known completed source states.

Two branches per seed:
- ADAPTER-CE control: existing protected rank32 corrections on frozen base.
- DECODER-CE primary: materialize SAME current student computation; optimize all
  internal Mamba, retained GQA, MLP and normalization parameters. Freeze both
  input embedding and final vocabulary readout (one shared weight if tied).

Both fresh AdamW at transition; same registered LR schedule, betas, clipping,
data order, steps and endpoints. Weight decay applies to non-bias trainable weights
with >=2 dimensions (including MIMO tensors), not biases/norm vectors; adapter A/B
both receive decay. This compares complete
training strategies, not a mathematical single-factor parameter-count intervention.
Original source optimizer is not reused; compatible resume preserves NEW moments.
Teacher forwards are needed only for original-Qwen evaluation, not training loss.

The primary uses FP32 master weights/moments for trainable leaves, BF16 forward
weights with FP32 sensitive calculations per existing model. Do NOT keep only BF16
trainable weights at1e-5: quantization can erase small updates. Frozen embedding/
readout retain source BF16 values bitwise. Shard masters, gradients and Adam state
with model mesh1/4/2; do not replicate full optimizer on all eight devices.

Start materialization must equal the current adapter-applied BF16 student kernels
exactly. Re-evaluate both initial models on validation and BOTH tests; reject
unexplained initial NLL difference >1e-4. No hidden zero-adapter/random fallback.
Dense primary need not preserve protected dynamics during training: all decoder
weights are allowed to adapt, as in the paper's second-stage strategy.

## Budget and optimizer

Four trajectories:2 strategies x2 seeds. Context256, batch1, horizons512/1024/2048.
Target2048 additional steps per trajectory:524,288 input tokens each,2,097,152
across the experiment. This is a small feasibility test, not the paper's9B recovery.
Do not describe repeated source preparation or old research cost as free.
Rotate strategies/seeds across horizons;123 control then primary,456 primary then
control. Never spend whole session on one seed and call it paired evidence.

Peak LR1e-5 for both, warmup64 steps from zero, cosine decay through2048 to1e-6.
AdamW beta1=.9,beta2=.95,eps1e-8; decay.1; stable max-scaled global-norm clip1.
Loss true-token CE ONLY. No KL, hidden matching or additional hyperparameter arms.
Exact mask determines effective trainable count and optimizer footprint, recorded
from actual model; no guessed parameter-count proof. Log actual update magnitudes
and finite health; norm-only events use safe clipping, nonfinite leaf events stop.

Invocation max8h; training deadline6.5h, final1.5h reserved for LARGE binary states
and uploads. Setup notebook time outside runner deducted from wall budget. This
larger reserve replaces small-adapter45min reserve. No full-completion promise.
Compiled full-model step and large HF checkpoint may dominate. If partial, retain
last compatible parameters/moments/cursor and mark gate unavailable; continue same
registered horizon next session without switching seed or LR settings.

## Data and evaluation

Pinned WikiText103/PG19 and Qwen tokenizer revisions unchanged.
New train offset39,845,888 /2048 windows x256; new validation offset114,688 /
32 windows; new locked Wiki test294,912 /24 windows; new PG19 test294,912 /
64 windows. Exact ranges and hashes require real tokenizer preflight before model
allocation. Wiki test observed total capacity301,829; new24-window test ends301,056.
This smaller6144-token Wiki test is forced by remaining unused packed suffix;
report its limitation and per-window paired differences, not broad quality proof.
New offsets do not establish universal semantic disjointness/pretraining cleanliness.

Validation every256 diagnostic only; no checkpoint selection. Original Qwen,
materialized recovered start and final2048 on BOTH tests, save all window NLLs.
Compare to measured recovered start, not historical7.3. Do not reuse all test
windows or select best checkpoint to inflate main result.

Primary gate: all four final trajectories complete and finite; DECODER-CE >=0.1
Wiki NLL better than ADAPTER-CE AND >=0.1 better than its recovered start at BOTH
seeds. PG19 must not worsen versus own start by >0.1 in either seed. A transfer
confirmation gate additionally requires >=0.1 PG19 control advantage in BOTH.
Record absolute gap to Qwen. Primary success is evidence of improved recovery,
not full teacher capabilities, best initializer,14B efficacy, or token-efficiency win.

If no win: the particular short-budget full-decoder strategy failed, not the
entire paper. If divergent/OOM/unfinished: resource or optimization failure,
not scientific rejection. If win: confirm under longer/fresh held-out evaluation
and total-compute controls before14B; do not return to blind rank/anchor sweeps.

## Required implementation checks before launch

1. Offline plan and exact source/data SHA validation; no user secrets in artifacts.
2. Real JAX CPU tiny-model update: trainable leaves change, input/output embeddings
   stay bitwise identical; stable norm finite; LR/count exact after binary resume.
3. Eight virtual-device mesh: FULL optimizer/masters layouts follow parameter
   shards; peak memory estimate includes gradients, forward weights, activations,
   compilation and source/teacher residency. Validate optimizer exclusion of frozen
   embeddings; refuse unsupported HBM estimate instead of silently reducing scope.
4. Actual initial kernels equal adapter materialization; finetuning updates survive
   FP32 masters -> BF16 cast; no donated/reused buffers invalidate common source.
5. Mock interrupted full checkpoints, auth/429 deferral, file checksums and resume.
6. Record real checkpoint size, available RAM/disk, measured upload timing and
   remaining save reserve. Use chunked checkpoint payloads, no huge JSON states;
   source checkpoints read-only. Original-Qwen model released before full training.

Stem `extent-m3q-paper-decoder-recovery`, separate HF prefix
`experiments/exp104-paper-decoder-recovery`. Same notebook setup retained; final
import/main cell in the launch document. Preparation does not establish a TPU result.

## Implementation update (6 October)

Runner `scripts/m3q_paper_decoder_recovery_campaign.py`; numerical core
`extent/decoder_recovery.py`, binary storage `extent/chunked_checkpoint.py`.
Matrix/tensor decay includes both adapter A/B; named biases are excluded even when
multidimensional (Mamba B/C bias). New Adam and schedule counters are
checked against the restored cursor. Source stores cannot upload. Source binary
SHA/contracts and data hashes are checked before production weights are allocated.
Teacher device arrays are released; two frozen BF16 sources are cached on HOST.
Initial dense BF16 materialization is bitwise checked; start window NLL parity
checked at1e-4. Final embeddings/readout checked bitwise.

Actual token preflight completed locally with pinned tokenizer/datasets; hashes
in `EXP-104-data-preflight.json`. CPU abstract production-shape/virtual-eight-device
memory preflight in `EXP-104-memory-preflight.json`:1,364,139,136 internal trainable
parameters,311,164,928 frozen vocabulary parameters; raw checkpoint16,369,669,640B.
Conservative per-device estimate12,321,462,528B includes current/proposed states,
gradients/updates/BF16 forward and6GiB POLICY allowance for workspace/compilation.
It is NOT measured TPU HBM. Runtime also checks reported capacity, compiler memory,
host RAM/disk, and logs allocator/upload sizes/timing. Refuse unsupported capacity.

Eight virtual-device CPU tests cover both real tiny-model compiled updates,
FP32->BF16 updates, frozen vocabulary, exact next-step binary resume, preserved
moment sharding, checksum corruption, interrupted save, auth/429 handling,
mock campaign deadline/resume and unavailable gate for incomplete branches.
No real full-size TPU allocation, throughput or full source-payload local validation
is claimed. Checkpoints use <=128MiB chunks with bounded upload batches; manifest
advances only after all chunks. Local obsolete generations are removed only after
atomic new manifest publication; immutable remote generations remain for durability.
HF summaries cannot advance beyond durable optimizer checkpoints. A deferred upload
is explicitly pending; it cannot promise survival of unsynced local progress after
Kaggle destroys the session. Same cell resumes the last durable remote cursor.

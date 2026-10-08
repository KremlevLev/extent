# EXP-106/107: long decoder recovery, global versus layer-group adaptive clipping

Registered 8 October 2026, before TPU outcomes. PREPARED, not a measured win.
Two accounts, one TPU v5e8 invocation up to eight hours on each. This replaces
the proposed long Mamba-only scope comparison; both accounts now train the same
full internal decoder. Only clipping differs between accounts.

## Evidence and question

EXP104/105 completed four trajectories each at2048, in1.071/0.970 runner-hours;
both primary gatesFALSE. Dense/Mamba own-start Wiki improvements are only
-0.0034/+0.0214 and-0.0054/+0.0246 for seeds123/456. Seed456 comparative wins
mostly reflect adapter-control deterioration. Validation has severe individual
window spikes, including105 seed456 step512 window NLL22.512. Finite arithmetic
does not imply smooth learning. Fixed validation inputs mean these spikes cannot
be attributed to random evaluation sampling.

EXP097 established norm-only overflow and numerical benefit of safe clipping.
Current SAFE uses a FIXED global threshold1, not adaptive clipping. EXP080–088
already tested coordinate sweeps, acceptance and pair/downstream variants;
they do not justify rerunning an unchanged layer sweep.

HYPOTHESIS: one decoder layer can dominate the global gradient norm and suppress
other layers before Adam. Independently bounding each layer relative to its
weight norm may improve recovery. Adam partially cancels gradient scaling;
the mechanism therefore needs measured POST-ADAM update norms, not only
gradient-retention fractions. Smoother curves alone do not count as recovery.

Literature: Attention to Mamba, https://arxiv.org/html/2604.14191v1 sections3.2/4,
uses full-model CE finetuning with vocabulary frozen and9B stage2 tokens.
We adapt that training scope; we do not reproduce HedgeMamba or its token budget.
AGC reference: https://proceedings.mlr.press/v139/brock21a.html.
Our clipping is LAYER-GROUP relative clipping, not the paper's unitwise NFNet AGC.
No new bridge, EMA threshold, layer selection, KL or larger batch is introduced.

## Matched arms and source

- EXP106: DECODER-CE, stable scaled FP32 global norm clip1.
- EXP107: DECODER-CE, independent layer-group clipping:
  `limit_l = 0.01 * max(norm(weights_l), 0.001)`.
  Each top-level decoder layer forms one group including its Mamba/GQA/MLP/norms;
  final decoder norm is a separate group. No subsequent global clip.
  Gradient direction within a group is preserved; NaN/Inf are not repaired.

Both seeds123/456, exact pinned EXP098 SAFE-PROTECTED-R32 final16384 from
EXP-101-103-source-manifest.json. EXP104/105 final states were not saved in HF
and are unavailable after cloud runtimes disappeared; do not attempt to continue
their final2048 state. Copied source layer/binary SHA guards remain active.
Materialize the same recovered BF16 student, train all1,364,139,136 internal
parameters with FP32 masters/moments. Vocabulary311,164,928 parameters stays
source-BF16 and outside optimizer. Canonical initial reconstruction/evaluation
parity stays1e-4. Cross-account comparison verifies start window parity too.

Each account has one method x two seeds; the control is on the other account.
These are matched experiments, not independent replications. Do not compare
different final token counts or select the better seed.

## Training, data and budget

CE only, AdamW beta(.9,.95),eps1e-8, matrix/tensor decay.1 excluding vectors
and named biases; fresh moments at transition in BOTH arms, then exact resume.
Peak LR1e-5, warmup512; cosine to1e-6 at32768. This is a new registered schedule,
not an extension of the completed2048 schedule. Same schedule in both methods.
Batch1/context256/no accumulation.32768 steps per seed,8,388,608 input tokens
per trajectory;16,777,216 per account,33,554,432 across both registered campaigns.
These are input tokens; CE predicts255 labels per window.

Horizon blocks4096/8192/.../32768, rotating seed order each block. Checkpoint at
every4096 and any deadline/failure boundary; validation every1024. Frozen Wiki
and PG19 endpoint metrics are retained at every4096 for matched partial analysis,
but cannot select updates/LR/best endpoints. Confirmatory gate uses final32768.

Pinned Qwen tokenizer/dataset revisions unchanged. Real local tokenization and
capacity/hash preflight passed; EXP-106-107-data-preflight.json is binding:

| Split | Offset | Windows x256 | Interpretation |
|---|---:|---:|---|
| Wiki103 train |40,370,176|32768|After104/105 train range; shared between methods/seeds|
| Wiki validation |114,688|32|Reused diagnostic data|
| Wiki test |294,912|24|Reused, small exploratory set6144 tokens|
| PG19 test |327,680|128|New primary range32768 tokens, after prior registered suffix|

New packed offsets do not prove document-level/pretraining independence.
Do not train on evaluation data. Test windows remain correlated within books.

Eight-hour invocation includes1.5h save reserve; training/loading/compilation
deadline6.5h, deduct notebook setup with --max-wall-hours. TARGET5–6.5h useful
work, not a runtime guarantee. The measured short campaign1h for8192 total updates
motivates65536 updates/account: nominal8x total updates, rather than an unchanged
one-hour horizon. Dense share, diagnostics, setup and uploads affect actual time;
this estimate does not establish eight hours or one-session completion. Expect
deadline-partial if necessary; continue same contract/moments/data cursor next
session. Current speed and actual update counts are logged. Never add repeated
data or extra seeds solely to consume quota.

## Recorded diagnosis and success rule

Both arms record per-group raw gradient log-norm, weight log-norm, clipping retained
fraction, actual AdamW update log-norm and relative update log-norm. Every-step
summaries count clipped steps and extrema; every1024 diagnostic retains layer
values and all validation-window NLLs. Last health is recorded. After rollback
to an older durable cursor, aggregate counters with lost coverage are cleared.
Mechanism checks are exploratory and cannot substitute for quality improvement.

Primary paired gate at32768: all four finite trajectories complete; at BOTH seeds
EXP107 improves fresh PG19 NLL by>=0.1 over EXP106 AND>=0.1 over its own start;
reused Wiki own-start regression<=0.1. No smoothness-only PASS. Compute gate with
`compare_campaigns(global_result, adaptive_result)`. Individual account gateNULL
is intentional until both files are available. Matched partial milestones are
descriptive with gateNULL; earlier best point cannot replace the registered final.
Report own-start gains for BOTH methods, Qwen gaps and paired window differences.
Two seeds/small PG19 slice do not prove full recovery,14B quality or efficiency.

If both improve similarly: longer full-decoder CE may help, no adaptive advantage.
If adaptive wins: confirm fresh data/more seeds before changing scale.
If only smoother: stability benefit without established quality benefit.
If neither improves materially: stop this particular long CE/clip comparison;
do not promise that another extension must recover the teacher.

## Cloud durability and launch

Import/main within the notebook process, no second JAX process. Existing setup
and secrets remain sufficient. Two independent prefixes:
experiments/exp106-layer-clip-recovery and experiments/exp107-layer-clip-recovery.
Defaults checkpoint into /dev/shm with host/tmpfs checks; source caches on disk.
Tiny complete binary HF save/fresh restore before model allocation. Each seed's
initial FULL state is uploaded before training, with fresh remote manifest and
largest-chunk SHA verification. Every later successful sync has the same remote
verification. This sample is NOT a complete16GB remote reload; exact full-state
resume is verified on compiled tiny CPU models, not yet on the real TPU states.

Large uploads batch up to2GiB/128files per commit, immutable chunks then manifest;
both accounts share HF repository limits.401/403/auth and SHA failures stop.
Transient retries preserve receipts;429 cooldown1h is respected even on final
force sync. Initial small/full probes have bounded15/20min retry windows.
Final retries use available1.5h reserve. Latest metrics can be published ahead of
durable state, with explicit pending_slots/durable_step; a new runtime reconciles
reported cursors against authoritative binary manifests before training.

One full optimizer checkpoint~16.37GB; two latest seeds plus atomic staging need
roughly49GB checkpoint RAM, separately from sources/model cache. Immutable remote
generations are retained: full completion can add roughly300GB/account including
initial states. No old HF files are deleted. Report real uploaded bytes/times.
External VM kill can still lose an unsynced tail; do not claim guaranteed saves.

Telegram default enabled: startup, training ended/stopped, caught error, and
checkpoint saved/pending. Delivery result is in JSON. Same secrets
TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID. Live delivery and real large uploads were NOT
tested locally. Launch cells and files to return: EXP-106-107-kaggle-launch.md.

# EXP143-146 long T4 follow-ups

Status PREPARED; source metadata/chunks verified read-only; no new GPU run.
Four accounts, each2xT4/12h; one independent branch per GPU. No135 rerun.
All standalone cells: EXP-143-launch.md .. EXP-146-launch.md; combined launcher
EXP-143-146-launches.md. New modules only; old135-142 contracts unchanged.

## Source and experiment matrix

143: continue136 RESET-ALL current12586 and RESET-PROJECTIONS current12051.
Preserve original correction factors, pregate weights, Adam moments and allsix
optimizer/schedule clocks, data cursor and full ramp strength. Source NLL A/B/C:
ALL6.000789/6.004432/5.617432; PROJ6.112766/6.179138/5.728322.
Question: how much further does the successful reset recover? Full/projection
sources have unequal earlier step counts, so compare both matched NEW budgets
and registered common original136 data-clock endpoints; do not conflate them.

144: two copies of138 CE current11965 (6.000900/5.992123/5.566396). Same source
weights, LoRA factors, Adam/moments/data clock for both. Control originalMambaLR3e-4,
pregate1e-4; second ALL LR groups divided by3 (Mamba1e-4,pregate3.333333e-5).
No warmup/Adam restart: inherited schedule11965 is already flat. Tests lower LR
at the current quality, not a retrospective validation-selected LR pilot.
PureKL loses138; it is not assigned a long continuation here.

145: common136-ALL current12586. BOTH arms get NEW Adam/warmup256 while SAME
original136 data cursor12586 and fully-active surgery are preserved. Arm0 retains
trained LoRA factors; arm1 folds effective Mamba corrections/reset into BF16 base,
starts new zero-effect rank32 factors, copies learned pregate scales, and disables
repeated reset (already baked). Frozen Qwen embedding/GQA/MLP unchanged.
Both starts must replay every source A/B/C window within.03 before accepting any
new update. Distinguishes factor reparameterization/new correction capacity from
Adam reset, with approximate BF16 folding equivalence; does not isolate whether
capacity or changed optimization coordinates accounts for any benefit.
No inherited source best is reinterpreted as a successful new-stage update.

146: reproduce136 FULL vsPROJECTIONS protocol from independently PREPARED
seed456 EXP111 MAIN32768. Prepared069/072 model and111 corrections use456;
source base digest, endpoint contracts and main manifest checked at immutable
revisions from EXP114/EXP111-113 pins. Fresh Adam/cursor0; ramp2048, same as136.
Data permutation and reset generator130123+layer stay shared with123 to isolate
sensitivity to preparation seed. Label model/preparation-seed replication, not
an all-RNG/independent-corpus replication. Source zero-strength NLL is checked
against originalPatient first window of EACH A/B/C panel within.03.
Compare146 with original136 at matched post-start full-strength checkpoints.

## Immutable provenance

EXP-143-146-source-pin.json fixes HF dataset levkremlev888/extent at
 e5a25058a543ff5e5811a8e676b924a60a32d029, three current checkpoint manifests,
full old contracts, per-window reference NLL and runtime. All1461 actual chunks
344,939,220bytes checked SHA256 locally; complete q/Adam/best restored using real
optimizer templates without the1.7B model. Source audit records allsix counters
12586/12051/11965 and correct previous best12288/10240/10240. No remote writes.
Seed456111 manifest independently fetched and matched existing immutable pin.
Underlying original numerical code hashes checked after CRLF->LF normalization;
raw downloaded model/chunk/manifest hashes remain byte-exact. New code/descriptor
hashes normalized likewise, avoiding Windows/Linux line-ending contract drift.

Reconstruction shares frozen host MLP/embedding arrays rather than deep-copying
multi-GB NumPy buffers. Folding modifies/copies Mamba subtrees only; exact tiny
array/function parity is checked against the original folding implementation.

## Shared training and registered evaluation

BF16 resident frozen weights + host-activation, per-layer VJP; FP32 correction
masters/Adam/recurrence. Existing working streamed GPU engine unchanged. Batch1,
context256,255 targets; same16384-window pinned FIT pool, original training_index
permutation. Stableclip1, Adam.9/.999 eps1e-8, original Mamba3e-4/pregate1e-4,
flat after warm256, except144 registered LR/3. No full-decoder/dense/teacher KL here.
Source130 and old111 preparation remain inherited cost, not free initializations.

step=original source-run data cursor (136 or138), new_steps=step-source_step.
Neither is full-pipeline cost.145 keeps source data cursor despite fresh Adam;
146 starts0 after already-trained111 source. Evaluation at registered NEW counts
0,128,512,1024,2048,2560,3072,3434,3584,4096,4608,5120,6144,8192,10240,12288,
16384,20480,24576,28672,32768 and larger points to131072. Additional common GLOBAL
source-run endpoints16384,20480,24576,28672,32768,40960,49152,65536,81920,98304,
114688,131072 let143 compare equal total136 updates despite unequal restart steps.
Source current becomes registered baseline; previous selected best remains in
immutable source metadata, while new-stage best is selected onA64 at these
registered full-strength points. Before ramp2048, A16 monitors only. B/C never
select. Arbitrary last separate. Do not compare A16/A64 as identical panels.

Source/periodic/final/best layer profiles; reject diagnostic mechanism claims if
ordinary/profile discrepancy>.02, keep ordinary NLL as quality metric. Final
context128/256/512/1024 matched last127 targets on8 windows. First finite proposal
persisted before adoption; all stages checked. Failed proposal does not advance
weights, Adam or data. No NaN masking, clipping guard removal or threshold relaxation.
Long improving curves can reverse: no extrapolated convergence/quality claims.

## Duration, delivery and quiet notifications

11.7h runner includes30min save/final-evaluation reserve,18min notebook margin.
Work deadline11.2h includes setup/reconstruction/JIT/eval. Additional131072 ceiling
ensures long sessions do not stop after completing a tiny schedule. Continue as
wall time allows, no guaranteed32k. Local128 steps, remote120min+exit, one current
q/Adam plus selected-best checkpoint generation,4GB new-upload cap per SESSION
split2GB perworker. Total cap16GB for allfour; not mandatory storage usage.
Source/frozen models not reuploaded; no remote cleanup. Abrupt VM destruction can
lose local unsynced tail. Same cell resumes OWN experiment namespace expID-long-v1.

Parent-only Telegram: start,at mostoneprogress/4h,onefailure,onefinal; hardmax5
perinvocation including reserved final. Normal11.7h run4 messages, not per-worker
messages. Children forced--no-telegram even for direct worker CLI. Logs/JSON remain
detailed and throughput appears after8 accepted steps without Telegram spam.
Parent includes both worker steps/best in sparse progress, child errors/log tails
in one aggregate. arm_failed is a visible parent failure rather than green Kaggle.
Each account returns ONE extent-m3q-long-ID.json, full states synchronized separately.

Short GEMM-only probe onworker0 measures BF16/FP16/FP32 forward for two MLP-shaped
matmuls. Compile/first call separate, ready synchronization, five timed samples,
no model, no loss scaling, no changes to training dtype. Report device/platform.
Cannot infer full-step speedup, memory or numerical safety of FP16 from these ratios.
Benchmark is optional engineering evidence; failure logged without rejecting training.
Physical new-source1.7B GPU HBM/throughput still awaits the user run. All results
from local tiny models or source restoration are engineering evidence only.

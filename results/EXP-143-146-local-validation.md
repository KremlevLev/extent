# 18 October2026: HF download resilience regression

36 CPU tests passed:18 in test_hf_download_retry.py + test_hf_artifact_sync.py
(2.08s),18 in test_reset_long.py + test_hf_checkpoint_sync.py (26.35s).
Fault injection includes8 consecutive429s then success with exact args/pin,
Retry-After seconds/date, deadline inside a long cooldown, 503/network timeout,
401/403/404/integrity rejection, safe logs, atomic artifact/legacy checkpoint
restoration, idempotent SDK installation. TWO real subprocesses share the same
file-locked pacing/cooldown state and recover from simulated429. A separate
preemption regression verifies that a delayed worker still reserves its slot. No real HF
traffic or Telegram messages in these tests. All four complete contract_for()
outputs equal frozen EXP-143-146-engineering-plan.json exactly. No numeric
kernel, dataset window, source revision or optimizer change. Source replay/SHA
checks remain active. Actual Kaggle/HF429 recovery remains to be observed after
relaunch; CPU tests are not a physical T4 training result.

# EXP143-146 preparation validation (2026-10-18)

No cloud model training launched. Scientific quality/speed remain unmeasured.

14 new cases passed33.98s; subsequently addedLR-div3 case passed6.31s (15 total
new cases covered). Earlier combined long/streamed set25 cases passed158.43s.
Final parent/notification/CLI subset4 passed.86s after metadata cleanups.
Four standalone launch cells parsed withAST, old135-142 contracts compared equal.
New frozen plans generated for final code, with canonicalLF code/descriptor hashes.

Coverage: actual checkpoint chunk restoration with real optimizer template; corrupt
chunk rejection; correct original six Adam/schedule counters; stage145 fold preserves
initial function and shares frozen large MLP/embedding buffers; lightweight source130
fold array-identical to original; seed456 constructor routes prep/main/contracts/digests
to456; seed146 full/proj initialization identical to original recipe at strength0/1;
real orchestration starts with old Adam/data cursor, NEW128 evaluation at global
source+128, rejects bad proposal before accepting, finalizes/saves on deadline;
optional precision-probe exception does not halt training; actual parent spawns two
GPU-isolated--no-telegram workers; arm_failed parent failure visible; direct worker
also forces Telegram off;720 simulated minute events stay below hard5-message cap;
small ready-synchronized timing probe reports its narrow forward-GEMM/device scope;
LR/3 changes updates by1/3 while preserving Adam state and schedule counts.

Read-only REAL source audit (not synthetic source fixtures): all1461 chunks
344,939,220bytes SHA verified, payloads q/Adam/best restored at12586,12051,11965;
all arrays finite, correct best12288/10240/10240,7,182,336 correction/norm params,
114,917,404 payload bytes each. On each REAL source, one CPU synthetic-gradient
optimizer step exactly matches old recipe and increments allsix counts. This is
optimizer wiring evidence, NOT a full model gradient/quality/physical GPU test.
All48 old069/072 seed456 preparation metadata contracts and111 seed456 MAIN32768
manifest checked independently. Binary prepared456 model state reconstruction
still performed/validated by the real cloud run, not assumed proven here.

Initial test adaptations caught a mock claiming two GPU devices for a one-GPU
worker; corrected to match real dispatch. A benchmark timer using low-resolution
Windows monotonic could give zero for tiny CPU kernels; actual timing now uses
perf_counter. Code/manifest raw source bytes retain original SHA; descriptor/source
text hashes normalizeCRLF->LF to compare Windows versus Linux safely.

No precision switch, GPU-memory-guard bypass, NaN masking, threshold relaxation,
source overwrite, remote deletion, or fullmodel upload. Source replay.03 and normal
finite proposal gate occur before each new-source trajectory can accept a step.
The full1.7B new starts, seeded source reconstruction, first real model gradient,
physical GPU HBM/speed, long-run stability, final upload and NLL remain pending.

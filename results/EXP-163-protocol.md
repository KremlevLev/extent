# EXP163 — fresh-Adam capacity recipes with output KD

Prepared26October2026 for one14h RTX PRO6000Blackwell96GB invocation, adjustable
and softly resumable. CPU engineering evidence only; no server job launched here.

## Observation and mechanism

EXP161 C4 CURRENT154852 improves held-out C4/EN/RU while frozen rank256 plateaus
on other panels. EXP157 full decoder CE on cyclic Wiki deteriorated; it did NOT
test full-decoder KD on a new broad corpus. Earlier MLP-down LoRA32 interventions
are not the same as opening all gate/up/down matrices in the top eight MLPs.

Hypotheses: (1) existing limited corrections still benefit from further KD;
(2) inherited upper MLPs cannot use transformed hybrid representations, requiring
their own adaptation; (3) a broader decoder change is required, but with KD and
smaller learning rates than157. This is comparison of practical recipes, not
isolated proof that parameter count or a particular layer causes the plateau.

## Same starting function, three explicit methods

All start from immutable EXP161 KD-C4-FRESH CURRENT154852, not old BEST118304 or
paused162. Source descriptor, report/chunk manifest SHA and CURRENT A-D per-window
references are frozen. Existing155 is used only to reconstruct frozen base and
shape templates; actual current parameters are restored from161 before conversion.

All use fresh Adam(.9,.999), warmup512 accepted updates, stable global norm clip1,
batch6/sequence256 and .5true-token CE+.5forward KL(Qwen||hybrid), temperature1.
No hidden auxiliary loss, memory reset, residual cap or rank growth. Architectural
Mamba MIMO rank remains4; adaptation rank256 is separate.

| Method | Trainable coordinates | Peak LR |
|---|---|---|
| R256-CONTROL | same protected rank256 factors + pregate | mixer3e-4, pregate1e-4 |
| UPPER8-MLP | same + all three MLP matrices in blocks20..27 | same mixer/pregate; MLP1e-5 |
| FULL-KD | all28 decoder blocks incl Mamba dynamics,4GQA,MLPs,block norms + pregate | mixer/GQA1e-5,MLP1e-5,norm/pregate1e-6 |

Embeddings and final norm/head remain frozen/source-identical in all methods.
Upper8 adds302M parameters approximately; all-layer full decoder is about1.364B.
FULL materializes learned rank32+224 corrections in source FP32 addition order,
one BF16 cast, then FP32 masters. No double application of factors. Upper MLP
masters are copies of original BF16 MLPs; the existing Mamba factor tree is retained.
Both expanded representations MUST replay source A-D within.03 per-window before
ANY accepted update. Failed replay aborts; never restart from random/original155.

Optimizer reset in ALL arms is intentional: low-rank moments do not directly
represent dense-weight Adam moments. This does not test preserving versus resetting
Adam. Global model phase starts154852, fresh optimizer count0. For n updates:
global154852+n, Adam n, Edu cursor6n, new inputs1536n, targets1530n PER arm.
Old161 moments are verified when reading source but NOT reused. First proposal
audit compares actual compiled gradient to mean of six fixed-q streamed gradients
followed by ONE Adam update; relativeL2<=.15,cosine>=.98,CEerror<=.03 gates retained.
Memory/gradient failure falls back to explicit accumulated batch6, not batch1.

## Data and equal-cost comparison

All three see the SAME FineWeb-Edu windows in the SAME order, starting at0.
Use the pinned100M Edu pack prepared for162; same exact-document exclusions,
tokenizer, EOS packing and C4 D/E validation. If partial, preparation resumes;
if absent, bounded CPU preparation runs before GPU model loading. No new C4
training branch, Wiki cycling or switch between corpora within this experiment.
New data corpus does not imply that all documents were unseen during donor or
earlier student pretraining. Exact exclusion is not semantic decontamination.

Both remaining wall time and actual method speed determine achievable equal
token cost. At each invocation first16 accepted pilot updates per arm are counted
as real training; first audit/compile latency excluded from last-six timing median.
Largest method first. Compute largest common target T satisfying
sum(max(0,T-current_new_updates_i)*median_seconds_i) <=.85*remaining_work_seconds.
Record medians and target in that invocation, then alternate blocks<=256updates,
choosing least advanced. When target reached, extend while remaining work allows;
it is NOT an artificial early-stop quota. Fixed hard ceiling is available Edu
data (65104whole batch6 updates/99,999,744 inputs perarm for100M pack), never cycle.

Rates can change; .85 is an estimate allowance for saves/evals/transfers, NOT a
completion guarantee. Time stop may leave unequal last tails. Compare common
registered endpoints, not equal wall time or incomparable last points. Repeated
soft pauses remeasure pilot rates without resetting q/optimizer/data clocks.
If a method fails numerical health, stop comparison and save other states;
do not silently drop the failed arm and call another method the winner.

## GPU memory and durability

Three full optimizer states must not all occupy VRAM. After initial source replay,
inactive q/Adam/best are losslessly offloaded into host NumPy arrays. Move only
active state back to GPU at block transitions. CPU RAM~500GB available by user;
transfers/checkpoint/load overhead is real and affects wall throughput. Frozen
weights and teacher resident as before. First-proposal memory guard and accumulated
fallback remain; GPU96GB success/speed is not established by CPU tests.

Local current generations pruned at30min+exit; HF sends small metrics during
work, full states ONLY at exit/soft pause,32GB session upload cap. No repeated
four-hour dense generations. Work80GB/free20GB default caps. Full state inherently
about22GB; total child state about29GB. Atomic replacement temporarily holds an
old and new generation; estimated work peak~60GB with received parent's~10GB,
not a guarantee about later caches. If cap/network prevents HF completion, save
locally and report pending explicitly.
Do not delete own states or parent161 state. Frozen base/corpus not uploaded.

Source replay/control, final CURRENT and BEST all compared on common A/B/C/D/E.
C4 D selects BEST for all methods; it is diagnostic, not an untouched test.
EN/RU16 progress,128final; keep panel sizes distinct. Include original Qwen D/E.
Lower NLL does not establish generation or benchmark capability recovery; a winner
needs a subsequent locked quality assessment and later seed confirmation.

## Operation

New engine_methods.py, runtime-methods.json/status/log,states/runtime-methods,
methods/source.json,HF experiments/exp163-method-sweep-v1. Old160/161/162 scientific
files remain unchanged and resumable. Optional engine_launch --runner methods
routes to163; default compare continues162. Same server.lock/STOP prevents competing
launches in this work dir. STOP saves all current states before final diagnostics;
--clear-stop resumes OWN163 state, not paused162 or source154852. Wall limit resets
per invocation, accumulated training does not. Only Telegram start/finish; console
heartbeat5min. Return one runtime-methods.json, not checkpoint chunks.

References: https://github.com/jxiw/MambaInLlama supports opening MLPs during
end-to-end KD; this is motivation, not a replication or proof for Mamba3MIMO.
https://goombalab.github.io/blog/2024/distillation-part1-mohawk/ describes staged
cross-architecture recovery; these163 methods are a separate controlled probe.

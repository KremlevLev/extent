# EXP162 — same C4 CURRENT, continue C4 versus FineWeb-Edu

Prepared26October2026. No physical server job launched during preparation.

## Observation and hypothesis

Received EXP161 CURRENT C4 step154852: A/B/C5.958907/6.023339/5.602312,
C4 D5.231273; EN/RU1284.616741/7.762478. Parent Wiki CURRENT154749 is
better on reused Wiki, worse on D/EN/RU. Final D slightly rebounds from5.203392
at153187; last matched EN16 still improves4.663120 to4.651509. This is progress
with domain tradeoffs, not proof of a universal NLL, downstream recovery or a
permanent plateau. FineWeb-Edu quality filtering motivates a test, not guaranteed
superiority for cross-architecture recovery.

Question: at equal additional token budgets and identical optimization, does a
switch to FineWeb-Edu improve cross-domain recovery beyond continuing C4?

## Target/control

One RTX PRO6000Blackwell96GB, user-selected adjustable wall limit (default21h,
including CPU preparation, load, evaluation and final18min save reserve). Both
arms bootstrap EXACT SAME EXP161 KD-C4-FRESH CURRENT154852 and full Adam state.
Never bootstrap Wiki arm, global118304 BEST, original155 or fresh Adam instead.
Original report/checkpoint contracts and CURRENT reference required. Pin local
manifest SHA and report SHA in corpus-switch/source.json; report includes this
plan and source-state metadata. Parent files remain read-only.

C4-CONTINUE resumes unused C4 window208713 onward. EDU-FRESH starts Edu window0.
Same batch6, length256, protected rank256+pregate, frozen teacher/backbone,
same .5CE+.5outputKL, original LR/beta2/global clipping, Adam moments/count and
schedule. Mamba architectural MIMO rank stays4. No hidden-state objective, rank
growth, dynamics release or LR reset. Initial Adam count36548; each accepted
six-window update increments count/global once and data cursor6.

Reset only BEST bookkeeping to CURRENT154852 so both select on the SAME C4 D
criterion. CURRENT parameters/Adam are unchanged. C4 D selection is pragmatic,
not a claim D is an untouched test. LAST remains independently saved and shown.
Compare all common validation panels; do not select winners using only D.

Alternating blocks<=256 updates; compare registered common new-update/token
endpoints. Finite work limit can leave unequal last tails; do not treat those as
exactly equal-cost. Sparse history and15min checkpoints are not per-update live
heartbeats. Compilation audit checks actual batch6 gradient against six fixed-q
microgradients followed by ONE Adam step; existing BF16 gates/fallback retained.

## Data and evaluation

Original pinned C4 pack remains unchanged, with100M train positions. At the fork,
208713of390625windows were consumed:30318full six-window updates remain, giving
46,568,448 additional inputs per arm (93,136,896 summed). Four final C4 windows
cannot form batch6 and are unused. This is a DATA limit, not an8192-step early
stop. A shorter user wall limit may stop earlier. Never cycle either stream.

Edu: HuggingFaceFW/fineweb-edu, immutable revision
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, sample-10BT source files000..013.
Prepare at most100M Qwen-tokenized train positions plus131072validation positions.
First4096raw documents are reserved as validation source, rest training; exact
normalized-document dedup prevents within-Edu validation leakage. Edu validation
excludes ALL exact C4 document digests (past and future C4 training included).
Edu training excludes C4 validation digests. Wikipedia URLs excluded by the
existing packer. These are exact-document controls, NOT semantic deduplication
or benchmark decontamination. Existing binary format name extent-c4-packed-v1
is reused as an encoding; manifest dataset/revision explicitly identify Edu.

One raw compressed Parquet shard at a time, removed after processing. Existing
SQLite packing journal/fsync/truncation resume retained. Tokenizer pinned to the
same Qwen revision. C4/Edu identities, file SHA and code hashes fixed in contract.
No model targets cached, full Edu dataset downloaded, corpus uploaded to HF,
or second frozen model download created. Storage budget80GB/free20GB, same caches
and parent states; temporary source shard included in corpus8GB budget checks.

COMMON panels: A/B/C historical Wiki, D original C4 D128, E Edu128. Both parents
replay A/B/C/D at CURRENT within.03 per-window before any accepted update; E gets
same-source baseline, teacher E reference also computed. Original Qwen references
for D/E use identical windows. EN stories/RU Wiki: same fixed diagnostic16windows
during progress,128at final; never connect different panel sizes as one curve.
Reused and inspected diagnostic validation is NOT a fresh confirmatory test.
No generation/QA competence conclusion follows solely from these NLLs.

## Operational isolation and decisions

New neutral engine_compare.py; results/runtime-compare.json,
states/runtime-compare, experiments/exp162-corpus-switch-v1. Original160/161
scientific files/hashes unchanged. Shared server.lock/STOP prevents simultaneous
training processes on this work directory. Soft stop saves before final diagnostics.
Local15min checkpoint pruning, HF4h+exit,32GB upload/session cap. Start/finish
Telegram only. Reuse existing retry transport for HF429; pending/error explicit.

Success: consistent advantage at common costs on multiple common panels and
especially EN/RU, supported by final full panels. D-only advantage favors C4's
domain; E-only advantage favors Edu's domain. A mixed outcome motivates a later
mixture test, not an immediate claim that one corpus restores general quality.
If curves still improve, continue own CURRENT with the same contracts and data
cursors. If corpus exhausted, prepare an explicit new pack/contract; never wrap.

Sources: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu and
https://huggingface.co/datasets/HuggingFaceFW/fineweb . Author comparisons are
pretraining evidence, not results of this Extent experiment.

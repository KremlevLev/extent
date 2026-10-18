# EXP147/148: seven-hour continuation screen and missing control

PREPARED, no cloud job launched by this work. User has two accounts with7h T4x2
each and asks to continue145 while143/144 remain running. Old code/contracts
must remain immutable. Two new independent one-GPU arms per account.

| EXP/worker | source | Adam at start | LR multiplier | purpose |
|---|---|---:|---:|---|
|147/0|145 current24965|12379, RESTORED|1|exact trajectory continuation|
|147/1|same|12379, RESTORED|1/3|smaller updates, no restart|
|148/0|136 current12586|0, FRESH|1|repair failed145 SAME-FACTORS control|
|148/1|145 current24965|12379, RESTORED|1/10|conservative continuation|

145 current A/B/C5.782680/5.804467/5.364202, oldbest24874 recorded as provenance.
For new experiment selection baseline use CURRENT24965, not worse inherited
oldbest. All LR groups(Mamba,norm,mlp even if empty) scaled together. Same stable
clip1 Adam.9/.999 eps1e-8 and original warmup256 schedule; exact continuations
keep learned moments and both schedule/Adam counters, so no renewed warmup.
Only148/0 restarts Adam by design; its original learned factors/pregate retained.
Global data cursor is24965 or12586, independently from optimizer counters.

145 frozen base is reconstructed by folding original136 current12586 ONCE into
prepared072 base (111 corrections represented through136 factors), then loading
actual145 learned q. Do NOT fold145 q again or initialize another correction.
Only ancestor136 q chunks are needed for this base; no unused136 moments/best
are downloaded for continuation workers. Current145 q/moments/best restored from
complete pinned manifest/raw chunks, empty GQA dictionaries reconstructed from
model config before optimizer template. Counts checked against12379, not24965.
All source/runtime/code/data hashes remain enforced. Zero-update control bug
must be reproduced/fixed by real roundtrip/eval/update tests before publishing.

Same architecture/Qwen source, BF16 forward +FP32 q/moments/recurrence and same
streamed full-chain gradient. Same pinned FIT pool, training_index(globalstep),
A64/B32/C64 panels. Reused validation, no independent-test claim. Initial NLL
per-window replay tolerance.03 across160 windows, before updates, and first
proposal health persisted before acceptance. Normal source/global data step
24965 follows old last accepted24964; control starts12586. No token gaps/replay
caused by confusing Adam/global cursors. Existing old136+145 cost stays attached.

Register new128/512/1024/2048/2560/3072/3434/3584/4096/4608/5120/6144/8192/etc;
source-relative plus existing original global grid28672/32768/etc. A selects only
registered points; B/C descriptive, final separate. Source CURRENT starts as
eligible baseline. New first goal8192 updates, ceiling65536; time governs stop.
Base recipe/ramp strength remain unchanged. All running143/144 numerical and
resume contracts remain equal to their frozen plans. Do not await143/144 to run
these sessions, but use their later ordinary-continuation results for inference.

Primary147/0 answers whether145 keeps improving;147/1 and148/1 compare matched
NEW-token budgets to147/0. LR changes are screened on exposed panels, so report
search cost and selection, do not claim globally optimal rate.148/0 compares to
archived145 treatment at SAME new counts from source13612586; both freshAdam0,
same data/permutation, originalLR. This separates the tested factor-stage change
from Adam reset but still cannot isolate capacity from optimization coordinates.
148's two arms have DIFFERENT starts; their raw end NLL is not a paired contrast.
No extra seed invented; no independence claim across continuations of same source.

Budget6.8h includes30min saving reserve,12min external setup margin under7h.
Source restore/JIT/eval count toward6.3h work budget. At1452.8s/update, ideal
training-only ceiling~8100 updates; expect less with setup. Never promise30k/7h.
Local128 new saves, remote120min+exit;4GB cap TOTAL/account,2GB/worker. User's HF
repo already~1.9TB: no frozen-base uploads or destructive cleanup. Final state
plusmoments+selectedbest only. Separate exp147/148-fold-continue-v1 prefixes,
source writes forbidden, OWN checkpoint resume supported. Quiet parent-only
Telegram start/4h/final, hardmax5 incl failure; children always noTelegram.
HF429 uses Retry-After/300s+jitter, shared local cooldown; work/save-verification
limits separated to avoid previous falsely pending final manifests.

Launch cells EXP-147-launch.md / EXP-148-launch.md. One aggregate per account:
extent-m3q-fold-continue-147.json /148.json. Local CPU tests and actual source
restore audits are engineering evidence, not completed physical T4 results.

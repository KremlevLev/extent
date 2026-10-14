# EXP-115: aggressive one-seed rank-growth search

PREPARED locally,14 October2026. User explicitly requests a broad, fast search
on one seed first; additional source seeds are for later confirmation. One new
Kaggle account, TPUv5e8,9h. This replaces the earlier proposed two-seed design.
No TPU results for115 exist yet.

## Hypothesis and source

The stable protected rank32 plateau might be escaped by adding correction
directions after recovery, while retaining learned weights and optimizer history.
Static rank64-from-start EXP099 did not show a paired benefit; it does not test
this transition. EXP114 readout and local-rescue observations do not settle rank
growth;115 can run independently of114 continuation.

Start: seed123 EXP111 MAIN32768, exact immutable manifest and binary chunk hashes
from `EXP-114-source-pin.json`, HF revision
`f839c2a898c3e9eff9907ea3cf1e022fe16f3e09`. Reconstruct only this seed's original
ONPOLICY base and recover its rank32 coordinates AND Adam/count. No random-model
fallback, best-checkpoint selection, EXP098 substitution, or EXP114 readout.
Source validation per-window replay1e-4, all corrected initial BF16 kernels
bitwise equal across expansions. Fresh added directions have random A and zero B.
No alpha/rank rescaling changes the inherited contribution.

## Seven prespecified candidates

| Arm | Effective correction capacity | Old32 | Added directions |
|---|---|---|---|
| R32 |32|Continue with inherited Adam, LR3e-4|None; additional-token control|
| G64 |64|Continue|Fresh32, peak3e-4|
| G128 |128|Continue|Fresh96, peak3e-4|
| G256 |256|Continue|Fresh224, peak3e-4|
| ADD256 |256|Frozen, including Adam time|Train only fresh224, peak3e-4|
| FAST256 |256|Continue at3e-4|Fresh224, peak1e-3|
| LADDER256 |32->64->128->256|Continue|Fresh cohorts32/64/128 activate at additional cursors1024/2048/3072|

Each new cohort has independent Adam beta(.9,.999),eps1e-8, fresh moments/count,
128-update linear warmup from0, then constant LR; no decay. The old32 block keeps
its original Adam moments AND32768 count and constant3e-4. One stable global
clip1 applies jointly to the active derivatives before each group's Adam update.
Inactive cohorts have zero derivative, no weight change and no optimizer-clock
advance. Shared random224-direction A prefix across capacities gives nested
initial directions. All candidates retain the same protected in-projection
columns (timestep/decay/trapezoid/angle frozen); no dynamics or MLP unfreezing.

The deployed kernel is baseFP32 + oldAB + active-newAB sums, cast once to BF16.
Do not fold the old correction to BF16 and discard its FP32 coordinates before
growth. Two separate blocks can express a combined correction of rank at most
their summed rank; nominal rank is a capacity bound, not measured effective rank.

## Fast search and promotion

All seven get4096 additional CE updates, context256,batch1, same ordered train
windows. Validation at0/512/1024/2048/3072/4096. Rank32 is mandatory control.
Select top2 finite noncontrol candidates by mean validation NLL at3072 and4096,
lexical arm name breaks ties. No test values, best-validation checkpoints, or
manual candidate switches. A failed/nonfinite proposal is rejected; preserve its
last finite state and continue other candidates. Completion is not a success gate.
Compiler-memory guard failures are recorded as engineering_unavailable and are
not evidence that the candidate's learning method is worse.

R32 and selected candidates continue their SAME states/data cursor to8192,
12288,16384, with no replay if local/durable state exists. The selection rule
is exploratory: a4096-step ordering may miss a delayed improvement. This search
does not prove unselected methods cannot work at longer horizons. Even candidates
temporarily worse than control may be promoted by the fixed top2 rule.

Maximum planned updates with two finalists:7*4096 +3*(16384-4096)=65536;
16,777,216 input tokens /16,711,680 CE targets. Search costs, failed attempts,
replays, compile/restore/eval/upload are reported; branch x-axis shows its own
additional tokens. Equal steps are not equal parameters or FLOPs. Actual TPU
throughput/HBM for rank256 is not yet measured, so one-session completion is a
target. Deadline preserves partial results and the same cell resumes.

## Registered data and read-only preflight

Pinned tokenizer and dataset revisions match111–114. Real capacity and SHA
checked locally without model weights: `EXP-115-data-preflight.json`.

| Panel | Split |Offset|Windows|Role|
|---|---|---:|---:|---|
| Train |Wiki103 train|74973184|16384|After111 train and114 FIT ranges|
| Selection |Wiki103 validation|122880|64|Intentionally reused; never called fresh confirmation|
| Locked Wiki |Wiki103 test|262144|64|Final confirmation for this campaign|
| Locked PG19 |PG19 test|524288|128|After114 locked range|

No exact duplicate windows between checked train/selection/locked panels.
Earlier project/pretraining corpus exposure and independent PG19 books are not
established. Final locked panels are evaluated only after durable validation
choices, at16384 for R32 and finalists, with original Qwen and unchanged111 start
on the same windows. Failed/incomplete finals are missing, not zero. Report each
candidate's paired NLL difference to control and to start. A promising winner
must subsequently repeat on another source seed; no replicated result from115.

## Runtime, storage and engineering checks

Default8.8h includes60min save reserve;12min external setup allowance. Budget
includes source reconstruction, token packing, compile, training, diagnostics
and upload. Check actual remaining Kaggle quota before using this default.

Source models stay immutable and are not uploaded. Only coordinates/Adam/cursors
and metrics. Latest local snapshot every512 updates/branch-end, one remote
generation per active branch at invocation end; default6GB cumulative upload
payload cap/invocation. Approx3.35GB raw final tensor states for all7 arms,
plus serialization/report overhead (DERIVED, not an HF quota statement).
Reports sync periodically; validation choice sync can bypass report interval
without forcing model-state uploads. Known429 cooldown is respected, including
forced sync. Abrupt VM destruction before final checkpoint sync loses local
updates: roll evaluation records back to restored cursor, retain attempted cost,
replay paid selected-path pilots with recorded per-window1e-4 parity checks.

Eight virtual8-CPU tests passed: zero-change expansion/shared prefixes, control
Adam update agreement, frozen old block, protected columns, cohort clocks, exact
next-update checkpoint restore, real tiny branch continuation/source immutability,
validation-only selection and mocked cloud/cap/429 behavior. Plan-only and real
token preflight passed. These do not establish full1.7B TPU feasibility or quality.
An additional cursor regression checks that an old report claiming16384 cannot
skip replay when only a4096 checkpoint survives. Parameter counts distinguish
allocated coordinates from potentially trainable coordinates, excluding hard
protected B columns and the frozen old block.

Runner: `scripts/m3q_rank_growth_115_campaign.py`. Launch cell:
`EXP-115-kaggle-launch.md`. Outputs `extent-m3q-rank-growth-115.json` and summary.

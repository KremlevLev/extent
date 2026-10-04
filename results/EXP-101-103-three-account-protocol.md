# EXP-101--103: registered three-session plateau recovery

Status: PREPARED/HYPOTHESES, no TPU outcomes. One Kaggle TPU v5e8 invocation per
experiment, eight-hour upper budget including45min final-save reserve. Training
deadline7.25h from runner entry includes data, model/source loading and compilation.
Time spent in earlier notebook cells is outside runner and must be deducted with
`--max-wall-hours`; compatible partial runs resume same cell. No completion promise.

## Common warm start and transition

Full frozen BF16 Qwen3-1.7B-Base hybrid,24 Mamba +4 GQA, unchanged EXP072-v2 ONPOLICY
bases for seeds123/456. Start from EXP098 SAFE-PROTECTED-R32 FINAL16384 in BOTH seeds.
Pinned HF revision and original checkpoint contracts/hashes:
`EXP-101-103-source-manifest.json`. Both remote metadata endpoints verified against user results; local full binary
download did not complete (network interruption), recorded explicitly in
`EXP-101-103-source-preflight.json`. Kaggle checks exact binary SHA and coordinate
health before reuse; never silently substitute an unverified state. Do not use099 better-test endpoints,097 states,
a failed branch, or randomly initialized adapters as fallback.

Reuse FP32 rank32 LoRA A/B on identical frozen base, no folding into BF16. Reset
Adam ONCE at the stage transition for EVERY arm, including controls. This tests
continuation with a registered optimizer reset, not exact continuation of098 Adam.
Later session resumes preserve new Adam moments and cursor; no additional reset.
New step0 is the recovered model, evaluated on new Wiki and PG19 tests. Source
base layer hashes, checkpoint SHA, step16384, finite coordinates and zero masked
input B-columns checked before reuse. Releasing the mask leaves initial weights
exactly identical. Other weights remain frozen; released arm can change all Mamba
input projection columns, not every parameter of Mamba.

Stable max-scaled FP32 global-norm clip1, Adam defaults, no decay, context256,
batch1, FP32 coordinates/moments. No nonfinite repair, skipped batches or numerical
optimizer reset. Norm-only overflow diagnostic checkpoint retains pre-proposal
state. Structural errors save and propagate; failed numerical branches terminal.

## Registered campaigns

| Campaign | Arms | Primary vs control | Final new steps per arm |
|---|---|---|---:|
|101 late-distillation|protected32 CE; CE+0.1 Qwen KL; CE+0.5 Qwen KL. All LR3e-5|QWEN-01 vs CE; QWEN-05 secondary|8192|
|102 plateau-learning-rate|protected32 CE LR3e-4 /1e-4 /3e-5|LR-3E5 vs LR-3E4; LR-1E4 secondary|8192|
|103 late-dynamics-release|CE protected32 vs same rank32 with protection removed, both LR3e-5|RELEASED vs PROTECTED|8192|

101 distinguishes LATE anchoring from failed100 anchoring at the original start.
Forward KL is KL(Qwen||student), temperature2 withT^2; CE coefficient1. Teacher
forwards add compute; equal student tokens do not mean equal FLOPs/time. 102 tests
whether late slow fine-tuning helps, distinct from097 lowLR at original start.
103 tests whether preserving dynamics helps recovery but restricts its endpoint.
This does not repeat early unprotected098: stage and LR differ; within103 starts,
LR, rank and data are matched. Two arms only, leaving time for evaluation/save.

Rotate all arms/seeds through horizons2048/4096/8192. Seed123 listed order,
seed456 reverse. Each arm2,097,152 extra input tokens,6,291,456 total with its098
recovery preparation alone; earlier069/072 warm-start work additionally required.
Campaign totals:101/102 each12,582,912 extra student tokens;1038,388,608.
Shared controls across campaigns are consistency checks, not extra independent seeds.
Conservatively expect101 near previous100 runtime,102/103 lower absent unexpected
compilation/loading costs. Measure actual TPU time; no guaranteed finish.

## Data and evaluation

Same pinned tokenizer and WikiText103/PG19 revisions as098--100, exact new packed
ranges verified locally without model weights (`EXP-101-103-data-preflight.json`):
Wiki train offset37,748,736 /8192 windows; validation98,304 /32;
Wiki locked test278,528 /64; PG19 test278,528 /64. All context256.
These ranges start after previous registered ranges. Fresh token offsets do not
establish universal semantic disjointness or absence of pretraining contamination.
Initial expected token hashes enforced BEFORE model allocation; resume hashes fixed.

Validation every1024 is diagnostic, no best-checkpoint/early-stop selection. Original
Qwen and recovered stage2 start evaluated on BOTH tests; finals evaluated only at8192.
Every window NLL retained. Prior7.3 test values use older ranges; compare new finals
against their measured recovered starts, not against an assumed7.3 constant.

Primary gate: all arms complete, primary Wiki final >=0.1 NLL better than control
AND better than its recovered start at BOTH existing seeds. No requirement that
control improve (could degrade). Secondary cross-domain gate requires primary gate
plus >=0.1 PG19 advantage over control and improvement over recovered PG19 start
in both seeds. Secondary arms cannot be retrospectively renamed primary. Report
negative results and incomplete sessions explicitly. No random-init superiority,
14B recovery, downstream capabilities, long-context or speed claim from these gates.

## Kaggle operations and cells

Reuse notebook clone/install/secrets cells; obtain fresh main before imports. Three
accounts can run concurrently: independent experiment HF prefixes/stems. Default
output `/kaggle/working/output`, atomic adapter+new Adam saves every256 steps and
horizons, bundled HF commits at most10min plus final. HF429 defers uploads1h while
retaining local states; transient failures defer120s;401/403 propagate. Check pending
HF flag before ending session. Compatible rerun uses same cell; implementation and
scientific contract drift cause refusal, not silent restart.

Account1:
```python
from scripts.m3q_late_distillation_campaign import main
result = main([])
```
Account2:
```python
from scripts.m3q_plateau_learning_rate_campaign import main
result = main([])
```
Account3:
```python
from scripts.m3q_late_dynamics_release_campaign import main
result = main([])
```
Send full JSON AND summary.md for stems `extent-m3q-late-distillation`,
`extent-m3q-plateau-learning-rate`, `extent-m3q-late-dynamics-release`.
Offline contract: `main(["--plan-only"])`. Shorter wall budget example:
`main(["--max-wall-hours", "7.5"])`.


## Local verification boundary

Eight registered arm entries compiled on real JAX CPU, with nonzero recovered
coordinates, exact protected/released initial kernel equality, protected columns
unchanged after update and exact binary next-step Adam resume. All eight also
passed on eight virtual CPU devices with1/4/2 mesh. Fourteen mocked cloud workflow
cases passed, including nonzero stage2 start, interruption and resume, initial/final
evaluation interruptions, auth/rate limits, numerical failure and data/time guards.
Two additional source/plan checks passed: SHA/base/protected-column guards,
offline plans and completed098--100 contracts unchanged after Git LF conversion.
Actual corpus/tokenizer capacity and hashes verified; source metadata pinned and
verified. Full source binary download did NOT finish locally (see source preflight).
These checks do not verify real TPU HBM, throughput, full model stage2 quality,
or universal improvement from teacher/LR/releasing protection.

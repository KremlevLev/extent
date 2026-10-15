# EXP125–127: dynamic bounds, residual scale and MLP adaptation

PREPARED, 16 October 2026. Three independent Kaggle T4×2 sessions, 12h
allocations. Each runner uses11.5h INCLUDING source loading, compilation and
60min reserve. No physical GPU outcome exists for these recipes yet.
Use the existing T4 notebook; replace its LAST cell. No new notebook required.

## Why these six arms

EXP120 completed all461 static-gain conditions with only tiny NLL improvements.
121 learned24 output gains only;122 jointly trained rank32/gains;123 added
T2 logit KL. All training campaigns stopped cleanly at their session deadline,
with durable states. They do not establish long-horizon failure. Within the
measured range, free gains/KL did not provide a clear advantage. Scale can grow
while NLL improves. Instrumented profiles show late MLP additions absorbed on
one inspected window; this does not prove the sole cause of the plateau.

New question: does an ENFORCED, input-dependent bound change the regime? If so,
does extra MLP flexibility help adaptation? Single seed123 exploratory search,
not independent confirmation or a claim that dynamic scaling itself is novel.

| EXP | GPU0 | GPU1 | Main comparison |
| --- | --- | --- | --- |
|125|R32-CONTROL|R32-REL05|Dynamic Mamba bound versus ordinary rank32 recovery|
|126|R32-REL05-ABS10|R32-REL05-ABS100|Can bounding the entire residual stream help?|
|127|R32-REL05-MLP32|R32-REL05-ABS10-MLP32|MLP adaptation: compare GPU0 with125/GPU1, GPU1 with126/GPU0|

All start from the SAME pinned EXP111 seed123 MAIN32768. Protected old rank32
coordinates are retained; fresh Adam everywhere. No learned scalar gains and
no distillation in these six arms. Source outside trainable corrections stays
frozen. EXP127 adds rank32 corrections to `mlp/down_proj/kernel` in ALL28 layers,
including the4 retained GQA blocks' MLPs; zero-effect initialization, identical
MLP random seed across the two arms. Other MLP matrices/norms stay frozen.

## Exact intervention

For each token separately, across its hidden channels only:

```text
x = incoming residual
m = Mamba(norm(x))
c_target = min(1, 0.5 * max(RMS(x), 0.001) / RMS(m))
m_scaled = exp(strength * log(c_target)) * m
middle = x + m_scaled
```

RMS is evaluated using stable FP32 log arithmetic; smooth numerical floor
approximately1e-12 at zero. No batch/sequence statistics or future tokens.
At full strength, small Mamba outputs stay unchanged; large outputs are capped
to half the incoming RMS, with the input reference floor0.001. Finite-precision
rounding permits small discrepancies from the ideal bound.

Relative bounds alone need not prevent accumulated growth: repeated additions
can still grow the residual. Therefore126 and127/GPU1 ALSO apply per-token
absolute RMS ceilings AFTER mixer addition and AFTER MLP addition, in ALL28
layers (including attention blocks). This scales whole token vectors rather
than clipping individual coordinates. Ceilings10 and100 are exploratory
registered values, NOT established teacher targets or optimal scales.

All caps ramp geometrically from identity to full strength over2048 updates:
`strength=min(step/2048,1)`. The coefficient is exponentiated in log space;
there is no last-step jump from a large linear blend to a tiny coefficient.
The same strength is used for a forward and its backward recomputation.
Gradients flow through BOTH RMS estimates and the cap; no detached gate or
straight-through derivative. Differentiable RMS floors guard zero inputs.

At strength0, all six arms must reproduce the unmodified source within the
registered GPU hook tolerance. A separate full-strength dry forward on FIT8
records initial NLL and limiter activation; it does not select hyperparameters
or abort a finite but poor-quality recipe. The full-strength profile must pass
its instrumentation gate and actual relative/absolute bounds within2% rounding
tolerance before training. Training then starts at strength0.

## Training, comparison and diagnostics

- Batch1/context256, CE true next token, same16384×256 FIT pool and permutation
  `(step*7919+120123)%16384` for all arms. Additional source preparation excluded.
- Horizon32768 accepted updates/arm:8,388,608 input presentations and8,355,840
  next-token targets/arm, two passes over4,194,304 pool tokens. Not a promise to
  finish in one12h session. T4 throughput is measured, not inferred from TPU.
- Adam .9/.999, eps1e-8; shared stable global clip1 BEFORE parameter groups;
  Mamba correction LR3e-4, MLP correction LR1e-4,256-update LR warmup.
- Fixed endpoints0/128/512/1024/2048/4096/8192/16384/24576/32768.
  Before2048, A16/B32 are monitoring ONLY. At/after2048, A64 selects the best,
  with B32/C64 and a layer profile recorded at the SAME step. Pre-ramp states
  are ineligible for best selection in EVERY arm, including the control.
- Source controls A64/B32/C64 and source profile/context are stored explicitly.
  All panels reuse exposed114/115 data; C is NOT a new independent test. Report
  per-panel changes relative to its own source and compare equal update counts.
- Profile: actual input/output/mixer/MLP RMS, max token RMS, relative factor
  min/mean/max, fraction of tokens capped, bound ratios, absorbed nonzero MLP
  additions. Ordinary/instrumented per-layer discrepancy must be≤.02.
  Profiles use one explicit A window, not a population estimate.
- Terminal last and post-ramp best: A64/B32/C64, profile and same last127-target
  context128/256/512/1024. A run stopped before2048 has NO eligible best state;
  its last result is labelled with its partial ramp strength.
- Compare125/GPU1 with125/GPU0 for the primary intervention;126 with125/GPU1
  for an absolute cap;127 with its matched125/126 counterpart for MLP changes.
  Cross-session GPU source/hook errors are recorded; new-seed confirmation
  follows only after a candidate wins. Tiny gains are not declared breakthroughs.

Success requires lower NLL beyond the matched control, preferably across panels,
plus evidence that the intended scale regime changed. Smaller numbers without
better quality are not success. Better quality without recovered context use
is a narrower recovery result. A finite negative result is still informative.

## Durability and limits

Fresh clone/pull of main, existing CUDA/T4 setup, same Kaggle/HF/Telegram secrets.
No `nvidia-smi` or redundant vendor checks. GPU visibility is set before JAX in
two independent subprocesses; `LD_LIBRARY_PATH` is preserved.

Local checkpoints every128 updates: CURRENT corrections + Adam + BEST parameters
and its cursor. Best is for evaluation, not an optimizer-resume endpoint.
Remote every2048 accepted updates or120min and on clean exit. Checkpoint chunks
are SHA checked; uploaded manifests/sample chunks verified by existing store.
Default4GB new uploads TOTAL across two workers per invocation; no full-model
checkpoints, no deletion of historical HF artifacts. This is a traffic cap,
not a guarantee that the account has4GB free. Pending/durable cursors explicit.
Old HF generations remain; a lost cloud VM before sync can require paid replay.

Same cell resumes CURRENT+Adam and restores ramp position from the cursor.
Future metrics are rolled back to the real restored cursor; endpoint caches
include state SHA and phase. Completed runs only retry outstanding sync.
Reserve is for checkpoint upload and endpoint diagnostics, not idle training.
Start/progress about30min/worker completion/parent completion notifications.

One aggregate output per session: `extent-m3q-t4-125.json`,126,127, respectively,
under `/kaggle/working/extent-results/`. Send only those three files.
No need to download the individual worker logs or checkpoint files.

## Common update cell (once per notebook, before the launch cell)

```python
import os
import subprocess
import sys

repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)
```

`git_env` is from the existing deploy-key clone cell. Retain the existing T4
setup cells. Fresh kernel avoids importing an old in-memory runner revision.
11.5h is from runner entry; if setup consumed more than30min of the12h runtime,
reduce `--max-wall-hours` to the actual remaining time.

### Session1 — EXP125

```python
from scripts.m3q_t4_dynamic_campaign import main
result = main([
    "--experiment", "125",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "4",
])
```

### Session2 — EXP126

```python
from scripts.m3q_t4_dynamic_campaign import main
result = main([
    "--experiment", "126",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "4",
])
```

### Session3 — EXP127

```python
from scripts.m3q_t4_dynamic_campaign import main
result = main([
    "--experiment", "127",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "4",
])
```

## Local engineering scope

CPU checks cover token-local bounds/causality, geometric ramp, finite zero-input
derivatives and finite-difference gradient, all six streamed gradients versus
monolithic autodiff, zero-effect MLP initialization, actual MLP gradient,
nonfinite rejection, optimizer serialization and clean timeout/resume. Physical
1.7B T4 HBM, throughput and quality remain unmeasured until the cloud runs.

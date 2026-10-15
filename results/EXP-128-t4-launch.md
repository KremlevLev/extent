# EXP128 — component-wise hard versus soft bounds

PREPARED,16 October2026. One additional Kaggle T4×2 session,11.5h from runner
entry INCLUDING60min reserve. Same existing T4 notebook. Physical1.7B GPU
outcome pending. Separate modules preserve125–127 code and resume contracts.

## Question

RMS scaling in125–127 multiplies an entire token vector by one coefficient.
A few enormous components can therefore shrink small components too. Does
limiting components separately retain useful small signals and improve NLL?

| GPU | Recipe |
| --- | --- |
|0|R32-CHANNEL-HARD-MLP32: individual component hard caps|
|1|R32-CHANNEL-TANH-MLP32: individual component smooth tanh caps|

Same pinned111 seed123 MAIN32768, same trained protected Mamba rank32, same
zero-effect MLP-down-proj rank32 additions in ALL28 layers as127. Same random
MLP initialization, fresh Adam, data order and warmup. No KL or learned gains.
Compare both with127/GPU1 (whole-vector RMS bound10 + MLP adaptation).

## Intervention

For each token, `b=.5*max(RMS(incoming residual),.001)`, computed only across
that token's hidden channels. For each component of the Mamba output:

```text
GPU0: corrected_m[i] = clip(m[i], -b, b)
GPU1: corrected_m[i] = b * tanh(m[i]/b)
```

Then add to the residual. After mixer addition AND MLP addition, in ALL28
layers including retained GQA blocks, apply the corresponding element-wise
absolute cap with bound10. Hard clips preserve components inside their bounds;
tanh also gently changes components below the bound. Neither method scales
small components merely because other components of the same vector are huge.

Geometric identity→full interpolation of coefficients over2048 updates,
identical schedule to125–127. Stable FP32 log arithmetic, tiny-argument tanh
expansion and bounded exponential arguments avoid zero/overflow singularities.
Actual forward remains mixed precision with existing BF16 casts. Gradients
pass through the RMS reference and caps; hard-capped directions can have zero
gradient. No straight-through estimator. At full strength both componentwise
and implied RMS bounds are verified with2% rounding allowance.

This is an exploratory practical comparison. Component caps are stricter than
whole-vector RMS caps with the same numerical threshold: interventions are NOT
matched for total signal removed. A win would motivate a subsequent calibrated
comparison and fresh-seed confirmation, not prove an intrinsic geometry advantage.

## Training, measurements and persistence

Same32768-update horizon, FIT16384×256 pool, permutation, Mamba LR3e-4, MLP LR1e-4,
Adam .9/.999 eps1e-8, joint stable clip1,256 LR warmup as125–127. Both arms have
the same trainable coordinates.8.39M input presentations/arm at full horizon;
not a promise to finish in one session. Compare equal updates/ramp strength.

Source A64/B32/C64 controls explicitly saved. Initial full-strength FIT8/profile
dry diagnosis before training, engineering source/hook/bound checks required.
Before2048: A16/B32 monitor only. At/after2048: A64 selects best at fixed
2048/4096/8192/16384/24576/32768 endpoints; B32/C64/profile at those same steps.
Last/best A/B/C, profiles and identical last127-target context128..1024 at exit.
Previously exposed panels, one seed, no independent test claim.

Profiles report component coefficient min/mean/max, fraction of COMPONENTS
limited, output RMS/peak, absorption of nonzero MLP additions. Fractions in128
are not directly comparable to token-level limiter activation fractions in125–127.
Whole-model ordinary/profile discrepancy guard≤.02; no recurrence trace claim.

11.5h includes loading/compilation and60min save/diagnostic reserve. Local128,
HF2048 or120min and exit; default4GB new uploads TOTAL per invocation across two
workers. Only corrections+Adam+best, no full-model saves or deletion of history.
Same cell resumes current optimizer and ramp cursor; lost unsynced cloud work
can require paid replay. Best parameters are evaluation-only. No eligible best
before full ramp. Source/last/post-ramp-best remain distinct. Start,30min
progress and completion/pending notifications. Frozen125–127 contracts unchanged.

## Last cell in the existing T4 notebook

```python
import os
import subprocess
import sys

repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)

from scripts.m3q_t4_channel_bound_128_campaign import main
result = main([
    "--experiment", "128",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "4",
])
```

`git_env` is from the existing deploy-key clone cell. Keep T4 setup and enabled
Secrets. If setup consumed>30min of the12h runtime, reduce runner wall limit.
Send ONE aggregate file: `/kaggle/working/extent-results/extent-m3q-t4-128.json`.

## Local validation scope

Tiny CPU checks: preserve small hard-capped components next to huge outliers,
tanh values/zero derivatives, identity ramp, both streamed gradients versus
monolithic full-chain autodiff, peak/RMS/profile controls, timeout/resume.
Physical1.7B HBM, throughput and quality pending.

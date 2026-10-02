# EXP-097: same-gradient overflow diagnosis and long paired recovery

Prepared2026-10-03. This is a numerical-stability experiment, not another
initializer search or a full-recovery claim. EXP094--096 show final improvements
but many nonfinite stops; their single aggregate guard does not identify which
quantity became nonfinite. Pre-stop norms reach1e17, but that alone does not
prove overflow of the gradient elements or of the norm calculation.

## Six paired arms, two source seeds

| Arm | Rank | Dynamics projections | Clipping | Adam LR |
|---|---:|---|---|---:|
| NAIVE-R8 | 8 | adaptable | original sum-of-squares norm | 3e-4 |
| SAFE-R8 | 8 | adaptable | max-scaled norm | 3e-4 |
| SAFE-R8-LOWLR | 8 | adaptable | max-scaled norm | 3e-5 |
| NAIVE-PROTECTED-R32 | 32 | frozen | original norm | 3e-4 |
| SAFE-PROTECTED-R32 | 32 | frozen | max-scaled norm | 3e-4 |
| SAFE-PROTECTED-R32-LOWLR | 32 | frozen | max-scaled norm | 3e-5 |

The unchanged EXP072-v2 ONPOLICY full hybrids (seeds123/456) are the warm starts.
Only FP32 low-rank input/output corrections and Adam moments train; base weights
remain BF16 frozen, 24 Mamba+4 GQA layers. No MLA intervention or teacher pass.
Protected in_proj dt/raw_a/trap/angle weights remain unchanged; upstream inputs
may nevertheless change. Zero-change adapter initialization is shared by matched
arms. No EXP094--096 failed optimizer state is reused.

Same text/order and locked-test offsets as EXP094--096: WikiText103 train offset
8,388,608,8192 windows; validation offset49,152,32 windows; test offset32,768,64
windows, length256, batch1. This is reused evaluation, not a fresh confirmation.
Horizon rotation2048/4096/8192, validation every1024, test only final8192 plus
unchanged start. Maximum98,304 student steps /25,165,824 input tokens across12
trajectories. Equal tokens are not identical FLOPs because diagnostics and both
shadow Adam proposals add work. Runtime is not yet measured; default8.25h wall
cap includes30min reserve. Resume the same cell after a partial session.

## Stable norm and matched shadow proposals

For finite gradients g, set m=max(abs(g)), and s=sqrt(sum((g/m)^2)).
Clip without forming m*s or squaring unscaled g: if m>1/s, use(g/m)/s;
otherwise leave g unchanged. Zero gradients use a safe denominator and remain
zero (their recorded log norm uses sentinel-30). True NaN/Inf gradients are not
replaced, skipped or silently repaired.
FP32 log10(norm)=log10(m)+log10(s) stays reportable when the naive norm overflows.
The display norm is saturated at1e38 if needed; the logarithm is authoritative
for finite nonzero gradients.

The original frozen Qwen is evaluated once on the identical locked test to
report the remaining quality gap. It is not used to select steps, methods or LR.

Every step computes ordinary and stable clipping/Adam proposals on the **same
gradient and same existing Adam state**, without advancing the unused proposal.
The first norm-only event requires finite forward values, finite loss/gradient
elements, nonfinite naive norm, and a finite stable proposal. This directly tests
the numerical calculation; differences between separate long trajectories alone
cannot prove causation, since FP32 rounding can make them diverge earlier.

## Diagnostics and durability

Latest health separates logits/hidden activations, loss, gradient elements,
naive norm, current parameters/moments, and each proposal's parameters/moments.
Per-layer forward maxima and per-coordinate gradient maxima/finite flags are
recorded. First norm-only event and first actual failure retain detailed health,
attempted step, input-window cursor, token hashes, and last finite adapter+Adam
checkpoint. Other arms continue after a numerical failure.

Local binary saves every256 steps, bundled HF saves at most once/10min plus
final/caught-failure attempt. HF429 defers locally for1h; no guarantee exists
against losing the unsynced tail on external VM termination. Start/end/error
Telegram callbacks remain enabled. Namespace:
`experiments/exp097-numerical-stability`, summary stem
`extent-m3q-numerical-stability`. Existing EXP094--096 scientific defaults and
checkpoint contracts remain unchanged via their registered legacy engine digest;
EXP097 hashes the current engine, diagnostic step and stable clipping code.

## Registered decision

Diagnostic gate: all12 arms terminal (completed or numerically failed), at least
one matched norm-only event, and SAFE-R8 final test NLL improves unchanged start
at both seeds. Failed controls count as observed diagnostic outcomes, not as
successful recovery. Protected and low-LR results are prespecified secondary
comparisons. A false gate may still show real forward/backward instability
instead of norm-only overflow. No initializer superiority follows from either.

## Last Kaggle cell

Local verification: float64-reference comparisons for gradient magnitudes through
1e38, zero and nonfinite-input handling, a synthetic finite-gradient/overflowing
naive-norm matched-proposal test, compiled steps for all six arms, eight-virtual-
CPU sharded steps, binary optimizer resume, diagnostic event checkpoints before
the offending update, and simulated HF429/interruption workflows. This is local
CPU verification, not measured TPU throughput or a real-model norm-overflow win.

Existing clone/install/secrets cells stay unchanged; clone the new main revision.
```python
from scripts.m3q_numerical_stability_campaign import main as run_exp097
result = run_exp097([])
```
Send `extent-m3q-numerical-stability-summary.md` and the JSON from output/HF.
No checkpoint transfer via the user's computer is needed.

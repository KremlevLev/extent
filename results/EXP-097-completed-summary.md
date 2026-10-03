# EXP-097 completed numerical-stability result

Reviewed2026-10-03 from user-supplied full JSON and summary. No new model run performed locally.
Full artifact SHA256: `ea63119ea8b5b556e9292a14441aa63d2c763cc850db1f5ff4d3e59c16f6f72d`.
Submitted summary SHA256: `fe7d47e0ed7e48b4792a1ff08fe44344e707b2d924cb883f619422f19fe8976f`.
Recorded Git revision: `627c9d98d861d140e9f2090ab0b38869c500ed47`.
Status `completed`; registered diagnostic gate **PASS**, independently recomputed from branch outcomes.
Cumulative invocation duration7.329166h; no pending HF uploads reported.

| Seed | Arm | Completed steps | Start test NLL | Final test NLL | Outcome |
|---|---|---:|---:|---:|---|
| 123 | NAIVE-R8 | 4340 | unavailable | unavailable | failed |
| 123 | SAFE-R8 | 8192 | 12.968006 | 7.288451 | complete |
| 123 | SAFE-R8-LOWLR | 8192 | 12.968006 | 18.128816 | complete |
| 123 | NAIVE-PROTECTED-R32 | 3206 | unavailable | unavailable | failed |
| 123 | SAFE-PROTECTED-R32 | 8192 | 12.968006 | 7.266784 | complete |
| 123 | SAFE-PROTECTED-R32-LOWLR | 8192 | 12.968006 | 19.434973 | complete |
| 456 | NAIVE-R8 | 5330 | unavailable | unavailable | failed |
| 456 | SAFE-R8 | 8192 | 12.276975 | 11.756609 | complete |
| 456 | SAFE-R8-LOWLR | 8192 | 12.276975 | 19.034607 | complete |
| 456 | NAIVE-PROTECTED-R32 | 4751 | unavailable | unavailable | failed |
| 456 | SAFE-PROTECTED-R32 | 8192 | 12.276975 | 7.309761 | complete |
| 456 | SAFE-PROTECTED-R32-LOWLR | 8192 | 12.276975 | 20.044104 | complete |

## Directly observed diagnosis

All four NAIVE branches fail solely on the raw FP32 norm criterion at attempted steps
4341/3207 (seed123 rank8/protected32), 5331/4752 (seed456). Forward/loss/gradient
elements/current coordinates/current Adam and SAFE proposals are finite on those exact
gradients/states. log10 norms20.117682/19.795311/19.433687/19.796949 confirm large
finite gradients whose sum-of-squares norm overflows. NAIVE proposal coordinates/moments
are also recorded finite; the stop is the explicit norm guard, not NaN parameters.
This establishes the calculation failure, not recovery after switching a failed state:
SAFE trajectories diverged before the event and have zero recorded norm-only events.

All eight SAFE branches finish8192; SAFE-R8 improves own test start at both seeds,
so all three registered gate requirements hold. Its seed456 gain is only0.520366 NLL
and its final validation curve degrades from9.080587 at7168 to11.398067 at8192.

## Prespecified secondary result and selected next direction

SAFE-PROTECTED-R32 test NLL12.968006->7.266784 and12.276975->7.309761.
Total-NLL reductions43.96%/40.46%; removed excess above same-test Qwen57.08%/53.43%.
Original Qwen locked-test NLL2.980255: substantial recovery gap remains.
Protected32 is near7.2--7.6 validation after3072 at both seeds; seed456 has an earlier
2048-step excursion to22.402059. Final raw norms about59/4000, versus6.17e5/8.51e10
for SAFE-R8. These are last-step norms, not maxima or proof of universal stability.
Protection and rank are confounded versus rank8; no SAFE ordinary32 comparator exists
in EXP097. All four tenfold-low-LR SAFE arms worsen final test NLL to18.13--20.04.

## Decision and limits

Proceed with protected32/stable clipping/CE at3e-4 as a candidate, not a final method.
EXP098 isolates protection at rank32/longer horizon; EXP099 isolates protected rank
capacity; EXP100 tests CE-first weak anchoring. Protocols frozen before new outcomes.
EXP097 reuses EXP094--096 evaluation; only two existing source seeds. No new initializer
superiority, random-compute win, recovered capabilities, long-context, MLA or speed claim.
Do not retroactively assign EXP094--096 stops the same cause without their diagnostics.
83,163 completed steps imply0.317268 campaign seconds/step including overhead, not pure throughput.
Raw auditable JSON: `results/EXP-097-numerical-stability.json`.

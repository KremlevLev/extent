# EXP-076 — protected objective bridge (completed)

- Status: completed in 3.609939 hours at implementation revision `21da854e2c901d12bec6aea702a7669b7f577246`.
- Artifacts supplied by the user: full JSON SHA-256 `a072f4db2149363340381a74f6e5c86b1cb4a0861bcd0eb86abfdadd871fbaeb`; summary SHA-256 `730f2a27fb0ff7f2c0a1e95e5f3a09180e3196bb0fcd93260188848f09f7a643`.
- Registered gate: **FAIL**. No trained checkpoint in either arm/seed beats its own step-zero NLL.

| Seed | Arm | Step-0 NLL | Final NLL | Final delta | Max/start |
|---:|---|---:|---:|---:|---:|
| 123 | DELTA-BRIDGE | 10.916678 | 14.093563 | +3.176885 | 1.386338x |
| 123 | STATE-BRIDGE | 10.916678 | 16.507872 | +5.591194 | 1.557009x |
| 456 | DELTA-BRIDGE | 9.692907 | 14.952869 | +5.259962 | 1.634287x |
| 456 | STATE-BRIDGE | 9.692907 | 18.012545 | +8.319638 | 1.883769x |

## Interpretation

The contribution target is directionally better than accumulated-state matching at both seeds: final NLL is lower by `2.414309/3.059676`, and final degradation is reduced by `43.18%/36.78%`. This supports the narrower claim that subtracting the residual identity produces the better hidden-space target.

It does not stabilize joint recovery. Both hidden objectives are worse than EXP-075's historical output-KL/CE Mamba-only arm, and the sampled hidden losses vary from order one to above 200,000. Pre-clipping gradient norms reach approximately `2.16e19`; all recorded gradients remain finite. Thus hidden-only training is rejected as the next recovery stage, while delta matching remains the preferred hidden objective if a later staged method needs one.

The clean next factor is optimizer dynamics: keep the EXP-075 source, mask, prediction objective and data fixed, replacing Lion with AdamW. This is an optimizer diagnosis, not evidence against exact-lift initialization or sequential on-policy recovery.

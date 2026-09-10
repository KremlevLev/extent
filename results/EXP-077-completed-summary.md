# EXP-077 — protected AdamW recovery (completed)

- Status: completed in 3.841757 hours at implementation revision `fbef933f29f8082492647a205aafb11e6de8edc4`.
- Artifacts supplied by the user: full JSON SHA-256 `0af26b6e6d1a19d6d617827751a28c9e6d2405f307d262435427d846729ef336`; summary SHA-256 `06ef5fe585d760a163941f2f39a30db31e71d6f5be356fc6fa6deebb9db91d65`.
- Registered gate: **FAIL**. No trained checkpoint in either arm/seed beats its own step-zero NLL.

| Seed | Arm | Step-0 NLL | Final NLL | Final delta | Max/start |
|---:|---|---:|---:|---:|---:|
| 123 | ADAMW-3E-6 | 10.916678 | 11.205888 | +0.289210 | 1.203924x |
| 123 | ADAMW-1E-5 | 10.916678 | 15.363796 | +4.447118 | 1.515681x |
| 456 | ADAMW-3E-6 | 9.692907 | 9.981154 | +0.288247 | 1.049923x |
| 456 | ADAMW-1E-5 | 9.692907 | 10.666269 | +0.973362 | 1.164883x |

## Interpretation

AdamW at the matched `3e-6` peak is substantially more stable than EXP-075 MAMBA-ONLY Lion: final degradation falls from `+0.777928/+1.611303` to `+0.289210/+0.288247`, reductions of `62.82%/82.11%`. Maximum excursion also falls from `1.543568x/1.483537x` to `1.203924x/1.049923x`. Optimizer family therefore matters materially.

The registered success condition is nevertheless not met: the untouched sequential endpoint remains the best checkpoint for both seeds. Raising AdamW to `1e-5` is clearly harmful, especially for seed 123. Training gradients remain finite but highly uneven and reach approximately `1.51e15` in the sampled records.

Ordinary global optimizer/LR replacement has reached diminishing returns. The next test should bound each Mamba tensor's update relative to that tensor's own parameter norm, keeping the protected source and prediction objective fixed. Random-init comparison remains premature until recovery improves rather than merely preserves the start.

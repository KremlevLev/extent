# EXP-074 — completed stabilized joint recovery

Reviewed 2026-09-09. Status: completed; the registered `LR3E-6` stability gate fails, and the exploratory `LR1E-6` arm also fails.

The supplied full artifact has SHA-256 `d6b427776dc966f2211e69d544e5d5ca6ee01017400b9ff0e13601f94040a600`; its compact summary has SHA-256 `e6ac4689b496c77c794d5e23a006c7a15674cb41872f5dce19dd734637bc92ef`. Runtime was `3.929462` TPU hours.

| Seed | Arm | Initial NLL | Final NLL | Final change | Maximum/start |
|---:|---|---:|---:|---:|---:|
| 123 | LR3E-6 | 10.916678 | 15.672053 | +4.755375 | 1.5974× |
| 456 | LR3E-6 | 9.692907 | 15.944340 | +6.251433 | 1.6783× |
| 123 | LR1E-6 | 10.916678 | 12.653587 | +1.736910 | 1.2901× |
| 456 | LR1E-6 | 9.692907 | 12.504670 | +2.811763 | 1.3196× |

Neither arm ever beats its own step-zero NLL at a registered checkpoint. The lower `1e-6` arm is less destructive, but still raises final NLL by `15.91%/29.01%` and exceeds the pre-registered `1.25×` excursion bound at both seeds. The result rejects the hypothesis that global LR/warmup/clipping reduction alone stabilizes all-parameter Lion recovery.

Together with EXP-073, the evidence points to parameter ownership rather than only global step size: updating all copied Qwen parameters with Lion damages an already useful ONPOLICY start even at a ten-times smaller peak LR. The next causal test should freeze copied embeddings, attention, MLP and normalization weights and update all Mamba subtrees jointly. A secondary arm may unfreeze normalization scales to test minimal residual-stream adaptation. Random comparison remains premature until the shared recovery recipe can improve its own starting checkpoint.

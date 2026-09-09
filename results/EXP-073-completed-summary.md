# EXP-073 — completed sequential warm-start joint recovery

Reviewed 2026-09-09. Status: completed; the pre-registered relative warm-start gate passes, while the joint recovery recipe is unstable.

The supplied full artifact has SHA-256 `b45d7d684a836c6ca3b8002273b2c97fe60a74af512e17fc37a582fac4643f44`; its compact summary has SHA-256 `df58662ff031bd1085c587b7756f1739f1d5b128bc80bc1f9f2a9003d002e0c3`. The run completed in `3.871453` TPU hours.

## Registered comparison

| Seed | TEACHER start/final NLL | ONPOLICY start/final NLL | Final ONPOLICY−TEACHER | Delta AUC |
|---:|---:|---:|---:|---:|
| 123 | 21.987584 → 32.003811 | 10.916678 → 9.278139 | -22.725672 | -16.711477 |
| 456 | 21.742845 → 37.150260 | 9.692907 → 23.089646 | -14.060614 | -10.608272 |

Both registered conditions pass at both seeds. Mean final NLL difference is `-18.393143`. At the final checkpoint, ONPOLICY reduces NLL relative to the equal-update TEACHER warm start by `71.01%` for seed 123 and `37.85%` for seed 456; excess NLL above Qwen is lower by `77.86%` and `40.95%`.

The warm-start advantage therefore survives joint optimization and is not immediately erased. This is meaningful evidence for sequential deployment-distribution recovery as an initialization/recovery stage. The absolute training trajectories, however, are highly unstable. Seed-123 ONPOLICY briefly spikes to NLL `40.445658` at step 1024, then reaches its best registered point `8.804904` at step 4096 and ends at `9.278139`, better than its `10.916678` start. Seed-456 ONPOLICY reaches `8.949895` at step 2048, then spikes to `39.923966` at step 4096 and ends at `23.089646`, much worse than its `9.692907` start. Both TEACHER arms finish worse than step zero.

Consequently, the passed gate is a relative method result, not successful model recovery. Much of the final margin comes from TEACHER degrading more severely. Still, both ONPOLICY seeds contain a checkpoint better than their own step-zero model, showing that useful joint recovery is possible if optimization is stabilized and selected without evaluation leakage.

## Decision

Accept the claim that sequential on-policy recovery produces a full-model warm start whose advantage persists under equal joint training. Reject the current constant `3e-5` all-parameter Lion schedule as a production recovery recipe. The next experiment should pre-register a stability intervention using training-only signals or a fixed schedule—such as a lower learning rate, gradual unfreezing, and/or protected Mamba learning-rate groups—then compare against the unchanged EXP-073 recipe. Random initialization remains required in the later compute-accounted comparison; adding it before stabilizing the shared joint optimizer would mostly compare failure modes.

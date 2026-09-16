# EXP-085 completed result

Reviewed 2026-09-16. All four data replications complete for both source seeds.
Last invocation: 5.239 TPU hours; no outer durability warnings.

| Seed | Pair mean final-start NLL | Normal 95% upper | Improving runs | Pair minus greedy | Gate |
|---:|---:|---:|---:|---:|---|
| 123 | -0.587421 | -0.070664 | 4/4 | -0.298767 | pass |
| 456 | -0.532418 | -0.217849 | 4/4 | +0.144568 | fail |

Pair lookahead improves its unchanged start in all eight trajectories. It beats
greedy in all four seed-123 replications but only two of four seed-456 replications.
Joint-only rescues appear in 3/4 and 4/4 replications respectively. This supports
real multi-coordinate interactions and consistent improvement over the unchanged
warm start on these registered ranges, but not cross-seed superiority over matched
greedy selection. The overall pre-registered gate fails. Normal bounds use only
four data replications and are descriptive approximations, not a broad guarantee.

The seed-averaged pair-minus-greedy point estimate is -0.077100 NLL, but pooling
seeds cannot override the registered per-seed gate. No claim of a 20% model-quality
gain, recovered Qwen capabilities, or improved initialization follows here.

Numeric-key checkpoint dictionaries caused an ancillary accepted-layer-frequency
bug. After decoding dictionary values, primary accepted counts are 12/7/5/3 for
seed 123 and 5/7/7/4 for seed 456. The previously reported seed-456 replication-0
count of ten was inflated; NLL, rescue flags and the overall gate are unchanged.

Decision: stop EXP-085 and scalar-threshold variants. Pair selection is a useful
mechanistic baseline, not the final recovery method. A subsequent method should
change proposal training to account for composition, rather than only select
among independently trained local updates.

Full artifact SHA-256:
`942082eca0bc93d051bfda74e5d7d3848cc010d0273ced5a76f4ed5298c2178b`

Summary SHA-256:
`033935cb9fab07454682c03750d091a88ab01cfc218f6ec1d04bb43d7eb6bbff`

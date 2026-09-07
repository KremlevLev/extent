# EXP-070 — completed frozen input-shift diagnosis

Reviewed 2026-09-07. User full artifact SHA256 `2a3c14032d9513eed8e971ec854ac7d017bd52b5b3ef2ff80db089112981224d`; run revision `ef700b15a356b379a2eebf169821fc896c2c30a0`. Status completed: 40/40 accepted probes in 0.519455 hours. No training or optimizer update was performed. The supplied artifact, rather than a live HF listing, is the evidence source.

## Controls

All eight all-GQA controls (the same four text/length cells repeated across two model seeds) passed. With highest matmul precision, maximum per-layer FP32 hidden relative L2 was `4.28e-7`, logits relative L2 was at most `7.66e-6`, absolute NLL difference at most `5.72e-6`, and top-1 agreement was 1.0. This localizes the prior v2 discrepancy to numerical lowering rather than parameter-tree mismatch at the resolution measured here. It is not a general TPU determinism claim.

The real BF16 paths retained a numerical-background prediction KL of approximately `0.00308–0.00337`; BF16 NLL differences ranged from `-0.00159` to `+0.01339`. Hybrid effects below this background would be ambiguous. Observed replacement effects are much larger.

## Frozen whole-model behavior

Mean NLL and mean base-10 logarithm of the raw pre-clipping gradient norm across two seeds and two windows:

| Replacements | Context | Mean NLL | Mean log10 gradient norm |
|---:|---:|---:|---:|
| 0 | 64 | 3.08 | 0.80 |
| 0 | 256 | 2.68 | 0.46 |
| 1 | 64 | 7.57 | 2.67 |
| 1 | 256 | 5.94 | 2.48 |
| 4 | 64 | 7.30 | 2.80 |
| 4 | 256 | 6.31 | 2.31 |
| 12 | 64 | 12.60 | 3.25 |
| 12 | 256 | 13.55 | 3.11 |
| 24 | 64 | 19.58 | 14.34 |
| 24 | 256 | 20.23 | 14.82 |

One replacement already causes a large capability shock. Four replacements do not monotonically worsen NLL relative to one, so this nested path does not support a simple per-layer additive law. At 24 replacements the raw gradient norm is approximately `6.21e13–1.21e15`, explaining EXP-069's extreme gradients before clipping.

## Matched-input test of H3.4

For each replacement, the same Mamba and Qwen decoder blocks were evaluated on the same Qwen-generated input and then on the same hybrid-prefix-generated input. Residual identity was removed from the error target. Aggregates below pool layer/cell measurements and therefore are descriptive, not independent-replicate confidence estimates.

| Nested replacement count | Measurements | Median hybrid/teacher-input error-RMS ratio | Mean ratio | Fraction ratio > 1 | Mean input relative drift |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1.00 | 1.00 | 0.75 | 0.00 |
| 4 | 32 | 1.05 | 1.13 | 0.72 | 0.63 |
| 12 | 96 | 2.16 | 5.00 | 0.91 | 0.90 |
| 24 | 192 | 17.55 | 85.73 | 0.95 | 52.13 |

The equality of repeated per-layer values across nested counts is expected: the prefix before a given layer is unchanged when only later replacements are added. It is a useful internal consistency check, not extra replication.

Along the complete 24-replacement prefix, mean hybrid input RMS grows from `0.03` at layer 0 to `25.18` at layer 15, `297.52` at layer 18, `2,800.58` at layer 22 and `27,343.58` at layer 26. Mean relative input drift reaches `424.40` at layer 26. Mean hybrid-input decoder error RMS reaches `36,847.25`, versus `236.08` on teacher inputs. This is decisive evidence that independently prepared blocks encounter a severe out-of-distribution residual stream when composed.

However, the teacher-input error also degrades with depth: mean teacher-input error RMS rises from below one in early layers to `3.21` at layer 17, `15.02` at layer 22, `81.51` at layer 24, and `236.08` at layer 26. Thus sequential/on-policy calibration is strongly motivated but cannot yet be claimed sufficient. Fresh-text generalization of later independent endpoints is itself poor.

Local-versus-full student execution relative differences grow to about `0.06` in late layers, far below the hundreds-fold hybrid errors but not literally zero. The interpretation relies on magnitudes well beyond that execution background.

## Gradient attribution

Across packed Mamba input projections, value and gate together account for roughly `94–98%` of their squared gradient norm, depending on replacement count. At 24 replacements their mean internal fractions are approximately `0.47` each; dt is `0.03`, B and C about `0.01` each, decay and angle about `0.01`, and trapezoid below `0.01`. Globally, embedding, early MLP/down projections, and early Mamba input projections frequently dominate. This rejects the narrow hypothesis that dt/decay/angle gradients are the primary source of the observed explosion in this frozen diagnostic. It does not rule out dynamics as the upstream cause of exploding activations.

## Decision

H3.4-style input-distribution sensitivity is supported as a mechanism. The next method experiment should compare matched extra recovery from identical prepared endpoints: teacher-input recovery, pure sequential hybrid-input recovery, and a mixed teacher/hybrid-input recovery that protects original-distribution behavior. Use a fixed nested prefix and identical per-layer updates/data. Evaluate held-out whole-model NLL after intermediate depths and across seeds. This will test whether addressing the measured shift improves recovery; EXP-070 alone does not establish that intervention.

Do not spend TPU quota repeating EXP-070 unchanged. It did not test MLA, long context, generation speed, or a trained sequential method.

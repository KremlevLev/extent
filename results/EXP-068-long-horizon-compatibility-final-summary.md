# EXP-068 — completed long-horizon compatibility confirmation

- Completed 28/28 layers; numerical, early-predictor, and independent confirmation gates pass.
- Final invocation: 1.302316 hours; restored 23 results and uploaded the final five layer results according to the supplied artifact. Total across three reported invocations: approximately 9.706 TPU hours.
- EXP-067/068 final difficulty Spearman: 0.9901477833.
- EXP-068 step-1024 versus step-8192 difficulty Spearman: 0.9994526546.
- EXP-067 selection: [0, 1, 20, 26]. EXP-068 selection: [0, 1, 26, 27]. Overlap: 3/4, satisfying the pre-registered threshold.
- Layers 27 and 20 are close: final dual decoder relative L2 0.08519217 and 0.08446497. Their swap does not establish a statistically significant difference between these positions.
- Dual preparation wins final decoder L2 over random on 14/28 layers (0–2 and 17–27) and loses on 14/28 (3–16). It remains a depth-dependent intervention.

## Interpretation and next comparison

The cheap probe reproduces the long-horizon ordering under the tested settings. Treat [0, 1, 26, 27] as the long-horizon candidate and preserve [0, 1, 20, 26] as the independently checked cheap-probe candidate. Whole-model NLL on fresh evaluation data must establish whether either allocation improves over a fixed evenly spaced four-attention-layer control.

High rank correlation alone does not prove model-level layer importance: decoder relative L2 includes the residual stream and can vary with layer scale. Nor does correlation near one mean 99% model quality or accuracy. The second experiment changes seed, text, recovery horizon, and preparation budget together; it confirms robustness across these settings without isolating their individual effects. Dual preparation has extra compute relative to the random control, which must be accounted for in efficiency claims.

## Provenance

- Source: Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1.
- Full supplied artifact SHA-256: 9a3593290a453f1c3e7aab132ddd85497bdf1af001b191c0a500c34500b85e20.
- Supplied summary SHA-256: e612689c5cfef4c3c0f5450f38415815435b78478e1f1e2a2fe5fe5818943514.
- Earlier partial runs remain recorded in EXP-068-long-horizon-compatibility-partial-summary.md.

# EXP175 received · 13 November 2026

The aggregate reports completed. All three224-update pilots completed on
the same fresh Edu prefix. Physical batch8, effective16, context1024.
The selected CE arm continued to511 updates and reports verified HF storage.
All received source-code hashes match the local files after LF normalization.
[Received summary, raw SHA and per-window values](EXP-175-received-summary.json).

## Matched comparison

Each pilot presents3,670,016 input positions. All active student weights
are trainable at LR1e-6 after32 warmup updates. Source is clean171 CURRENT5120.
Mean validation CE NLL uses128 available Edu1024 windows.

| Objective | Initial NLL | NLL at224 | Initial KL | KL at224 |
| --- | --- | --- | --- | --- |
| CE | 2.544436 | 2.540840 | 0.133931 | 0.131580 |
| KL | 2.544436 | 2.541114 | 0.133931 | 0.130709 |
| 0.5CE + 0.5KL | 2.544436 | 2.541728 | 0.133931 | 0.132934 |

All objectives produce a small held-out NLL improvement. CE wins the
registered endpoint, but its advantage over KL is only0.000274 NLL.
CE improves105/128 windows against source and beats KL in68/128 windows.
A descriptive paired-window resampling interval for CE minus KL crosses0.
The shared validation panel, adjacent packed windows and single source
limit statistical and generalization claims.

CE continues to8,372,224 winning-trajectory inputs including its pilot.
Final128-window NLL2.539805, a0.004630 improvement over source and0.001035
over its224-step endpoint. Final KL0.131716. Four-window monitor NLL at
256/320/384/448 is2.577279/2.577995/2.578597/2.577071. That small panel is
nearly flat, while the expanded endpoint still improves slightly.
There is no post175 task or generation benchmark.

## What the diagnostic establishes

The disposable mixed-objective worker makes32 updates on one fixed batch.
CE falls2.536957 to2.524765, KL0.102025 to0.100193, and the mixed objective
1.319491 to1.312479. The end-to-end model can improve its loss on that batch.
The first step alone slightly raises CE and the mixed objective. Single-step
behavior therefore does not provide a blanket monotonic-descent guarantee.
Forward-only and gradient-path metrics at unchanged parameters also differ
slightly, consistent with numerical-boundary sensitivity.

At the initial measured batch, global CE/KL gradient cosine is+0.535085.
Raw CE gradient norm is about2.27 times the KL norm. Most recorded groups
are positively aligned. Final-norm cosine is-0.073001, so local disagreement
exists despite positive global alignment. These measurements do not support
global CE/KL opposition as the explanation for the175 learning curve.
The actual first Adam displacement has cosine-0.269509 with the mixed
gradient. This supports the intended local update direction in that check.
It does not validate every subsequent derivative or mixed-precision step.

## Diagnostic limitation

The reported BF16 changed fraction equals the FP32 fraction bit-for-bit in
every group, including97.949% for final norm after a1e-6 update. This
counter is unvalidated on the actual TPU and cannot establish how many
changes survive BF16 rounding.

A local CPU sanity check on FP32 ones versus ones+1e-6 produces FP32
fraction1 and BF16 fraction0 under both eager execution and JIT. That
check validates the intended distinction and does not reproduce the TPU
report. Independent host bit-pattern samples or optimized TPU HLO are
needed to identify the discrepancy.

Compiler handling of reduced precision is one possible cause. OpenXLA has
precision-conversion optimizations controlled by excess-precision settings,
but their involvement in this run is unconfirmed.
[OpenXLA conversion pass](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/float_normalization.cc).

## Cost and next interpretation

All pilots plus continuation consume15,712,256 accepted training inputs.
Disposable same-batch updates add524,288 presentations, and the two
component-gradient evaluations add32,768. These counts exclude validation,
additional forward-only checks and teacher forwards. Source171 plus only
the winning CE trajectory totals92,258,304 input presentations.

The CE continuation reports about2982 input tok/s across its training loop.
The run establishes small improvements with both supervised text prediction
and distillation at the tested rate. A longer CE continuation is a candidate,
while superiority over KL and improvement of downstream abilities remain
open. Neither a severe general gradient failure nor BF16 as the cause of
slow learning has been established.

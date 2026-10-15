# EXP125--127 engineering validation, 16 October2026

Prepared, NOT physical1.7B GPU results. Local CPU JAX environment.

- 10 dynamic-recovery tests passed (66.77s) after final math/runner changes.
- Additional BF16 identity, full bound/profile and finite-gradient test passed
  (11.97s). Total11 cases. BF16 tolerance source token-NLL .01, instrumented
  relative-output .02, ideal bound relative rounding allowance2%.
- All six recipes: streamed full-chain LoRA/MLP gradients agree with monolithic
  autodiff on a tiny FP32 model. Test ceilings/ratios adjusted to exercise caps
  at the tiny fixture's scale; not evidence of1.7B training behavior.
- Token-wise causal factors preserve small branches, bound huge branches,
  geometric ramp is gradual at its endpoint. True derivative checked against
  finite differences; zero-input derivatives finite.
- Zero-effect MLP initialization, actual MLP gradients, nonfinite proposal
  rejection, exact Adam next-step checkpoint restore, clean timeout and resume
  from2 to4 accepted updates without reset. Pre-ramp best stays ineligible.
- Plan-only generated all six contracts without loading a patient/GPU.
- Earlier120--124 scientific modules/contracts unchanged.

Physical GPU memory, throughput and quality remain pending. New source controls
on A/B/C explicitly address missing exact source C in the previous122 report.
Raw120--123 reports archived alongside an analysis with per-download SHA and
same4096-step comparisons; exposed panels do not become independent tests.

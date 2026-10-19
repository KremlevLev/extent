# EXP153 v2 local validation,19 October2026

6 tests pass in71.59s: tests/test_rank_growth_153.py.

- Real tiny BF16 hybrid with rank32 SOURCE B factors materially NONZERO and
  learned pregate weights; checkpoint saved and restored through NumPy reader.
  Target64/256 keep all original source leaves exactly and add zero-B factors.
  Source/grown NLL checked on three windows with maximum token-NLL tolerance.003,
  stricter than the cloud source-window gate.03.
- Actual streamed growth forward AND VJP produce finite updates; new extra B
  changes after the first accepted step. Protected input columns receive zero
  gradients. Gradient parity uses the existing full-tree relativeL2<=.15 and
  cosine>=.98 plus CEerror<=.03, rather than elementwise relative ratios near0
  across different BF16 fusion boundaries. Optimizer output verified against
  applying the actual streamed gradient; all6 counters increment0->1.
- Actual new-state save/restore roundtrip retains arrays and optimizer state.
- Actual runner tested through initialization, source replay, first update,
  timeout save. Deliberate source mismatch persists measured/reference/worst
  windows and source_controls BEFORE rejection, zero accepted updates.
- CLI dispatches correct new subprocess module; only153 accepted. Original
 149-154 contracts remain identical to frozen engineering plan.
- REAL147 payload restored again:487 SHA-verified chunks,114,979,740B.
  Growing actual source coordinates preserves all7,182,336 original elements
  bitwise, extra B zero/extra A finite. Target64:14,327,808 total trainable
  elements; target256:57,200,640. Frozen base SHAPES only used in this audit;
  no full1.7B forward was performed. See source-auditJSON.

An exploratory check with actual147 layer0 out-projection factors and a
SYNTHETIC BF16 frozen kernel found5/3,145,728 differing stored coefficients
between NumPy host folding and JAX compiled CPU evaluation. This is NOT proof
of the full-model153v1 replay failure cause and is not reported as its repair.

Old StreamedStep ignores its block_fn argument. New isolated subclass overrides
both forward and backward kernels explicitly; regression tests require actual
extra-parameter updates, so silently unused added coordinates cannot pass.
No old numerical/source/runner files changed. Cloud source replay at.03 remains
mandatory and unreduced; physical T4 and full-model behavior not verified here.

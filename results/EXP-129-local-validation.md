# EXP-129 local engineering validation

16 October 2026, local CPU. Status PREPARED; no real1.7B GPU/TPU run.

`pytest tests/test_pregate_recovery.py -q`: **7 passed in43.32s**.

Checked:

- Explicit RMS formula, last-axis-only groups, nonuniform shared head weights,
  epsilon, zero input gradient and finite behavior near1e30.
- Strength0 scan exactly equals legacy scan. Independent quadratic dual agrees
  with recurrent scan at partial and full strength, including input/weight
  gradients. Profile and ordinary scan outputs exactly equal on FP32 fixture.
- Both arms' streamed full-model loss/gradients versus monolithic autodiff;
  learned norm has a live gradient. Exact next Adam proposal after serialization.
- Both arms BF16 source NLL parity, ordinary/profile acceptance and finite
  full-chain gradient at full strength.
- Runner clean-timeout checkpoint and repeat invocation advance durable
  steps2→4 without overwriting continuation state.
- plan-only generates both source/code-pinned contracts without loading weights.
- All EIGHT frozen125–128 worker contracts compare exactly equal to committed
  engineering plans. Default/reference Mamba code unchanged.
- git diff --check.

Not verified: physical T4 memory/throughput, actual1.7B source replay,
long-run numerical health, HF transfer under real concurrent traffic, NLL
improvement, long-context recovery or reproduction of competitor model quality.

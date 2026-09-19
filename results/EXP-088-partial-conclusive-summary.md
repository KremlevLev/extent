# EXP-088 — downstream-sensitive pair recovery: partial but conclusive result

Reviewed 2026-09-20 from the user-supplied full and compact campaign artifacts.

## Outcome

The campaign completed three of four pre-registered data replications in 4.850 TPU hours. The registered scientific gate is already unreachable, so replication 3 must not be run merely to fill the table.

`DOWNSTREAM-KL` trains each current Mamba pair through the frozen decoder suffix against offline full-Qwen logits. `LOCAL-PAIR-MSE` is the composed local-segment control. Both arms evaluate both calibration domains; only proposal training differs.

| Seed | DOWNSTREAM-KL mean ΔNLL | 95% upper | Improving completed runs | Mean final NLL vs local control | Verdict |
|---:|---:|---:|---:|---:|---|
| 123 | -1.578313 | +1.477410 | 2/3 | +0.262729 | fails uncertainty and comparator gates |
| 456 | +0.730171 | +1.042370 | 0/3 | +1.432225 | conclusively fails |

For seed 456, the three `DOWNSTREAM-KL` deltas are `+0.450965`, `+1.002621`, and `+0.736928`. The registered rule requires at least three improving runs out of four. With zero improvements after three completed runs, the best possible final count is one of four; completion cannot change the verdict.

The seed-123 mean is dominated by one large win (`-4.522972`). The other two repetitions are `+0.782033` and `-0.994000`; relative to the local control, downstream training loses in two of three repetitions and has a positive mean difference. This is not stable evidence that the language-model-aware objective is better.

Across the two seed-level comparator means, `DOWNSTREAM-KL` is `+0.847477 NLL` worse than `LOCAL-PAIR-MSE` on average. This descriptive average is not a substitute for the pre-registered per-seed gate, which also fails.

## Interpretation

The failure is scientifically informative but is not a positive recovery result. Passing gradients through the frozen suffix and matching full-Qwen logits does not make pairwise recovery reliably composable at this budget. It can produce a large isolated win, but the direction changes with seed/data replication and is especially consistently harmful for seed 456.

This closes the current chain of increasingly elaborate local/pair selection methods: stricter thresholds, joint pair proposals, and downstream-sensitive pair proposals have not produced stable cross-seed superiority. Do not add another selector, threshold, or local objective variant without a genuinely new mechanism.

The original project objective remains unmet: there is still no demonstrated Transformer-to-Mamba-3 initialization/recovery recipe that beats random initialization after full-model assembly. Engineering readiness, local exact-lift advantages, and diagnosis of input-shift/composition failure are real outputs, but they are not evidence that the intended 14B transplant will work.

## Decision

- Stop EXP-088 at 3/4 for mathematical futility.
- Do not spend another TPU allocation completing replication 3.
- Freeze pairwise/local recovery method search.
- Before any 14B recovery run, choose between (a) presenting the accumulated result as a negative/diagnostic study of why locally successful Transformer-to-Mamba transplants fail to compose, or (b) starting a new full-model method line whose primary comparison is exact-lift versus random under the same meaningful token budget.

## Provenance

- Runtime: TPU v5e-8, 4.850 hours for this invocation.
- Source revision: `af0a6ed827dd40fd84ebaf4bda8b8b789bca651a`.
- Full artifact SHA-256: `ecb607893084721c44f36fe65472eb32175642e5e3cf612a0c345eec28a70140`.
- Compact summary SHA-256: `f566fbabbb93982cdb8bfd069501aca872e99cdea72151bf989f41a98652dad5`.


# EXP-083 — dual-domain assembled trust recovery (pre-registered)

## Question

Can an update rule that requires assembled-model improvement on both a source-like and a shifted corpus remove EXP-082's seed-dependent cross-domain failures?

## Frozen starting point and proposal

- Same Qwen3-1.7B EXP-072 ONPOLICY endpoints, source seeds 123/456, 24 Mamba replacements and four retained GQA layers.
- Forward coordinate order; 2,048-step conditional contribution proposal per layer; BF16 Lion, LR `3e-5`, warmup 64, cosine, no decay, clip 1.0.
- Same alpha grid `0/0.125/0.25/0.5/0.75/1.0` and 0.1% relative KL threshold.

## Matched arms

- `WIKI-ONLY`: select the alpha using assembled-model prediction KL on the WikiText-103 calibration slice. PG-19 calibration is computed but hidden from selection.
- `DUAL-CONSENSUS` (registered primary): a nonzero alpha is eligible only if it improves assembled-model KL by at least 0.1% on both WikiText-103 and PG-19 validation. Among eligible alphas, select the one with the lowest worst-domain KL ratio. Alpha zero remains the fallback.

Both arms pay for both calibration forwards, so their compute differs only in the selection rule.

## Eight data replications

- Proposal and primary calibration: disjoint ranges from pinned `wikitext-103-raw-v1/train`, stride 69,632 tokens.
- Secondary calibration: eight disjoint 4,096-token ranges from pinned PG-19 validation books.
- Locked final evaluation: eight disjoint 8,192-token ranges from PG-19 test books, never used in method development or alpha selection.
- Every replication contains both source seeds and both arms. Per-window NLL is retained, but the primary uncertainty unit is the data replication.

## Gate

For each source seed: all eight replications must complete; DUAL-CONSENSUS mean final-minus-start NLL and its upper normal 95% bound must be negative; at least six of eight replications must improve; mean DUAL-CONSENSUS final NLL must beat WIKI-ONLY; every milestone must remain within `1.25x`; and at least six replications must accept a nonzero coordinate. Both seeds must pass every condition.

A pass identifies cross-domain agreement as the missing acceptance signal. Stable rejection without improvement means the proposal itself is inadequate. Failure despite dual calibration rejects trust filtering as a general solution and redirects work toward proposal construction or joint training.

## Runtime and resilience

The campaign contains 1,572,864 local proposal updates plus two-domain alpha evaluation and is expected to use 6–8 TPU v5e-8 hours. Each replication has milestone HF checkpoints; the outer campaign resumes completed replications. PG-19 asset downloads and HF checkpoint commits have bounded retries. Inner notifications are disabled; the outer campaign sends one start and one terminal status.

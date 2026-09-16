# EXP-087 — composed joint-pair recovery

Pre-registered 2026-09-16, before any EXP-087 outcomes.

## Hypothesis

EXP-085 demonstrates useful interactions but not cross-seed superiority of joint
selection. Its proposals are still trained independently. EXP-087 tests whether
differentiating through the actual composed pair makes the proposals themselves
more useful to the assembled network.

## Matched comparison

- `INDEPENDENT-PAIR`: EXP-085 independent conditional decoder targets for each
  Mamba subtree, followed by pair-consensus selection.
- `JOINT-PAIR` (primary): optimize both Mamba subtrees through the complete decoder
  segment from the first replaced layer through the second. Any retained attention
  layer between them remains frozen but participates in forward/backward.
- The joint target is the frozen Qwen segment evaluated on the same hybrid-prefix
  input. Relative MSE subtracts the shared residual identity. Qwen MLPs, norms,
  retained attention, and all modules outside the two Mamba subtrees are frozen.
- Both methods use 2,048 updates per Mamba subtree, BF16 Lion with the existing
  recipe, identical alpha grid and 0.1% two-domain pair-consensus threshold.
  Joint steps update both subtrees simultaneously; independent steps update one.
  This is not equal FLOPs. Proposal wall time is recorded per pair, including
  cache/target preparation, so compute-normalized comparisons remain possible.

## Data and scale

Pinned Qwen3-1.7B-Base; unchanged EXP-072 ONPOLICY endpoints; seeds 123/456;
four fresh data replications; all 24 Mamba coordinates in 12 causal pairs.
WikiText-103 proposal/calibration starts at token 1,671,168; PG-19 validation
at 131,072; locked PG-19 test at 196,608. These start after the complete planned
EXP-086 ranges. Token hashes are frozen on first execution.

## Gate

At each seed, all four replications must complete; JOINT-PAIR final-minus-start
mean NLL and upper normal 95% bound must be negative; at least 3/4 runs improve;
mean final NLL beats INDEPENDENT-PAIR; registered excursions remain below 1.25x
start; and at least 3/4 runs accept a nonzero coordinate. Test results never choose
alpha, learning rate or endpoint. Mechanism rescue flags are descriptive.

## Operations and boundary

Dedicated `exp087` local/RAM/HF namespaces, registered-milestone checkpoints,
bounded retries, start/final Telegram notification and 8.1-hour soft budget with
finalization reserve. Expected work spans roughly 5-8 TPU hours, but actual
joint-backward cost is unmeasured and a deadline-partial result may need resuming.
No requirement to upload local checkpoints manually.

A pass supports composition-aware local proposal training over independent
proposal training, not stronger initialization versus random, recovered Qwen
capabilities, MLA, 14B scaling, or inference acceleration. Those require separate
controlled comparisons after recovery becomes reliable.

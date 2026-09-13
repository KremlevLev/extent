# EXP-085 — interaction-aware paired-coordinate lookahead (pre-registered)

## Question

EXP-082–084 tested three increasingly strict scalar rules for accepting one recovered Mamba coordinate at a time. None generalized across source seeds and fresh PG-19 ranges. EXP-085 tests a different mechanism: can two locally recovered coordinates be useful together even when greedy one-at-a-time selection misses their interaction?

## Matched arms

- `GREEDY-PAIR-GRID` is the control. Within each consecutive pair of Mamba coordinates, it chooses the first alpha using candidates `(alpha_1, 0)`, then chooses the second conditional on that first choice.
- `PAIR-LOOKAHEAD` is primary. It chooses both alphas jointly from the complete 6×6 Cartesian grid.
- Both arms train two proposals from the unchanged assembled state at the beginning of the pair. They use identical alpha grids, local steps, data, optimizer, two-domain evaluations, and full 6×6 evaluation cost. Adaptive branches can produce different later proposal tensors; the controlled quantity is the complete method and matched compute, not tensor identity after earlier decisions diverge.
- A nonzero pair still needs at least 0.1% assembled prediction-KL improvement on both WikiText-103 and PG-19 validation. Locked PG-19 test NLL never selects an alpha.

## Frozen design

- Source: pinned Qwen3-1.7B-Base and the same complete EXP-072 ONPOLICY endpoints used by EXP-081–084.
- Four fresh data replications and source seeds `123/456`.
- Twelve consecutive pairs covering all 24 Mamba coordinates in causal order.
- 2,048 BF16 Lion proposal updates per coordinate. Total local proposal compute is 786,432 updates, plus matched 6×6 two-domain assembled evaluation in both arms.
- Proposal/primary calibration begins at WikiText-103 train token 1,114,112. Secondary PG-19 validation begins at 98,304. Locked PG-19 test begins at 131,072. Replication strides preserve disjoint ranges and runtime hashes freeze exact tokens.

## Gate

For each source seed, all four replications must complete; mean `PAIR-LOOKAHEAD` final-minus-start NLL and its upper normal 95% bound must be negative; at least 3/4 replications must improve; primary mean final NLL must beat `GREEDY-PAIR-GRID`; all registered trajectories must stay within 1.25× start; and at least 3/4 runs must accept a nonzero coordinate. Mechanism support additionally requires at least one replication per seed containing a joint-only rescue: a selected `(alpha_1, alpha_2)` that improves both domains although neither component at its selected alpha passes alone.

## Runtime and durability

Expected TPU v5e-8 wall time is 6–8 hours, with deadline-partial completion allowed. Each replication has a two-hour inner budget, six-coordinate HF checkpoints, bounded upload retries, aggregate snapshots, and terminal Telegram notification. Rerunning the same cell restores completed replications and the latest milestone of an active one. Historical HF states are retained under the current storage policy.

## Decision boundary

A pass establishes that short-horizon multi-coordinate interaction is a missing part of transplant recovery and motivates wider beam/block methods. A failure rejects adjacent two-coordinate lookahead at this budget; it does not by itself reject different Mamba initialization, longer joint training, non-adjacent grouping, MLA, or the final 14B system.

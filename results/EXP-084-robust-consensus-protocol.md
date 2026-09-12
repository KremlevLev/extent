# EXP-084 — paired-window robust dual consensus (pre-registered)

## Question

Does EXP-083 fail because mean KL lets a small subset of easy calibration windows hide regressions on the rest of the distribution?

## Matched arms

- `MEAN-CONSENSUS`: the exact EXP-083 two-domain rule. An alpha needs at least 0.1% mean assembled-KL improvement on both domains; selection minimizes the worst normalized mean KL.
- `ROBUST-CONSENSUS` (registered primary): applies the same mean requirements and additionally requires strictly lower paired-window prediction KL on at least 60% of windows in each domain. Among eligible alphas it minimizes worst normalized mean KL, then favors the larger worst-domain window-improvement fraction.

Both arms train identical local proposals, evaluate the identical alpha grid on both calibration domains, and therefore use matched compute. Alpha zero remains available.

## Data and repetition

- Eight data replications and source seeds 123/456.
- Fresh WikiText-103 proposal/primary-calibration ranges beginning at token 557,056.
- Fresh PG-19 validation calibration ranges beginning at token 65,536.
- Locked PG-19 test ranges beginning at token 65,536; these test books/tokens have not been used by EXP-083.
- 24 forward coordinates, 2,048 local BF16 Lion proposal steps, alpha grid `0/0.125/0.25/0.5/0.75/1.0`.

## Gate

For each source seed separately: all eight replications; negative ROBUST-CONSENSUS mean final-minus-start NLL with a negative upper normal 95% bound; at least six of eight negative replications; negative mean final NLL versus MEAN-CONSENSUS; no milestone above `1.25x` start; and at least six replications accepting a nonzero update. Both seeds must pass.

If the robust arm stays stable but accepts too little to improve, the local proposal rather than the selector is the limiting factor. If it remains unstable, assembled greedy filtering is rejected as a general recovery strategy and the next method must retain multiple composition paths or change proposal training.

## Runtime and durability

Both arms already paid for two-domain alpha evaluation in EXP-083, so runtime should remain approximately 7–8 TPU v5e-8 hours. Eight resumable replication namespaces, bounded HF/PG-19 retry windows, outer progress snapshots and terminal Telegram notification are retained.

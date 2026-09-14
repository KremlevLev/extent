# EXP-086 — conservative pair-acceptance threshold (pre-registered)

## Question

EXP-085 replication 0 showed that pair lookahead can find real joint-only rescues,
but its utility changed by source seed. On seed 456 it accepted ten coordinates and
lost to a greedy arm that accepted only three. EXP-086 tests whether the pair rule
is over-accepting weak calibration improvements rather than whether interactions
exist.

## Matched arms

- `STANDARD-PAIR-0.1PCT` uses EXP-085's pair-consensus rule: the selected pair must
  improve assembled prediction KL by at least 0.1% on both calibration domains.
- `STRICT-PAIR-0.5PCT` is primary and raises only that threshold to 0.5% on both
  domains.
- Both arms train the same kind of two independent proposals from each unchanged
  pair-start state and evaluate the complete 6x6 pair grid on WikiText-103 and
  PG-19 validation. Alpha grid, proposal steps, optimizer, architecture and locked
  evaluation are identical.

Adaptive earlier choices can change later proposal inputs. The comparison is
between complete recovery policies with matched compute, not between identical
later tensors.

## Frozen design

- Pinned Qwen3-1.7B-Base; EXP-072 ONPOLICY endpoints; seeds 123 and 456.
- All 24 Mamba coordinates in 12 consecutive causal pairs.
- Four independent fresh data replications; 2,048 BF16 Lion proposal updates per
  coordinate.
- Fresh ranges start at WikiText-103 train token 1,392,640, PG-19 validation token
  114,688 and locked PG-19 test token 163,840, immediately after the complete
  planned EXP-085 ranges.
- Separate `exp086` local, RAM and Hugging Face namespaces permit concurrent
  execution with EXP-085 on another account.

## Gate

For each source seed, all four replications must complete. The strict arm needs a
negative mean final-minus-start NLL with negative upper normal 95% bound, at least
3/4 improving replications, lower mean final NLL than the standard-threshold arm,
no trajectory above 1.25x its start, and nonzero acceptance in at least 3/4 runs.

Joint-only rescue is reported as mechanism evidence but is not required: a stricter
threshold is explicitly allowed to reject marginal interacting pairs.

## Interpretation boundary

A pass shows that conservative pair acceptance improves cross-seed composability
relative to EXP-085's threshold. A failure rejects this fixed 0.5% regularization;
it does not establish that thresholds should be tuned on locked results. No EXP-086
outcome may retroactively alter the registered EXP-085 gate.

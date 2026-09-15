# EXP-086 partial but conclusive result

Reviewed: 2026-09-15

## Status

Three of four registered fresh-data replications completed in `5.289553` TPU
hours. The fourth replication is not required to decide the registered gate:
seed 456 can no longer attain a negative upper normal 95% bound for any possible
fourth final-minus-start value. The experiment is therefore stopped for futility
rather than spending approximately another 1.75 TPU hours.

## Results after three replications

| Seed | Strict mean delta | Negative runs | Strict minus standard | Max/start |
|---:|---:|---:|---:|---:|
| 123 | -0.990966 | 2/3 | +0.123984 | 1.037259x |
| 456 | -0.055387 | 2/3 | +0.693761 | 1.142444x |

Positive strict-minus-standard values favour the standard 0.1% threshold. The
strict 0.5% arm ties the control in replication 0, is modestly better for seed 456
in replication 1, and fails badly in replication 2:

- seed 123 replication 2: strict final delta `+0.565958`, versus standard
  `+0.194006`;
- seed 456 replication 2: strict final delta `+1.086182`, versus standard
  `-1.092979`.

Both seeds have only two improving runs out of three and would require the final
run to improve to reach the registered 3/4 count. More decisively, seed 456's
three deltas `[-0.466870, -0.785473, +1.086182]` are already too heterogeneous:
no real-valued fourth delta can make `mean + 1.96 * SE < 0`. The registered gate
is mathematically unreachable.

The strict arm still accepted nonzero coordinates in every completed replication
and found joint-only rescues in every replication. Raising the threshold therefore
does not remove the interaction phenomenon; it fails to turn it into reliable
held-out recovery.

## Decision

Reject fixed 0.5% pair filtering. Do not complete replication 3 solely for this
gate and do not continue tuning scalar acceptance thresholds against these locked
results. EXP-085 remains an independent test of joint versus greedy selection and
must continue unchanged.

## Artifact integrity

- Full JSON SHA-256:
  `d76d14801341e05b12a833e822f39c5d576392cab89aa1eb16f58c8618174add`
- Markdown summary SHA-256:
  `124d6c77e73d98238533336311b6f1db1ac1702dd0b0f98c84c1b06710300c18`

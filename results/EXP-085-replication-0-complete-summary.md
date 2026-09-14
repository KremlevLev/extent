# EXP-085 replication 0 complete result

Reviewed: 2026-09-14

## Status

Replication 0 completed both registered source seeds and both matched arms on TPU
v5e-8. This supersedes the earlier partial replication-0 observation, but it does
not complete the four-replication EXP-085 gate.

## Locked PG-19 test result

| Seed | Arm | Start NLL | Final NLL | Delta | Pair minus greedy |
|---:|---|---:|---:|---:|---:|
| 123 | GREEDY-PAIR-GRID | 13.526156 | 12.718641 | -0.807515 | |
| 123 | PAIR-LOOKAHEAD | 13.526156 | 12.400920 | -1.125236 | -0.317721 |
| 456 | GREEDY-PAIR-GRID | 14.348403 | 13.034693 | -1.313710 | |
| 456 | PAIR-LOOKAHEAD | 14.348403 | 13.881152 | -0.467251 | +0.846459 |

Both arms improve the untouched locked-test NLL for both seeds and neither arm
exceeds its starting NLL at a registered milestone. Pair lookahead is substantially
better for seed 123 but substantially worse for seed 456. Its mean paired advantage
over the two seeds in this one data replication is therefore `+0.264369` NLL: the
current point estimate favours greedy selection, but one data replication is not
the registered uncertainty unit and cannot decide the experiment.

Pair lookahead finds four joint-only rescue pairs for seed 123 (`4-5`, `7-8`,
`11-12`, `18-19`) and one for seed 456 (`4-5`). This confirms that pairwise
interactions exist; it does not show that exploiting them improves final recovery
reliably.

## Decision

Continue the unchanged pre-registered EXP-085 protocol through replications 1-3.
The primary claim requires four independent data replications for each seed. A
negative mean pair-minus-greedy result remains possible, but pair lookahead now has
to overcome the seed-456 reversal rather than merely repeat the seed-123 win.

The downloaded outer campaign JSON is an earlier snapshot from commit
`ff17e90f62e09ceafe602ba4ab9e847cc7b4c37b`: it contains the old partial inner
record and consequently reports `0/4`. The completed inner artifact was produced
from commit `674175c396a7ed89ffce1c54234884695b1ca915` and is authoritative for this
replication. Resuming the same campaign will restore the completed inner artifact,
replace the stale outer entry, and proceed to the remaining replications.

## Artifact integrity

- Full completed replication JSON SHA-256:
  `76de89b1dae141298b0fff86c5f40d2d3c0a927c47d9b5eb556fcbc54d00ebf1`
- Completed replication summary SHA-256:
  `63827341d4f2d560990258fc9d8bbe0b97b77c2ff8f0e1d62045a6273d0c458a`
- Earlier outer snapshot SHA-256:
  `d643f8c74f3704bbe9476a00341cd5cb1bd5b569622d82f80d59862d41c3544b`

# EXP-087 first partial result

Reviewed 2026-09-17. Three of four fresh data replications completed in 5.293
TPU hours at revision `f837423db675a47e2c156d772a187d4911b2dfd1`.
No outer durability warnings. Stop reason: conservative start-headroom guard.

| Seed | Joint mean final-start NLL | Normal 95% upper | Improving | Joint minus independent |
|---:|---:|---:|---:|---:|
| 123 | -1.420622 | -0.638980 | 3/3 | +0.222466 |
| 456 | -0.807370 | -0.496121 | 3/3 | -0.074296 |

Both methods receive equal updates per subtree, not equal FLOPs. Joint proposals
improve their unchanged start in every completed trajectory. They beat independent
proposals in only 1/3 seed-123 and 2/3 seed-456 replications. Seed-123 paired
differences are `[-1.483844,+0.828345,+1.322897]`; seed-456 differences are
`[-0.087638,+0.416631,-0.551882]`. This is not stable comparative superiority.

Recorded proposal-phase totals per replication are about 410-414 seconds joint
versus 483-485 independent for seed 123, and 314-318 versus 331-333 for seed 456.
These include cache/target preparation and compilation, omit alpha-grid cost, and
are affected by branch execution order; they are not FLOP or general throughput
claims.

The registered gate remains reachable. Seed 123 needs fourth-replication joint
minus independent NLL below `-0.667398` to reverse the mean comparator. Seed 456
needs that difference below `+0.222889` to preserve a negative mean comparator;
the remaining stability/mean-bound requirements also still apply.

Decision: resume the identical campaign to finish replication 3. Do not tune the
proposal objective, threshold or optimizer using this partial result. The existing
three replications are durable HF boundaries and should not retrain.

Full artifact SHA-256:
`3d433ff3336bf4040dce9d9a4d8b68aedaeef679057c1d388f10783176cb8e87`

Summary SHA-256:
`9a4ff435adffd5906845465788606cf1a573e97e13cded18b0bb2273619fbc71`

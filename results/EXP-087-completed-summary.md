# EXP-087 completed result

Four of four data replications complete for both seeds.
Final resumed invocation: 1.797 hours (rounded summary); previous invocation
5.293 hours. No outer durability warnings. Overall registered gate fails.

| Seed | Joint mean final-start NLL | Normal 95% upper | Improving | Joint minus independent | Gate |
|---:|---:|---:|---:|---:|---|
| 123 | -1.363270 | -0.799251 | 4/4 | +0.439904 | fail |
| 456 | -0.873675 | -0.618083 | 4/4 | -0.103454 | pass |

Joint training improves the unchanged warm start in all eight trajectories, but
beats independent training in only 1/4 seed-123 and 3/4 seed-456 replications.
The fourth paired difference is +1.092218 for seed 123 and -0.190928 for seed 456.
The seed-averaged paired point estimate is +0.168225 NLL, favouring independent
proposals; pooling cannot override the registered per-seed requirement.

This rejects the tested composed-segment relative-MSE proposal as a reliable
improvement over independent proposals. It does not reject every form of joint
training: this objective sees only the segment output, not its effect through the
frozen downstream network on the language-model distribution. There is no
demonstrated inference-speed benefit or capability recovery here.

Decision: no unchanged rerun and no more threshold tuning. Preserve artifacts as
a composition-aware proposal baseline. A next method needs a substantively
different training signal (e.g. downstream-sensitive loss), not more evidence for
this same negative comparator result. Such a method must have a fresh protocol
and compute accounting before being run.

Full artifact SHA-256:
`bd880edc420ddac85acda01d3b9e9f8971a02a31d0c1d36e0aab73798dc3707b`

Summary SHA-256:
`795b61ed7462b8fe91fbfa688dd31107bbfce0ee55fd4b295a72fc0051e1984a`

# EXP-072 v2 — first partial full-depth sequential confirmation

Reviewed 2026-09-08. Status: durability stop at `seed-456/ONPOLICY/layer-22`; the registered depth-24 gate is not yet evaluable.

The supplied full artifact has SHA-256 `70c7950609b8aece142ba836c39ad13f7f3f363a61652d286bae066a67d6cc74`; its compact summary has SHA-256 `05c810e7be555f705d185a056bfa7b7a82bc1ddf8a1631044c3e8aa1ec8823ae`. Runtime before the stop was `1.033192` TPU hours at revision `6da275f`.

## What completed

Seed 123 completed all three matched arms through 20 replacements. Seed 456 completed all three through 16; its `MIXED` branch also reached 20, while the `ONPOLICY` layer-22 payload remained only local after all HF retries failed. The checkpoint guard stopped before using an endpoint that was not durably stored. All completed training metrics and reported evaluations are finite.

| Depth | Seed | TEACHER NLL | ONPOLICY NLL | Delta | MIXED NLL |
|---:|---:|---:|---:|---:|---:|
| 4 | 123 | 6.770640 | 6.091796 | -0.678844 | 6.197317 |
| 8 | 123 | 7.695405 | 6.348043 | -1.347362 | 6.465469 |
| 12 | 123 | 12.648251 | 6.368623 | -6.279628 | 6.480289 |
| 16 | 123 | 16.957157 | 6.378428 | -10.578729 | 6.489104 |
| 20 | 123 | 23.431106 | 7.004257 | -16.426849 | 6.917953 |
| 4 | 456 | 6.553622 | 6.147485 | -0.406138 | 6.151043 |
| 8 | 456 | 7.261526 | 6.415367 | -0.846158 | 6.456084 |
| 12 | 456 | 11.359790 | 6.496694 | -4.863096 | 6.518410 |
| 16 | 456 | 16.240802 | 6.504203 | -9.736599 | 6.548587 |

At the last common registered milestone, depth 16, `ONPOLICY` reduces total NLL relative to equal-update `TEACHER` by `62.39%` for seed 123 and `59.95%` for seed 456. It reduces excess NLL above the cached Qwen teacher by `76.16%` and `73.91%`. The teacher-only recovery curve collapses after depth 8, whereas both deployment-conditioned arms remain close to NLL 6.4–6.5 through depth 16. This independently reproduces EXP-071's mechanism on a fresh train/evaluation slice and a four-times-longer per-layer budget.

At seed 123 depth 20, both sequential arms worsen by roughly 0.4–0.6 NLL, so the final four replacements remain scientifically important. `MIXED` is marginally better than `ONPOLICY` at that single point (`6.917953` versus `7.004257`); this does not overturn the registered primary comparison against `TEACHER`.

## Decision

Resume the identical EXP-072 v2 contract. Do not create a new experiment or change hyperparameters. A fresh session restores all durable endpoints; at worst it repeats only the failed seed-456 `ONPOLICY` layer-22 update before continuing through depth 24. The displayed `scientific_gate=False` is an incomplete-result default, not a failed hypothesis.

The shared checkpoint uploader now waits through more than seven minutes of transient HF outage (`2/5/15/30/60/120/240` seconds) before a durability stop. The scientific contract and all already stored endpoints are unchanged.

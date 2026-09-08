# EXP-071 — first partial sequential-recovery run

Reviewed 2026-09-08. Status: engineering failure after depth 7; the registered depth-12 scientific gate is not evaluable.

The user-supplied full artifact has SHA-256 `3a29615bc63fd4d7ecf60325345116d8039da0bce9a77c740f80e49245535665`; the supplied compact summary has SHA-256 `8c39f9fe9b06385af3459611e217f27571afb120f04de0a63e802bcf021bd113`. Run revision: `01e2f77cd15c2594b22503393eb8ae1a345d4aef` on one TPU v5e-8.

## What completed

The run lasted `0.137749` hours. All six arms completed and uploaded through layer 5 (depth 6). At layer 7, seed-123 completed all three arms and seed-456 completed `TEACHER` and `ONPOLICY`. Seed-456 `MIXED` trained but its checkpoint failed all three rapid HF attempts, so it was deliberately not accepted as a durable endpoint. The campaign stopped before dependent work. Forty-one new endpoint uploads and all 24 required EXP-069 restores are recorded as successful.

This is not a numerical failure: every reported evaluation is finite, and the terminal error is an `HfHubHTTPError` followed by the campaign's intentional durability guard. The failed layer must be deterministically repeated on resume; earlier durable layers must not be recomputed.

## Partial depth-4 evidence

Held-out NLL at the last common registered evaluation point:

| Seed | No-extra-update baseline | TEACHER | ONPOLICY | MIXED |
|---:|---:|---:|---:|---:|
| 123 | 6.880352 | 6.824862 | 6.287669 | 6.397811 |
| 456 | 6.895630 | 6.440594 | 6.337772 | 6.445904 |

Relative to the matched `TEACHER` arm, `ONPOLICY` improves NLL by `0.537193` and `0.102823` at seeds 123 and 456. It also beats the no-extra-update baseline by `0.592683` and `0.557858`. This is consistent two-seed early evidence that deployment-distribution recovery is actionable.

`MIXED` improves over `TEACHER` by `0.427050` at seed 123 but is worse by `0.005310` at seed 456. Therefore the registered primary method is not yet supported even at depth 4. Depth 1 is not informative for the input-distribution intervention because no earlier replacement exists; `TEACHER` and `ONPOLICY` are byte-identical there, as expected.

These values are interim looks from an incomplete pre-registered campaign. They do not change its arms, gate, depth or update budget.

## Engineering correction

The first implementation uploaded two compact summaries after every arm in addition to every required endpoint, creating avoidable bursts of HF commits. The correction keeps every layer endpoint durable but uploads campaign summaries only at registered depths and terminal states. Endpoint retries now use six attempts with waits of 2/5/15/30/60 seconds. The experiment contract and numerical method are unchanged.

Rerun the identical EXP-071 entry point on the corrected commit. A full result is required before deciding between `TEACHER`, `ONPOLICY`, and `MIXED` or designing the depth-24 extension.

# EXP-090 — session 3, first completed pairable intervention endpoint

Reviewed 2026-09-22 from user-supplied full and compact artifacts.

- Status: `deadline_partial`; session duration `7.7894` hours.
- Seed-123 `RANDOM-DT-FROZEN` completed 98,304 updates / 25,165,824 tokens.
- Seed-456 `RANDOM-DT-FROZEN` reached step 67,648 and remains resumable.
- All 13 HF checkpoint uploads passed; all reported evaluations were finite.

At the seed-123 registered endpoint, the frozen-dt arm has excess NLL `9.538814` and prediction KL `14.329857`. Its ordinary-random control is complete in EXP-089 at the same step and has excess NLL `9.763291`, prediction KL `14.612423`. The control trajectories are deterministic and identical at all shared registered checkpoints, so this is a matched descriptive comparison, not an independent replication. Frozen-minus-random differences are `−0.224476` NLL and `−0.282565` KL, favoring the intervention by approximately 2.30% and 1.93% relative to the control endpoints. This is a small positive seed-123 result, not a passed scientific gate: the seed-456 random control is not yet complete and seed-456 frozen-dt is only partial.

The seed-456 frozen-dt curve remains strongly non-monotonic: excess NLL `9.497803` at step 32,768, `10.429059` at step 65,536, `10.140849` at step 67,648. No comparison to seed-456 random is yet available. Resume EXP-090 unchanged on the remaining TPU allocation, finishing the frozen-dt endpoint and then the pending ordinary-random controls. Do not launch another dt schedule based on one small seed-123 win.

Full artifact SHA-256: `b66ee2daec864ee59c43b76e2392dfcf726109fef88c5edb1f47f2945262152106543`.
Run revision: `00ac38b3d3f853c487ecdd5457621b84dfb9dbcc`.

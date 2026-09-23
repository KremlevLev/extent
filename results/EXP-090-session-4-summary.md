# EXP-090 — session 4, three of four trajectories complete

Reviewed 2026-09-23 from user-supplied full and compact artifacts.

- Status: `deadline_partial`; invocation duration `7.7882` hours.
- Completed trajectories: seed-123 frozen-dt and random; seed-456 frozen-dt.
- Seed-456 random control reached step 37,056 of 98,304, so 61,248 updates remain.
- All recorded metrics are finite. All 15 HF checkpoint upload events passed, including seed-456 random at step 37,056.

## Completed seed-123 pair

At the registered 98,304-step endpoint, frozen-dt minus random is `−0.224476` excess NLL and `−0.282565` prediction KL. The modest seed-123 win is stable as an endpoint observation, though the within-training curves were highly non-monotonic.

## Seed-456 progress

Frozen-dt completed at step 98,304 with excess NLL `9.657643` and prediction KL `14.382832`. Random control is only at step 37,056 with excess NLL `9.693641` and KL `14.371384`. These are different checkpoints and cannot be interpreted as a paired endpoint comparison.

At the shared registered checkpoints, frozen-dt minus random excess NLL is `−2.224502` at 3,072, `+1.443281` at 8,192, `+2.219996` at 16,384, and `−0.521903` at 32,768. The signs alternate; the endpoint remains unknown. The scientific gate is pending, not failed.

## Decision

Resume the same EXP-090 entry point from the verified seed-456 random checkpoint at step 37,056. Finish the remaining 61,248 updates and compare both arms only at step 98,304. No further intervention tuning is justified before this endpoint.

Full artifact SHA-256: `08b2a11399893e6a48f93b4b62ea00e85e4deafe566c9941349106e228856b76`.
Compact summary SHA-256: `eac490c34f86c97c3bd68d019e27c6354e14c7cbfe80aa4f427cdf4ff08b415f`.
Run revision: `f602dd823c1ff383f41c42211106ab01bcbb8bcd`.

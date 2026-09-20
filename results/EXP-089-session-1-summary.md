# EXP-089 — session 1 partial result

Reviewed 2026-09-20 from the user-supplied full and compact artifacts.

## Durable progress

- Invocation status: `deadline_partial`, as designed.
- Wall time: `7.7867` hours.
- Completed trajectory: `seed 123 / RANDOM`, step `83,360 / 98,304`.
- Tokens processed: `21,340,160 / 25,165,824` (`84.80%` of the first trajectory).
- Numerical health: finite through the last recorded update.
- HF durability: all 12 upload events pass, including the session-boundary checkpoint at step 83,360. The next invocation must restore rather than retrain this trajectory.

## Measured random-baseline curve

| Step | Tokens | Excess NLL | Prediction KL | Top-1 agreement |
|---:|---:|---:|---:|---:|
| 0 | 0 | 16.044501 | 22.114488 | 0.00000 |
| 3,072 | 786,432 | 11.814042 | 17.189559 | 0.00760 |
| 8,192 | 2,097,152 | 14.182362 | 19.986859 | 0.00208 |
| 16,384 | 4,194,304 | 6.844993 | 11.040131 | 0.06667 |
| 32,768 | 8,388,608 | 9.817048 | 14.495236 | 0.01667 |
| 65,536 | 16,777,216 | 9.806951 | 14.564223 | 0.00931 |
| 83,360 | 21,340,160 | 9.718625 | 14.512564 | 0.01078 |

The first large result is not monotonic: the best registered checkpoint so far is step 16,384, after which held-out quality degrades and then plateaus around excess NLL 9.7–9.8. This is useful optimizer/recovery evidence, but it says nothing yet about the primary hypothesis because the paired exact-lift trajectory has not started. The scientific gate must remain false.

## Decision

Resume unchanged. Approximately 14,944 updates remain in the first random trajectory; after it completes, the same invocation should begin `seed 123 / EXACT-LIFT`. Do not restart, change learning rate, or reinterpret the step-16,384 minimum post hoc. The registered endpoint and full curves remain authoritative.

## Provenance

- Run revision: `dbb2baf8cfe322e6802d46530ff4fa58bb936181`.
- Full artifact SHA-256: `d96a7ad5e9405200b67fd4f80be524dacc32d4b3d8c35c069980629289eea585`.
- Compact summary SHA-256: `3cfc3255c3d507b75abb3010561a4bb95725a8a6f797f96c8eb82f09bec14abd`.

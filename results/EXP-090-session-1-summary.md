# EXP-090 — session 1 partial result

Reviewed 2026-09-21 from the user-supplied full and compact artifacts.

## Durable progress

- Invocation status: `deadline_partial`, as designed.
- Wall time: `7.7876` hours.
- Completed trajectory: `seed 123 / RANDOM`, step `83,936 / 98,304`.
- Tokens processed: `21,487,616 / 25,165,824`.
- Numerical health: finite.
- HF durability: all 12 uploads pass, including the session-boundary checkpoint at step 83,936.

The `RANDOM-DT-FROZEN` intervention has not started. Therefore this invocation contains no evidence for or against H1.2 and the scientific gate correctly remains false.

## Duplicate-control audit

EXP-090's control intentionally has the same source seed, model, initializer, data order, objective, optimizer, and evaluation set as EXP-089's first trajectory. The registered evaluations are byte-for-value identical at every common checkpoint through step 65,536:

| Step | Excess NLL | Prediction KL |
|---:|---:|---:|
| 0 | 16.044501 | 22.114488 |
| 3,072 | 11.814042 | 17.189559 |
| 8,192 | 14.182362 | 19.986859 |
| 16,384 | 6.844993 | 11.040131 |
| 32,768 | 9.817048 | 14.495236 |
| 65,536 | 9.806951 | 14.564223 |

This is a useful determinism check but not an independent replication. Spending the first session on it was operationally inefficient. The execution order is changed, without changing the scientific contract, so subsequent EXP-090 sessions prioritize `RANDOM-DT-FROZEN` for seeds 123 and 456 before returning to unfinished controls. The durable control checkpoint is retained for the final paired comparison.

## Decision

Resume with the updated entry point. It must begin `seed 123 / RANDOM-DT-FROZEN` at step zero, freeze dt through step 32,768, then continue the same trajectory with dt trainable. Do not delete the existing `seed-123/random` checkpoint.

## Provenance

- Run revision: `2e62da8a8f35c2ce3abf1451cda1648911ab1c60`.
- Full artifact SHA-256: `4c55ca6db61e71835beac1d74f14ad348f23db79eb6796a3f934bce021317ad2`.
- Compact summary SHA-256: `a3492b0a52a32660c4c9045945386c4a2b236867365d7cc0eb71ccc6fedb3bfd`.

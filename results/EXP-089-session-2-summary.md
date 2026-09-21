# EXP-089 — session 2 partial result

Reviewed 2026-09-21 from the user-supplied full artifact, SHA-256 `4d142142b011993f114106993b24457cd5ebf6bb650007ed653d40f099771f6b`.

- Status: `deadline_partial`; invocation duration `7.7872` hours.
- `seed 123 / RANDOM`: complete at step 98,304 (25,165,824 tokens), excess NLL `9.763291`, prediction KL `14.612423`.
- `seed 123 / EXACT-LIFT`: partial at step 67,072 (17,170,432 tokens), excess NLL `14.357255`, prediction KL `19.218206`.
- Numerical health: both trajectories finite. All 13 checkpoint upload events in this invocation passed; latest uploaded state is exact-lift step 67,072.

## Matched-step comparison

Positive differences mean exact lift is worse than random.

| Step | Random excess NLL | Exact excess NLL | Exact − random | Random KL | Exact KL | Exact − random KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16.044501 | 17.620153 | +1.575652 | 22.114488 | 22.876497 | +0.762009 |
| 3,072 | 11.814042 | 16.451861 | +4.637819 | 17.189559 | 21.467528 | +4.277969 |
| 8,192 | 14.182362 | 16.872633 | +2.690271 | 19.986859 | 22.175675 | +2.188816 |
| 16,384 | 6.844993 | 15.170726 | +8.325733 | 11.040131 | 20.196377 | +9.156246 |
| 32,768 | 9.817048 | 16.792630 | +6.975582 | 14.495236 | 22.170850 | +7.675614 |
| 65,536 | 9.806951 | 14.812061 | +5.005110 | 14.564223 | 19.787677 | +5.223454 |

The exact-lift curve remains worse at every shared registered checkpoint through 16.78M tokens. The partial point at step 67,072 improves relative to exact step 65,536, but it cannot be compared directly with random's step-98,304 endpoint. No late crossover has been observed. The pre-registered 25.166M-token endpoint and second seed are still pending; the scientific gate remains false because incomplete, not because the final hypothesis has already been fully tested.

## Decision

Resume the same campaign and checkpoint. Complete `seed 123 / EXACT-LIFT` before assessing the long-horizon endpoint; do not change the learning rate, data, optimizer, or stopping rule based on this partial curve. The next invocation should restore exact-lift step 67,072. No new initializer variation is justified by these data.

Run revision: `2e62da8a8f35c2ce3abf1451cda1648911ab1c60`.

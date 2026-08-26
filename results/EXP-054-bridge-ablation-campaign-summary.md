# EXP-054 attention bridge initialization screen

- Status: `completed`
- Numerical pass: `True`
- Complete: `True`
- Duration: `0.300` hours
- Screening gate: `False`

| Layer | Arm | Mean decoder relative L2 | Arm - random | Wins |
|---:|---|---:|---:|---:|
| 0 | APPLE-BRIDGE | 0.56515338 | +0.01732394 | 0/3 |
| 0 | BRIDGE-PLUS-ORIENTATION | 0.56612708 | +0.01829764 | 0/3 |
| 0 | MOHAWK-ORIENTATION | 0.54821586 | +0.00038643 | 0/3 |
| 0 | CONTROL-QKVO | 0.59610430 | +0.04827486 | 0/3 |
| 18 | APPLE-BRIDGE | 0.00633959 | +0.00083432 | 0/3 |
| 18 | BRIDGE-PLUS-ORIENTATION | 0.00631139 | +0.00080611 | 0/3 |
| 18 | MOHAWK-ORIENTATION | 0.00551168 | +0.00000641 | 0/3 |
| 18 | CONTROL-QKVO | 0.00553378 | +0.00002851 | 0/3 |

Negative `Arm - random` is better. This is a two-layer initialization screen, not full-model recovery.

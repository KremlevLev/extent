# EXP-055 stabilized partial-RoPE bridge confirmation

- Status: `completed`
- Numerical pass: `True`
- Complete: `True`
- Duration: `0.706` hours
- Screening gate: `False`

| Layer | Arm | Mean decoder relative L2 | Arm - random | Wins |
|---:|---|---:|---:|---:|
| 0 | APPLE-BRIDGE | 0.52526938 | +0.00442369 | 0/3 |
| 0 | BRIDGE-PLUS-ORIENTATION | 0.52631559 | +0.00546989 | 0/3 |
| 0 | MOHAWK-ORIENTATION | 0.52196162 | +0.00111593 | 0/3 |
| 0 | CONTROL-QKVO | 0.56601205 | +0.04516636 | 0/3 |
| 18 | APPLE-BRIDGE | 0.00616920 | +0.00106029 | 0/3 |
| 18 | BRIDGE-PLUS-ORIENTATION | 0.00606250 | +0.00095358 | 0/3 |
| 18 | MOHAWK-ORIENTATION | 0.00511375 | +0.00000483 | 0/3 |
| 18 | CONTROL-QKVO | 0.00505362 | -0.00005529 | 3/3 |

Negative `Arm - random` is better. This is a two-layer initialization screen, not full-model recovery.

# EXP-056 operator-preserving MIMO lift campaign

- Status: `completed`
- Duration: `1.713` hours
- Complete: `True`
- Numerical pass: `True`
- Screening gate: `True`

| Layer | Arm | Final Δ vs random | Wins | Earliest crossover |
|---:|---|---:|---:|---:|
| 0 | CONTROL-FLAT-QKVO | +0.04516636 | 0/3 | none |
| 0 | SINGLE-CHANNEL-LIFT | -0.01698945 | 3/3 | 256 |
| 0 | BALANCED-RANK-LIFT | -0.01695754 | 3/3 | 256 |
| 6 | CONTROL-FLAT-QKVO | +0.00086315 | 1/3 | none |
| 6 | SINGLE-CHANNEL-LIFT | -0.00135171 | 2/3 | 256 |
| 6 | BALANCED-RANK-LIFT | -0.00143303 | 3/3 | 256 |
| 18 | CONTROL-FLAT-QKVO | -0.00005529 | 3/3 | 4096 |
| 18 | SINGLE-CHANNEL-LIFT | -0.00029044 | 3/3 | 256 |
| 18 | BALANCED-RANK-LIFT | -0.00028904 | 3/3 | 256 |
| 29 | CONTROL-FLAT-QKVO | +0.00026044 | 0/3 | none |
| 29 | SINGLE-CHANNEL-LIFT | -0.00047378 | 3/3 | 256 |
| 29 | BALANCED-RANK-LIFT | -0.00048603 | 3/3 | 256 |

Negative delta is better. Layer 0 is a boundary diagnostic and cannot veto the internal-layer gate.

# EXP-058 exact-lift two-layer composition pilot

- Duration: `0.195` hours
- Numerical pass: `True`
- Pilot gate: `True`

| Branch | Mean NLL | Excess vs original |
|---|---:|---:|
| Original Qwen | 4.28904451 | +0.00000000 |
| Random Mamba layers 0+18 | 5.59118033 | +1.30213582 |
| Exact-lift Mamba layers 0+18 | 5.27788715 | +0.98884264 |

Exact lift minus random NLL: `-0.31329318`.

This is a one-seed, 1,024-step pilot. It ranks initialization under composition but is not publication-level confirmation.

# EXP-066 scale-stabilized Mamba-3 exact-dual bridge

- Status: `completed`
- Duration: `3.984` hours
- Completed layers: `5/5`
- Numerical pass: `True`
- Scientific gate: `False`

| Layer | Random | Exact lift | Legacy dual-random | Stabilized no-complex | Stabilized complex |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.70370574 | 0.72502176 | **0.52670581** | 0.72193706 | 0.73133985 |
| 6 | 0.00705545 | **0.00648532** | 0.00754297 | 0.00814421 | 0.00814574 |
| 13 | 0.01329527 | **0.01284126** | 0.01503091 | 0.01418646 | 0.01461715 |
| 20 | 0.11454336 | **0.09483430** | 0.09544526 | 0.11634783 | 0.11858792 |
| 27 | 1.08153756 | 1.58655853 | **0.07573538** | 1.06570580 | 1.19165889 |

- Stabilized complex beats random at `0/5` layer gates.
- Stabilized complex beats exact lift at `1/5` layer gates.
- Stabilized complex beats legacy dual-random at `1/5` layer gates.
- Stabilized complex beats stabilized no-complex at `0/5` layer gates.
- Legacy dual-random improves final layer-27 error over random by `93.00%` and over exact lift by `95.23%`.
- Legacy dual-random improves layer-0 final error over random by `25.15%`.
- Full artifact SHA-256: `d028fb20d1428ae54b19e4505aa60610d1f4ec821680f18e303f05b4ee0f3e86`.
- User summary SHA-256: `5d1608b330de088357c6b3d0df7bfe649719bbee84401a17525c6d771226818e`.

Pre-readout calibration plus token whitening is rejected.  The confirmed result
is extreme depth dependence of the original random exact-dual preparation,
motivating compatibility-guided placement of the fixed attention budget.

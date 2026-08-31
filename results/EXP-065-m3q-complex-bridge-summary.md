# EXP-065 Mamba-3 exact-dual complex bridge

- Status: `deadline_partial`
- Duration: `8.545` hours
- Completed layers: `4/5`
- Numerical pass: `True`
- Scientific gate: `False` (the pre-registered five-layer gate cannot pass on a partial run)

| Layer | Arm | Final decoder L2 | Normalized AUC |
|---:|---|---:|---:|
| 0 | CONTROL-RANDOM | 0.69553890 | 0.70275747 |
| 0 | BALANCED-RANK-LIFT | 0.71242480 | 0.72381520 |
| 0 | M3Q-DUAL-RANDOM | 0.48706192 | 0.49420254 |
| 0 | M3Q-DUAL-EXACT-NO-COMPLEX | 0.56518755 | 0.56870060 |
| 0 | M3Q-DUAL-EXACT | 0.56092268 | 0.56432777 |
| 6 | CONTROL-RANDOM | 0.00702391 | 0.00726951 |
| 6 | BALANCED-RANK-LIFT | 0.00640434 | 0.00671320 |
| 6 | M3Q-DUAL-RANDOM | 0.00719967 | 0.00731904 |
| 6 | M3Q-DUAL-EXACT-NO-COMPLEX | 0.00790757 | 0.00800015 |
| 6 | M3Q-DUAL-EXACT | 0.00783893 | 0.00793087 |
| 13 | CONTROL-RANDOM | 0.01341720 | 0.01407913 |
| 13 | BALANCED-RANK-LIFT | 0.01297816 | 0.01378432 |
| 13 | M3Q-DUAL-RANDOM | 0.01514914 | 0.01532126 |
| 13 | M3Q-DUAL-EXACT-NO-COMPLEX | 0.01564041 | 0.01579163 |
| 13 | M3Q-DUAL-EXACT | 0.01559061 | 0.01574119 |
| 20 | CONTROL-RANDOM | 0.12443635 | 0.17727814 |
| 20 | BALANCED-RANK-LIFT | 0.10517961 | 0.17620751 |
| 20 | M3Q-DUAL-RANDOM | 0.10675125 | 0.11037530 |
| 20 | M3Q-DUAL-EXACT-NO-COMPLEX | 0.25047976 | 0.57006991 |
| 20 | M3Q-DUAL-EXACT | 0.11196742 | 0.11318314 |

- Complete exact-dual beats no-complex at every completed depth (`4/4`).
- Exact QKVO seeding never beats random seeding after dual preparation (`0/4`).
- At layer 0, dual-random improves final error over random by `29.97%` and AUC by `29.68%`.
- At layer 20, dual-random improves AUC over random by `37.74%` but is `1.49%` worse at the final endpoint than balanced lift.
- Full artifact SHA-256: `c05cf63c39ba2911f514368d13d83313de56ca98a7c796d25bd872f48bf0782a`.
- User summary SHA-256: `f7371c893c8067f7a1582219397ea851201521d5fe5f6b9985db9888d9466fbc`.

The exact-seeded primary method is rejected.  The useful signal is instead a
random-seeded exact-dual bridge plus complex dynamics, with strong but
depth-dependent effects.  EXP-066 tests whether pre-bridge readout calibration
and token-whitened loss remove that depth dependence.

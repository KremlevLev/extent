# EXP-064 Qwen3-1.7B full-model M3Q distillation

- Status: `completed`
- Duration: `0.979` hours
- Completed arms: `3/3`
- Scientific gate: `False`

| Arm | Steps | Final excess NLL | Final KL | Agreement | Excess-NLL AUC |
|---|---:|---:|---:|---:|---:|
| EXACT-KL | 3072 | 15.0937189 | 19.9017954 | 0.00049 | 14.6369559 |
| M3Q-HIDDEN-BRIDGE-KL | 3072 | 15.3031470 | 20.2514548 | 0.00000 | 16.4739435 |
| RANDOM-KL | 3072 | 10.8779436 | 15.9447466 | 0.00980 | 11.0433967 |

- Staged final excess-NLL gain over exact KL: `-1.388%`.
- Staged excess-NLL AUC gain over exact KL: `-12.550%`.
- Random final excess-NLL gain over exact KL: `+27.932%`.
- Random excess-NLL AUC gain over exact KL: `+24.550%`.
- Full artifact SHA-256: `a94cb2e1a872043be326d7504a60edf337e051a4e3d2fbd03548c5612b3e3402`.
- User summary SHA-256: `867409f96db569997377c1d878473beddb8f3d603d3d9c2168abce14cef1e3d3`.

This rejects staged hidden-state alignment as implemented and rejects the current
QKVO exact lift as a useful whole-model initializer under this protocol.  It
does not establish that random initialization is universally preferable: the
three arms have one deterministic data order and no paired seed replication.

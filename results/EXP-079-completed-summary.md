# EXP-079 — attention-bounded depth segment recovery (completed)

- Status: completed in 4.576908 hours at implementation revision `2ab48a89279c59cb3d97df86987468550d222402`.
- Artifacts supplied by the user: full JSON SHA-256 `d8117dc51c5f8416c29d0720797d9a9dbe86075e7d34f45d35fdca1804e83719`; summary SHA-256 `c92075a1251fd9883afbf10f782a523dbaefdce915b90c526cbb6b7994de91`.
- Registered gate: **FAIL**. Primary `SEGMENT-4` improves seed 123 but worsens seed 456.

| Segment | Seed 123 final delta | Seed 456 final delta | Best observed delta |
|---|---:|---:|---:|
| 1 (layers 0-5) | +0.462950 | +1.193874 | 0.000000 |
| 2 (layers 7-12) | +1.140890 | +0.176714 | 0.000000 |
| 3 (layers 14-19) | +0.093457 | +0.218306 | -0.203057 (seed 123, step 2048) |
| 4 (layers 21-26) | -0.542634 | +0.483416 | -0.542634 (seed 123, final) |

## Interpretation

Restricting updates to one attention-bounded segment keeps every trajectory within the 1.25x excursion limit, confirming that simultaneous full-depth movement was a major source of violent instability. It does not produce seed-consistent recovery: no segment beats its own start at both seeds, and every seed-456 segment is worse at the final checkpoint.

The pre-registered depth hypothesis is only half-supported. The deepest segment is decisively best for seed 123 and improves monotonically after step 2048, but it does not generalize to seed 456. Segment 3 also briefly improves seed 123. This rejects selecting one globally trainable depth region from this atlas.

Together with EXP-072, the evidence now favors local conditional recovery over full-model KL updates even when those global updates are protected, adaptive and depth-restricted. The next method is an iterative second on-policy coordinate sweep: revisit every already transplanted Mamba block using the current hybrid prefix and compare forward versus reverse sweep order.

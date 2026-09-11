# EXP-081 — assembled-model trust-region coordinate recovery (completed)

- Status: completed in `0.694239` TPU hours; both registered seeds and both matched arms completed with finite trajectories and no durability warnings.
- Registered gate: pass.
- Seed 123 `TRUST-LINE`: locked NLL `10.116380 -> 9.902654` (`-0.213726`, `-2.113%`); maximum milestone ratio `1.02296`; six of 24 proposed coordinates accepted.
- Seed 456 `TRUST-LINE`: locked NLL `12.315644 -> 12.286946` (`-0.028699`, `-0.233%`); maximum milestone ratio `1.0`; five of 24 coordinates accepted.
- Hard-accept control: worsens NLL by `+0.598795` and `+0.460585`. TRUST-LINE beats its matched hard control by `0.812521` and `0.489284` final NLL.
- Selected interpolation is materially different from binary acceptance. Seed 123 uses alphas `0.25/0.5/1.0/0.75/0.25/0.5`; seed 456 uses `1.0/0.5/1.0/1.0/0.25`. Only layer 26 is accepted by TRUST-LINE at both seeds.
- Interpretation: evaluating a proposed coordinate inside the assembled model prevents most locally beneficial but globally harmful updates. Fractional interpolation, rather than order alone or binary accept/reject, is the first tested recovery rule to improve full-model NLL at both seeds.
- Boundary: the seed-456 gain is small, accepted layer identity is highly seed-dependent, and the original artifact retained only mean NLL rather than per-window values. This is a promising method discovery, not yet a statistically confirmed paper claim.
- Required confirmation: repeat the frozen recipe across fresh training, calibration and locked evaluation slices; retain per-window NLL; aggregate at the data-replicate level; do not tune the alpha grid or acceptance threshold from the confirmation data.
- Full artifact SHA-256: `d0d4bc607813f1ebbd962088d1e6030b649d6b9b6b2cb943660e7cfd7d246d59`.
- User summary SHA-256: `62d9a45e47d6d8b3494ec575febe7aa04f128ea2b5431979382179f1d412ebd`.

# EXP-082 — multi-split trust-region replication (completed)

- Status: all eight data replications completed in `5.239880` TPU hours; no durability warnings. Three individual EXP-081 gates pass, but the registered aggregate gate fails.
- Seed 123: TRUST-LINE improves PG-19 NLL in `8/8` replications. Mean delta is `-0.723808`, replication SE `0.113996`, and the normal 95% upper bound is `-0.500376`. Maximum milestone ratio is exactly `1.0`.
- Seed 123 control failure: mean `TRUST-LINE - HARD-ACCEPT` final NLL is `+0.340009`; binary acceptance is better on average despite being less reliable in the original EXP-081 slice. The primary arm therefore fails one registered condition.
- Seed 456: only `5/8` replications improve. Mean delta is `+0.129601`, SE `0.416917`, and the 95% upper bound is `+0.946758`. Individual deltas range from `-1.373439` to `+1.920572`; mean TRUST-LINE versus HARD is effectively tied at `+0.024053`.
- Domain-shift diagnosis: WikiText calibration KL decreases in every run by construction, yet seed-456 PG-19 NLL worsens in three runs. The size of calibration improvement does not predict cross-corpus direction. Single-domain assembled acceptance is therefore insufficient.
- Stable coordinate signal: TRUST-LINE accepts layer 0 in `13/16` seed-replications and layer 26 in `15/16`; most other accepted layers are sparse and seed-dependent. This is useful exploratory evidence, not a post-hoc passing subset.
- Decision: retain assembled-model line search as a promising mechanism but reject the current single-domain method as a confirmed general recovery recipe. The next registered change adds a second-domain acceptance constraint and evaluates on untouched PG-19 test books.
- Full artifact SHA-256: `7df5f7aa8da802c5103e8d845caad62beabe9b7955ad8524a6c571eb51c43edb`.
- User summary SHA-256: `8480e83ca6d39e8eb3267167686ba1f605aa41b042254ecc1cf2beb4c77e0fe3`.

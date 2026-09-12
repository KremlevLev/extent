# EXP-083 — dual-domain assembled trust recovery (partial, conclusive negative)

- Status: deadline-partial after `6.477068` TPU hours; seven of eight data replications completed without numerical or durability failures.
- The primary gate is already impossible. Seed 456 improves in only `3/7` completed replications, while the gate requires at least `6/8`; the missing replication can raise this to at most `4/8`.
- Seed 123: mean DUAL-CONSENSUS final-minus-start NLL `-0.401579`, replication SE `0.197579`, provisional normal 95% upper `-0.014325`, with `5/7` negative. It loses to WIKI-ONLY by `+0.276343` mean final NLL.
- Seed 456: mean delta `+0.206161`, SE `0.138688`, upper bound `+0.477989`, with only `3/7` negative. DUAL-CONSENSUS beats WIKI-ONLY by `-0.108642` on average but does not beat its unchanged start.
- DUAL-CONSENSUS accepts nonzero updates in every completed replication, typically one to six coordinates. It does not merely freeze the model, but the accepted updates remain insufficiently transferable.
- Layer 26 remains the most frequent accepted coordinate (`5/7` seed 123, `6/7` seed 456); layer 0 is accepted `4/7` for both. No post-hoc subset is promoted to a passing method.
- Decision: do not consume another TPU hour completing replication seven solely for a formally complete failed result. Mean improvement on two domains is not a sufficient acceptance statistic. Test a paired-window robust rule, matched against mean dual-consensus with equal forwards, on fresh PG-19 ranges.
- Full artifact SHA-256: `f495fdcbaadd941553905fcc207c1dcacc99b9c7256013d0a0ed313ee08e0cc2`.
- User summary SHA-256: `99d0f688440aab2c2f7b7a3b2b49625b2f7f22f68f900e88f03cb7f1dc1ec9c6`.

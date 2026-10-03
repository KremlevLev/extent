# EXP-098: rank-matched dynamics protection

Prepared2026-10-03, no TPU outcomes observed.

Question: at identical rank32, does protecting dt/raw_a/trap/angle projection
weights improve stable whole-model correction learning, and what happens beyond
EXP097's8,192-step plateau?

Primary `SAFE-PROTECTED-R32`; control `SAFE-R32`. Both INOUT low-rank CE,
Adam3e-4, stable clip1, unchanged EXP072-v2 starts at seeds123/456.
Horizons4,096/8,192/16,384;65,536 total steps,16,777,216 input tokens.
Primary gate: all four trajectories complete; primary final Wiki test NLL
at least0.1 below control and below its unchanged start at each seed.
PG19 comparator/unchanged-start confirmation is a separately reported secondary
gate. Validation improvement beyond8,192 is descriptive, not endpoint selection.
Compare both at matched horizons; do not compare one partial against a final.

Shared source pins, **exact data offsets**, evaluation/failure/resume/budget
contract and ready Kaggle cell: [three-account protocol](EXP-098-100-three-account-protocol.md).
Implementation: `scripts/m3q_dynamics_protection_campaign.py`,
`scripts/m3q_safe_recovery_campaign.py`, `scripts/m3q_subspace_engine.py`.
Default8h includes45min reserve; one-session completion is unverified.

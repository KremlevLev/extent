# EXP-099: protected correction capacity

Prepared2026-10-03, no TPU outcomes observed.

Question: can higher input/output correction rank improve the protected32 plateau
when stable clipping removes the diagnosed norm-overflow barrier?

Protected INOUT ranks16/32/64, CE, Adam3e-4, stable clip1, unchanged EXP072-v2
seeds123/456. Primary rank64; control rank32; rank16 exploratory economy arm.
Horizons4,096/8,192/12,288;73,728 total steps,18,874,368 input tokens.
Gate: all six complete; primary final Wiki test NLL at least0.1 below rank32
and below its unchanged start at each seed. PG19 confirmation separately reported.
Rank16 is not promoted to primary after seeing outcomes. Record parameter/memory
counts and elapsed cost; equal steps are not equal parameters or FLOPs.
EXP096's old rank64 failures did not identify their numerical cause; this is a
new controlled combination, not proof that stable clipping will cure those runs.

Shared source/data/evaluation/resume/budget contract and ready Kaggle cell:
[three-account protocol](EXP-098-100-three-account-protocol.md).
Implementation: `scripts/m3q_protected_capacity_campaign.py`, shared safe engine.
Default8h includes45min reserve; rank64 runtime/HBM is not yet measured on TPU.

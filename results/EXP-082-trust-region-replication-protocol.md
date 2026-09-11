# EXP-082 — multi-split trust-region replication (pre-registered)

## Question

Does EXP-081's assembled-model trust-region improvement reproduce across fresh data, or was the small two-seed result caused by one favorable proposal/calibration/test slice?

## Frozen method

EXP-082 repeats the complete EXP-081 protocol without changing its alpha grid, 0.1% relative calibration-KL acceptance threshold, optimizer, 2,048 proposal steps, model architecture, source endpoints, or forward coordinate order. `TRUST-LINE` remains primary and `HARD-ACCEPT` remains the matched control.

## Replications

- Eight pre-registered data replications, each containing both source model seeds 123/456 and both arms.
- Each replication uses a distinct 512-window length-128 proposal slice and an immediately following disjoint 32-window length-128 calibration slice from WikiText train.
- Proposal offsets are `1,642,496 + 69,632*r` for replication `r=0..7`; calibration begins 65,536 tokens later.
- Locked evaluation is cross-corpus: the pinned `deepmind/pg19@4d28bd7` validation manifest and deterministic Google-hosted PG-19 book assets, 32 length-256 windows at offsets `8,192*r`.
- The final validation examples never choose an alpha or checkpoint. Per-window NLL is retained for audit.

The replications vary data while sharing the same two EXP-072 initialization seeds. They are independent data replications, not 16 independent model initializations. Runtime token hashes freeze the exact downloaded book content.

## Registered aggregate gate

For each source seed separately:

1. all eight replications must complete;
2. mean replication-level `TRUST-LINE final - start` NLL must be below zero;
3. the upper normal 95% bound over the eight replication means must be below zero;
4. at least six of eight replication deltas must be negative;
5. mean `TRUST-LINE final - HARD-ACCEPT final` must be below zero;
6. every TRUST-LINE milestone must remain within `1.25x` its own start; and
7. at least one nonzero coordinate must be accepted in every replication.

Both source seeds must satisfy every condition. Window-level deltas and accepted-layer frequencies are descriptive secondary evidence; the primary uncertainty unit is the data replication, not an individual correlated token window.

## Compute and durability

The campaign contains eight full EXP-081 replications: 1,572,864 local proposal updates plus assembled-model alpha evaluations. Based on measured EXP-081 runtime, expected TPU v5e-8 runtime is approximately 5–7 hours.

Every replication has its own immutable HF namespace and four six-layer branch checkpoints. The outer campaign uploads a combined snapshot after each completed replication. A rerun skips completed replications and resumes the active one from its last milestone. Checkpoint synchronization retries transient errors for 52 seconds, then records a non-fatal durability warning so a prolonged Hub outage cannot consume the TPU session. Inner Telegram notifications are disabled; the outer campaign sends one start and one terminal notification.

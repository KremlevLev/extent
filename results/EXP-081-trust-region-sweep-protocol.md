# EXP-081 — assembled-model trust-region coordinate recovery (pre-registered)

## Question

EXP-080 showed that a locally improved Mamba block can damage the assembled model. Can a held-out, teacher-guided acceptance rule prevent this composition failure and turn local conditional recovery into a full-model improvement?

## Starting point

- The same complete Qwen3-1.7B EXP-072 `ONPOLICY` endpoints used by EXP-080, seeds 123 and 456.
- Four retained source GQA layers at `6/13/20/27`; all 24 Mamba replacements begin identically in both arms.
- Forward causal coordinate order only. EXP-080 already rejected reverse order as a solution.

## Arms

- `HARD-ACCEPT`: train a proposed replacement for the current layer and compare only alpha `0` (unchanged) with alpha `1` (complete proposal).
- `TRUST-LINE` (registered primary): compare alpha `0, 0.125, 0.25, 0.5, 0.75, 1.0` along the line from the current block to the locally trained proposal.

The score is full assembled-model prediction KL against frozen Qwen on 32 length-128 calibration windows. A nonzero alpha is accepted only when it lowers calibration KL by at least 0.1% relative to alpha zero. Including alpha zero makes rejection explicit; no candidate is selected using final test NLL.

## Data and training

- Local proposal: 2,048 BF16 Lion updates per coordinate, LR `3e-5`, warmup 64, cosine decay, no decay, clip 1.0.
- Proposal data: 512 length-128 WikiText train windows at offset 1,572,864.
- Acceptance data: disjoint 32 length-128 WikiText train windows at offset 1,638,400.
- Locked evaluation: 32 length-256 WikiText test windows at offset 196,608, evaluated only at 0/6/12/18/24 processed layers.
- Two seeds and two matched arms: 196,608 local updates total. Seed 456 reverses arm execution order, not layer order.

## Gate

At both seeds, `TRUST-LINE` must finish below its own step-zero locked-test NLL, never exceed `1.25x` its start at a registered milestone, and beat `HARD-ACCEPT` at the final milestone. It must accept at least one nonzero update per seed. Selection uses calibration KL only; locked-test NLL cannot choose alpha or checkpoints.

A pass establishes assembled-model trust-region acceptance as an M3Q composition method. A stable but non-improving result shows that rejection prevents damage but cannot recover quality. A calibration improvement with test degradation diagnoses selection overfit and requires a larger or different acceptance signal.

## Durability and runtime

The campaign stores one complete branch snapshot after every six coordinates, reducing Hub commits from EXP-080's 96 per-layer endpoints to at most 16 milestone checkpoints. A failed Hub synchronization is recorded but does not abort valid TPU computation. Resuming may repeat at most five completed local coordinates. The default soft wall budget is 7.5 hours with a 20-minute shutdown reserve.

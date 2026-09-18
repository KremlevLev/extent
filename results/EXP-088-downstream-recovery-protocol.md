# EXP-088 — downstream-sensitive pair recovery

Pre-registered before scientific outcomes. One entry point runs the fixed
three-update engineering preflight and immediately enters scientific training on
success, with an 8.1-hour total soft wall budget and finalization reserve.

## Comparison

Pinned Qwen3-1.7B-Base and unchanged EXP-072 ONPOLICY endpoints, source seeds
123/456, all 24 Mamba coordinates, four fresh data replications.

- `LOCAL-PAIR-MSE`: jointly train the composed pair against the frozen conditional
  Qwen segment output, subtracting residual identity (EXP-087 method).
- `DOWNSTREAM-KL`: cache hybrid-prefix states, train only that pair through the
  frozen assembled suffix, against full natural-text Qwen teacher predictions.
  Exact full-vocabulary next-token forward KL, temperature 1, no CE term.

Both use 512 updates per trainable subtree, 64 length-64 training windows, BF16
Lion and the existing 0.1% two-domain 6x6 pair selector. Optimizer settings remain
lr 3e-5, warmup 64, cosine, no decay, clip 1.0. Computational cost is not matched:
downstream backward is larger. Proposal wall time is recorded and comparator
claims are equal-update rather than equal-FLOP.

Teacher train logits are cached offline in host RAM, approximately 2.32 GiB per
replication; they never coexist with a teacher backward or trainable teacher state.
Teacher weights still reside for target/prefix evaluation. This is not a claim of
teacher-free host/device memory.

Fresh reserved starts after EXP-087: WikiText-103 train 1,949,696; PG-19 validation
147,456; PG-19 locked test 229,376. Existing strides reserve disjoint ranges.
Full token hashes and endpoint hashes are resume invariants.

## Gate

Per seed: all four runs, negative primary mean final-minus-start NLL and upper
normal 95% bound, >=3/4 improving runs, negative mean primary-minus-control final
NLL, no registered excursion beyond 1.25x start, nonzero acceptance in >=3/4.
No locked-test metric chooses proposals, alphas or optimizer settings.

## Operations

Separate EXP-088 HF/RAM/output namespaces. Startup, post-preflight and terminal
session reports are written atomically and uploaded to HF. Scientific summaries
and registered six-coordinate milestone checkpoints use the established HF resume
path; active downstream scalar health is saved locally every 64 updates and is
included in terminal scientific reports. Caught startup errors receive a terminal
session report and Telegram notification. External VM termination cannot run
callbacks and can lose work since the latest durable milestone.

Anticipated runtime is 5-8 hours, not measured: the observed 0.422-second step was
synthetic context 32, whereas scientific training uses context 64, real endpoints,
and different suffix lengths. Deadline-partial completion may require continuation
but does not retrain completed replications. Do not allocate a separate session for
the engineering preflight.

A pass supports downstream-sensitive proposals over local segment proposals.
It does not establish better initialization than random, capability retention,
MLA quality, inference acceleration or 14B transfer.

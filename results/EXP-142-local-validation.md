# EXP142 local engineering validation (2026-10-17)

PREPARED; no cloud training and no new NLL result.

8 tests passed in51.55s on8 virtual CPU devices. Four runner/rate tests rechecked
on final cleaned code:4 passed in7.50s. Tests cover exact initial held-out function
parity with139 dense, every original base leaf in trainable q, FP32 masters,
nonzero gradients for all six groups, actual8-device compiled update and sharded
nested Adam, finite proposal, CE versus ordinary token NLL, checkpoint roundtrip
for parameters/Adam/best and identical next update after restoring prescribed
layouts, profiler agreement with changed learned embedding/MLP, actual runner
acceptance/rejection branches with mocked remote/device kernels, plan-only CLI,
9h TPU budget, registered per-group LR and real BF16 full-decoder gradients.

One initial next-step test compared an automatically-sharded live output against
unsharded restored host arrays and saw tiny FP32 reduction differences. Both
paths now use the explicit layouts enforced by the real runner; all payload
leaves restore exactly and the next proposals agree exactly in this CPU test.
Do not infer cross-runtime/physical TPU bitwise equivalence from that test.

Abstract1.7B init shapes (jax.eval_shape, no full parameter allocations/download):
base1,675,304,064; added pregate36,864; total1,675,340,928 trainable parameters.
FP32 masters6,701,363,712 bytes; q+Adam(mu,nu)+best26,805,454,848 bytes plus
small optimizer counters/metadata. These are checkpoint arithmetic, not measured
peak HBM/disk. Generation replacement can temporarily need two local payloads.

All7 old engineering contracts135-141 compared equal after adding142. Standalone
launch cell parsed with AST; frozen142 plan regenerated from current code/source
hashes. Old runners/math were not edited. No HF writes/deletions or cloud runs.
Physical full model compile, HBM, disk capacity, source-window replay, finite
first gradient, throughput, final persistence and model quality await the user run.

# EXP108–110 engineering checks, 10 October2026

Scope: local Python3.12.14/JAX0.6.2, CPU with eight virtual devices.
No full Qwen model allocation, TPU training, live HF uploads or Telegram sends.

- Pinned real tokenizer/datasets loaded from cache. Capacity and exact SHA
  verified for new65536x256 train, reused Wiki validation/test and new128x256
  PG19 slice; EXP-108-110-data-preflight.json. This is real data evidence.
- tests/test_batch_recovery.py: **6 passed in86.48s**, final revision.
  Unclipped mean equals direct mean-loss gradient; every microbatch forward
  finite guard; three tiny sharded modes; one Adam update per accumulated
  batch; frozen vocabulary; binary checkpoint restores exactly the next
  weights/state/diagnostics; consumed-window cursor and update counters;
  schedule/budgets and cross-account gate/contract rejection.
- Existing tests/test_layer_clip_recovery.py: all15 passed in the earlier
  combined run. No old runner/core edits; measured106/107 guards remain intact.
- All three actual notebook entry points passed plan-only with8.8h/90min.
  Prepared contracts archived separately. Memory policy extends prior abstract
  production estimate by one FP32 carry for batch4:12,321,462,528 baseline,
  13,003,959,040 bytes/device with carry. These are policy estimates, not measured
  TPU allocator/compiler memory. Runner additionally measures those on TPU.
- git diff --cached --check passed. No costly cloud run launched.

Full scientific result remains pending, including throughput, feasibility and
matched final PG19/Wiki comparison of both seeds in all three accounts.

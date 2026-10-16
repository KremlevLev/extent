# EXP130–134 local engineering validation

16 October2026. PREPARED, not a physical1.7B GPU/TPU result.

Full new suite before final explicit FP32-angle cast:
pytest tests/test_competitor_recovery.py tests/test_competitor_tpu.py -q -x
with JAX_PLATFORMS=cpu,XLA_FLAGS=--xla_force_host_platform_device_count=8:
25 passed in244.82s.

After matching the inspected FP32-before-tanh convention:
pytest tests/test_competitor_recovery.py -q -x -k "bf16 or tanh_angle":
5 passed,14 deselected in30.28s (four BF16 regressions plus new dtype check).
26 unique checked cases across these invocations.

Coverage:

- All11 recipes: streamed CE/KL loss and gradients versus full-chain monolithic
  rematerialized/tiled TPU objective; actual optimizer update and exact next
  update after serialization. Unlocked dynamics receive nonzero gradients.
- BF16 source identity/finite full-strength gradients for reset, unlocked+MLP,
  KL and dense Mamba. Pre-recurrence tanh explicitly computed in FP32.
- Dynamics reset zero columns, dt target, simplex sum; headwise norm placement
  and recurrent/dual math additionally covered by unchanged129 checks.
- Real compiled eight-virtual-device updates:133 control and MLP,134 CE andKL,
  including production1024-token KD batch. Teacher hidden parity versus actual
  Qwen module. Exact ChunkedCheckpointStore restore and next update on8 shards.
- Nested Adam mu/nu arrays checked non-replicated for matrix parameters; empty
  coordinate dicts retained. Candidate finite health evaluated before acceptance.
- Missing interrupted endpoint retried; future metrics invalidated on rollback.
  Failed checkpoint save retried; same-cursor changed best forces fresh save.
- T4 runner clean deadline/resume: durable2→4 accepted updates, accurate teacher
  input token count on KL arm. No physical cloud execution implied.
- Long contiguous train-pool permutation visits4096 blocks once;1023 targets,
  and133 starts permutation cursor4096. Actual data hashes checked in loader.
- All5 plan-only calls passed without weights. All10 frozen125–129 worker
  contracts compare equal to committed plans.124 scientific code untouched.
- REAL126 binary retrieval: pinned manifest +best_step4096,96 best leaves,
  7,145,472 finite coordinates,all chunkSHA checks. Parameter tree hash EXACTLY
  matches the recorded selected-best hash63f995c88be4cf0b05d4293cfff826a6ec2513d3457b1d03105fa458ad8e307d.
  Receipt EXP-133-source-validation.json. No optimizer-at4096 claim.
- Real abstract1.7B geometry:134 trainable264,102,016 coordinates,
  payload estimate4.225632256GB/arm,8.451264512GB/paired save.

Unverified: physical1.7B TPU source/candidate parity (gated at runtime), T4/TPU
HBM plans,actual throughput and achieved updates,long-run optimization,HF
concurrent quota/transport and quality/long-context improvements. No new GPU/TPU
training launched by these local checks. Local source download is not a replay
of full-model NLL. No novelty,replication or total efficiency claim.

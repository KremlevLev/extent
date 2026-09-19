# EXP-089 — long-horizon full-model recoverability

## Question

The local exact MIMO lift frequently beats random initialization, while the short 3,072-step whole-model EXP-064 comparison strongly favored random. EXP-089 tests H0.1 directly: does exact lift provide a delayed optimization advantage that appears only after meaningful full-model recovery, or is its transferred prior persistently harmful after assembly?

## Registered design

- Source: pinned `Qwen/Qwen3-1.7B-Base`.
- Student: 24 Mamba-3 MIMO layers and four unchanged Qwen GQA layers at `6,13,20,27`. MLA remains excluded.
- Arms: canonical random Mamba mixers versus exact MIMO QKVO lift. Direct Qwen weights and retained GQA are identical.
- Seeds: `123` and `456`, with counterbalanced execution order.
- Training: 98,304 full-model updates, context 256, or 25,165,824 tokens per trajectory. Across four trajectories the registered campaign processes 100,663,296 student tokens.
- Objective: online frozen-Qwen forward KL at temperature 2 plus true-token cross entropy weight 0.1.
- Optimizer: BF16 Lion, learning rate `3e-5`, 512 warmup updates, no weight decay, FP32 global clipping at 1.0.
- Data: pinned WikiText-103 train slice from token offset 5,242,880. Held-out evaluation uses 32 WikiText-103 validation windows.
- Checkpoints: `0, 3,072, 8,192, 16,384, 32,768, 65,536, 98,304`.

The primary endpoint requires exact lift to beat random in both excess NLL and prediction KL at step 98,304 for both paired seeds. The complete curves and first crossover checkpoints are retained even if the endpoint gate fails.

## Session and recovery contract

This is intentionally not a one-session experiment. A Kaggle invocation uses a soft 8.35-hour budget, reserves 35 minutes for final synchronization, and saves the complete student parameters, Lion state, step cursor, metrics, and immutable contract every 8,192 updates and at the session boundary. Checkpoints use stable HF paths, so replacement uploads do not intentionally accumulate every historical multi-gigabyte state. Rerunning the same entry point restores the last verified state and continues it.

The compact result and summary are uploaded after each durable state. Any non-finite update preserves the preceding checkpoint. The campaign must not be restarted with a changed contract under the same HF namespace.

## Interpretation boundary

A win would establish only that exact lift improves long-horizon recoverability relative to random for the 1.7B GQA hybrid. A failure would close the current exact-QKVO transplant as a practical initializer and redirect the paper toward compositional failure or a genuinely different bridge such as explicit normalized linear attention. Neither outcome establishes MLA or 14B quality.

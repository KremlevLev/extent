# EXP-077 — protected AdamW recovery (pre-registered)

## Question

Does protected joint recovery fail because Lion's sign-style update is unsuitable for the highly uneven full-depth Mamba gradients? EXP-077 restores the best tested global objective and changes the optimizer family.

## Fixed design

- Qwen3-1.7B hybrid and the exact completed EXP-072 `ONPOLICY` endpoints for seeds 123 and 456.
- All and only the 24 Mamba subtrees train. Copied Qwen parameters and normalization scales remain frozen.
- Exact EXP-075 training/evaluation tokens, context 256, 8,192 updates, checkpoints `0/1024/2048/4096/6144/8192`.
- Teacher forward KL at temperature 2 plus causal cross entropy weight 0.1; no hidden loss.
- AdamW beta1 0.9, beta2 0.999, BF16 gradients and moment buffers, no weight decay, global FP32-stable clipping at 0.3, 512-step warmup and cosine decay to 0.1 times peak LR.

## Arms and controls

- `ADAMW-3E-6` (registered primary): peak LR `3e-6`. Against EXP-075 `MAMBA-ONLY`, this preserves source, trainable mask, data, objective, update count, schedule shape, clipping and peak LR; optimizer family is the intended difference.
- `ADAMW-1E-5` (exploratory calibration): peak LR `1e-5`. This tests whether the AdamW-specific useful scale is above the matched numerical LR and is not an isolated optimizer-only contrast.

EXP-075 Lion is a byte-identified historical control and is not rerun. EXP-076 shows hidden-only objectives are less stable and is not repeated.

## Gate and boundary

At both seeds, primary final held-out NLL must beat its own step-zero NLL and no registered checkpoint may exceed 1.25 times its starting NLL. A secondary-only pass requires locked confirmation at the selected LR.

A primary pass attributes stabilization to AdamW under the matched contract. A failure of both arms motivates a bounded/trust-region recovery method rather than further ordinary LR or objective sweeps. The experiment does not yet compare exact lift against random under equal stable recovery compute.

Expected TPU v5e-8 runtime is approximately 4–6 hours. All branches are independently checkpointed and resumable through the configured private Hugging Face artifact dataset.

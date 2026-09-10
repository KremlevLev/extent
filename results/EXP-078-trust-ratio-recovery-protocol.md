# EXP-078 — protected trust-ratio recovery (pre-registered)

## Question

Can per-tensor relative-step control turn EXP-077's near-stable AdamW trajectory into genuine improvement? Mamba contains projection matrices, recurrent scalars and biases with very different norms and gradient scales; one global normalized step need not be safe for all of them.

## Fixed design

- Exact completed EXP-072 `ONPOLICY` endpoints for Qwen3-1.7B seeds 123 and 456.
- All and only the 24 Mamba subtrees train; every copied Qwen leaf remains frozen.
- Exact EXP-075/077 train and evaluation tokens, context 256, 8,192 updates, checkpoints `0/1024/2048/4096/6144/8192`.
- Teacher forward KL at temperature 2 plus causal cross entropy weight 0.1; no hidden objective.
- BF16 gradients and BF16 Adam first/second moments, beta1 0.9, beta2 0.999, no weight decay, global FP32-stable gradient clip 0.3, 512-step warmup and cosine decay to 0.1 times peak.

## Intervention and arms

After Adam normalization, every parameter leaf's direction is rescaled by `||parameter|| / ||direction||` before the scheduled scalar step. A `1e-6` norm floor prevents zero/tiny recurrent leaves from escaping the bound. This is the LAMB trust-ratio rule without weight decay.

- `TRUST-1E-4` (registered primary): peak relative step `1e-4`.
- `TRUST-3E-5` (conservative control): peak relative step `3e-5`.

EXP-077 AdamW and EXP-075 Lion are historical controls with identical source, mask, data, objective and update count; they are not rerun.

## Gate and boundary

At both seeds, primary final held-out NLL must beat its own step-zero NLL and no registered checkpoint may exceed 1.25 times start. A secondary-only pass requires locked confirmation.

A pass supports per-tensor relative displacement as the missing stabilization mechanism and yields a candidate shared optimizer for the later exact-lift-versus-random comparison. Failure means joint updating of all 24 Mamba blocks remains too coupled even under relative steps; the next intervention should impose depth/stage boundaries rather than another global optimizer sweep.

Expected TPU v5e-8 runtime is approximately 4–6 hours. Every branch is independently checkpointed and resumable through the configured private Hugging Face artifact dataset.

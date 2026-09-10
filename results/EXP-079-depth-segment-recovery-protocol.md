# EXP-079 — attention-bounded depth segment recovery (pre-registered)

## Question

Which part of the 24-layer Mamba stack can benefit from global prediction recovery without simultaneous movement in the other replaced blocks? The retained GQA layers divide the hybrid into four natural runs of six Mamba layers.

## Fixed design

- Qwen3-1.7B hybrid initialized from exact completed EXP-072 `ONPOLICY` endpoints for seeds 123 and 456.
- Four arms train Mamba layers `(0-5)`, `(7-12)`, `(14-19)`, or `(21-26)` respectively. Every other Mamba subtree and every Qwen parameter remains frozen.
- Teacher forward KL at temperature 2 plus causal cross entropy weight 0.1; no hidden objective.
- The EXP-078 conservative BF16 LAMB rule: trust-ratio relative step `3e-5`, beta1 0.9, beta2 0.999, zero weight decay, clip 0.3, 384-step warmup and cosine end at `3e-6`.
- Exact prior train/evaluation slices; context 256; 6,144 updates; checkpoints `0/1024/2048/4096/6144`.

All arms have the same number of trainable Mamba layers and identical full-model forward/backward counts. Compared with full-depth EXP-078, each leaf receives a different exposure and parameter count is deliberately restricted; this is a depth-causality experiment, not a compute-identical claim against EXP-078.

## Registration

`SEGMENT-4` (layers 21-26) is primary because it is closest to the LM head and changes no downstream Mamba input distribution before retained attention layer 27. Segments 1-3 form a pre-registered depth atlas; arm order reverses for seed 456.

The primary gate requires final NLL below step zero at both seeds and no registered checkpoint above 1.25 times start. Secondary segment results are descriptive and require locked confirmation before selection.

If one or more segments recover consistently, their ranking determines the order of a later multi-stage coordinate-recovery experiment. If every isolated segment fails, the full-model prediction target itself is insufficient even after eliminating simultaneous depth coupling.

This campaign performs 49,152 optimizer updates and should require roughly 5.5–7 hours on TPU v5e-8. Each arm/seed is independently checkpointed and resumable through the private Hugging Face artifact dataset.

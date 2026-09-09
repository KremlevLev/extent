# EXP-074 — stabilized joint recovery (pre-registered)

Date: 2026-09-09. Status: implementation prepared; no result observed.

## Question

EXP-073 proves that the EXP-072 ONPOLICY warm start remains much better than clean-input recovery after joint optimization, but its constant `3e-5` schedule produces extreme NLL spikes and a seed-dependent final collapse. EXP-074 asks whether a fixed lower-step recipe makes full-model recovery stable without selecting an evaluation checkpoint after the fact.

## Frozen design

- Qwen3-1.7B 24-Mamba/4-GQA hybrid, EXP-072 ONPOLICY endpoints, seeds 123/456.
- Two arms start from byte-identical endpoints and use the exact EXP-073 train/evaluation tokens, objective, 8,192 steps, context 256 and all-parameter training.
- Registered primary `LR3E-6`: peak Lion learning rate `3e-6`. Exploratory conservative control `LR1E-6`: peak `1e-6`.
- Both use 512 warmup steps, cosine decay to one tenth of peak, zero weight decay and FP32-stable global clipping at 0.3. Thus the experiment evaluates a pre-declared stability recipe bundle; it does not attribute an effect separately to warmup, clipping or LR.
- Evaluate only at the fixed checkpoints `0/1024/2048/4096/6144/8192`. Primary success requires, at both seeds, final NLL below that arm's own step-zero NLL and maximum registered-checkpoint NLL no larger than `1.25 ×` step-zero. The `LR1E-6` arm is secondary and requires a later confirmation if it alone passes.

The unchanged EXP-073 ONPOLICY trajectories are an exact historical control because source endpoint hashes, data and objective are identical. They are not rerun, avoiding another two unstable 8,192-step branches. EXP-074 is not yet a compute-normalized random-initialization comparison.

## Durability and expected runtime

Each arm/seed stores full weights, BF16 Lion state, data cursor and endpoint hashes in the private HF dataset. Rerunning the same cell resumes exact state. Four branches should require roughly four hours based on EXP-073, within the eight-hour soft limit.

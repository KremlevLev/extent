# EXP-076 — protected objective bridge (pre-registered)

## Question

Does EXP-075 fail because a single output-distribution loss sends a badly conditioned global signal through 24 jointly replaced Mamba blocks? EXP-076 keeps the successful protection boundary fixed and changes only the recovery target.

## Fixed design

- Source: the exact completed EXP-072 `ONPOLICY` endpoints for seeds 123 and 456.
- Model/data: Qwen3-1.7B hybrid, 24 Mamba replacements, the exact EXP-073/074/075 train and held-out token slices.
- Training: 8,192 updates per arm/seed; context 256; BF16 Lion; peak LR `3e-6`; 512-step warmup; cosine end LR `3e-7`; no weight decay; clip 0.3.
- Trainable parameters: all and only Mamba subtree leaves. Every copied Qwen parameter and norm remains frozen.
- Evaluation: held-out language-model NLL/KL at steps `0/1024/2048/4096/6144/8192`. Hidden losses are training targets, not the scientific endpoint.

## Arms

- `DELTA-BRIDGE` (registered primary): for each of all 24 replaced positions, match the decoder-block contribution `h_l - h_(l-1)` to Qwen. Layer zero uses its output state because the embedding state is not exposed.
- `STATE-BRIDGE` (objective control): match the accumulated hidden state `h_l` at the same 24 positions.

Both arms use hidden alignment only. Prediction KL and causal cross entropy have zero training weight. EXP-075 `MAMBA-ONLY` is the exact historical output-KL/CE control and is not rerun.

## Gate and interpretation

At both seeds, `DELTA-BRIDGE` final held-out NLL must beat its own step-zero NLL, and no fixed checkpoint may exceed 1.25 times step-zero NLL. Best-checkpoint selection cannot pass the gate. `STATE-BRIDGE` is secondary and cannot substitute for a primary failure without locked confirmation.

A pass would support the claim that preserving local block contributions is the missing bridge between successful sequential transplantation and stable joint recovery. A failure would rule out the current hidden-bridge objective at this optimizer scale; the next isolated factor would be Lion versus a magnitude-sensitive optimizer. It would not falsify the exact-lift initialization itself.

Expected TPU v5e-8 runtime is roughly 5–7 hours because both student and teacher expose all decoder hidden states. The estimate is operational, not evidence. State, summaries, and failure records are resumable through the configured private Hugging Face artifact dataset.

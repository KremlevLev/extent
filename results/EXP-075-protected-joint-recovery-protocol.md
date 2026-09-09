# EXP-075 — protected joint recovery (pre-registered)

Date: 2026-09-09. Status: implementation prepared; no result observed.

## Question

EXP-074 shows that reducing the global Lion LR from `3e-5` to `3e-6/1e-6` does not prevent degradation when every copied Qwen parameter trains. EXP-075 tests the causal alternative: preserve the pretrained Qwen backbone exactly while optimizing the 24 Mamba replacements together under the full language-model objective.

## Frozen design

- Same Qwen3-1.7B hybrid, EXP-072 ONPOLICY endpoint hashes, seeds, train/evaluation tokens, objective, 8,192 steps and fixed evaluation checkpoints as EXP-073/074.
- Both arms use BF16 Lion with peak LR `3e-6`, warmup 512, cosine decay, zero weight decay and global clipping 0.3.
- Registered primary `MAMBA-ONLY`: gradients are zeroed for every parameter outside a `mamba` subtree. Copied embeddings, GQA, MLPs, all norms and LM head remain byte-stable.
- Exploratory `MAMBA-NORMS`: trains the same Mamba parameters plus all copied normalization `scale` parameters. Embeddings, GQA projections, MLP matrices and LM head remain frozen.
- Primary gate: at both seeds, final NLL below its own step-zero value and no registered checkpoint above `1.25×` step-zero. `MAMBA-NORMS` requires later confirmation if it alone passes.

The exact EXP-074 `LR3E-6` all-parameter trajectories are the causal historical control: same starting endpoint, LR, warmup, clipping, data, objective and updates; only the static trainable mask differs. This experiment does not yet compare total compute against random initialization.

## Runtime and durability

Four branches should take roughly four TPU hours. Full states still include optimizer slots for structural simplicity, while the static gradient mask guarantees frozen leaves receive zero updates. Checkpoints and exact cursors are uploaded to the private HF dataset and resume across sessions.

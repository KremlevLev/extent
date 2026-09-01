# EXP-068 — long-horizon compatibility confirmation (partial)

- Deadline-partial: `12/28` layers completed in `4.232` TPU hours.
- All completed trajectories are finite and synchronized to the private HF dataset.
- On the 12 shared layers, EXP-067 step-2,048 versus EXP-068 step-8,192 difficulty ranking has Spearman `0.979021`.
- Within EXP-068, step-1,024 versus step-8,192 ranking has Spearman `1.0`.
- No retained-layer overlap can be computed until all 28 layers complete. The raw partial artifact's `0/4` display is a presentation artifact caused by suppressing selection on incomplete atlases, not a measured disagreement.
- Long-horizon dual gains replicate at boundary/deep layers; it improves over random at 0, 1, 17, 20, 22, 24, 26, and 27, while losing at 3, 6, 10, and 13.
- Rerunning the identical EXP-068 entry point restores the 12 completed layers from HF and continues with layer 5.

Artifacts supplied by the user:

- Full SHA-256: `8c6d4df2b247dacb59ca8bfa5c06ce16fe4cc5bdffb1f121b5c001272309f5b4`
- Summary SHA-256: `d64ef6271ec9b208215a728ca8ec9f11203de6fbdeb69cad683c4dbb067686ef`

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

## Second resumable execution

- Cumulative completion increased from `12/28` to `23/28`; the second invocation took `4.172` hours and restored the first 12 layers from HF.
- The newly completed layers are `5,15,8,19,12,25,2,23,4,21,7`.
- EXP-067 versus EXP-068 final ranking Spearman increased to `0.987154`; EXP-068 step-1,024 versus step-8,192 Spearman is `0.999012`.
- Remaining layers are `18,9,16,11,14`; selection and the registered gate remain pending until they complete.
- Second full SHA-256: `596ca6e20764a874393e0b49cf359a8c20f8cb85ef052fe562ab1ce01e5936b6`.
- Second summary SHA-256: `a85f84c64d5c5a349fd2d3e0d03eaf11cb81d2b8713bb4dc67b9ef94432a2ff3`.

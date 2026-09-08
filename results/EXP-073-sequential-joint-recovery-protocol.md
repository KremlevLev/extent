# EXP-073 — sequential warm-start joint recovery (pre-registered)

Date: 2026-09-08. Status: implementation prepared; no result observed.

## Question

EXP-071 and the partial EXP-072 show that training each replacement on its deployed hybrid residual stream prevents the depth collapse caused by clean-Qwen-only layerwise recovery. EXP-073 tests the publication-critical next link: does that advantage survive when the completed 24-Mamba hybrid is subsequently trained end to end?

## Frozen comparison

- Qwen3-1.7B-Base 15/85 proxy with retained source GQA at layers `[6,13,20,27]`.
- Seeds 123/456 and arms `TEACHER`/`ONPOLICY`. Each arm restores all 24 matching completed EXP-072 v2 endpoints. The local-recovery update count is therefore equal; only the residual-stream distribution differs.
- Every student parameter then receives 8,192 identical BF16 Lion updates: context 256, LR `3e-5`, warmup 128, no decay, FP32-stable clipping at 1.0, teacher forward KL at temperature 2 plus 0.1 causal cross-entropy.
- Training samples 4,096 fixed WikiText train windows from offset 1,310,720. Evaluation uses 32 untouched test windows from offset 131,072. Token bytes and all 24 input endpoint hashes enter each resume contract.
- Evaluate at steps `0/1024/2048/4096/6144/8192`. The primary gate requires `ONPOLICY` to beat matched `TEACHER` in final NLL and normalized NLL-delta AUC at both seeds.

This isolates persistence of the sequential-recovery advantage under joint optimization. It does not yet compare total pretraining compute against a from-random Mamba model, convert retained GQA to MLA, establish 14B transfer, or measure long-context inference. A pass advances the method to a separately compute-accounted random-initialization comparison.

## Runtime and recovery

The expected joint phase is roughly 4–6 TPU hours based on earlier full-model runs. Four large model/Lion states are saved independently to the private HF dataset; interruptions resume parameters, optimizer and step together. The shared uploader now tolerates more than seven minutes of transient HF outage. EXP-073 refuses to start unless every registered EXP-072 endpoint exists and matches its base hash.

The chain entry point first finishes EXP-072, clears compilation caches, and immediately begins EXP-073 in the same TPU session.

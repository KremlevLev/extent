# EXP-090 — long-horizon Mamba-3 timestep stabilization

## Question

Mamba-3's learned timestep simultaneously changes recurrent decay, complex phase accumulation, and input injection magnitude. EXP-090 tests H1.2 at full-model scale: does preventing this coupled degree of freedom from moving during early recovery make a randomly initialized Mamba-3 hybrid easier to train?

## Registered comparison

Both arms use the same canonical random Mamba initialization, direct Qwen weights, retained GQA layers, tokens, optimizer, and full-model teacher objective. `RANDOM` trains every parameter normally. `RANDOM-DT-FROZEN` zeros only the raw-dt projection columns and `dt_bias` gradients in all 24 Mamba layers for the first 32,768 updates (8,388,608 tokens), then unfreezes them for the remaining 65,536 updates (16,777,216 tokens). Decay, phase, trapezoid, B/C, values, gates, readouts, MLPs, norms, embeddings, and retained attention remain trainable throughout.

Two paired seeds receive 98,304 context-256 updates each: 25,165,824 tokens per trajectory and 100,663,296 registered student tokens overall. Evaluation checkpoints are `0, 3,072, 8,192, 16,384, 32,768, 65,536, 98,304` on 32 pinned WikiText-103 validation windows. The primary arm must beat ordinary random in both final excess NLL and prediction KL at both seeds.

## Operational contract

EXP-090 uses a separate HF namespace from EXP-089 and can run concurrently on another account. Each invocation uses up to 8.35 hours, reserves 35 minutes for synchronization, and saves complete parameters, Lion state, cursor, and metrics every 8,192 steps and at the session boundary. Rerunning the identical entry point resumes the verified state.

This experiment diagnoses timestep coupling; it is not a new weight-transplant claim. A pass supplies a stabilization component for a future bridge. A failure rejects early hard freezing at this duration and prevents further full-model spending on the same intervention.

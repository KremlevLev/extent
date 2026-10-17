# T4 streamed-v2 fix: evidence and validation

13 tests passed in85.70s. Previous8-case pass81.34s covered all seven numerical
recipes; final suite additionally covers real parent log surfacing, memory guard
failure with byte counts, orchestration source Adam/cursor/finite acceptance, and
BF16135 legacy-gradient replay. CPU tiny models only, two virtual devices for135.
Full compiled versus streamed FP32 gradients: bounded tolerances rtol.002/atol2e-5,
loss rtol/atol2e-5, proposed params rtol.0002/atol2e-6. BF16 actual replay uses same
registered CE.03, relativeL2.15, cosine.98 bounds as135, with no accepted update before
that check. Model/data/optimizer algebra unchanged; GPU trajectories not claimed bitwise.

Old135 source A/B/C replay succeeded on physical2xT4 BEFORE failed backward compile:
A6.74723 B6.77329 C6.48021, largest per-window error.00385809<.03. New accepted updates0.
Both136 workers stopped at0 with the same guard error. See archived exact JSON and
HF snapshot receipt.137/138 root errors remain unknown from that snapshot.

The guard exception was raised before the executable ran; it is not evidence of
NaN gradients, actual CUDA allocator OOM, or a scientific failure. Old helper saved
no memory-analysis numbers. New execution bounds backward scope to one layer,
keeps full chain rule via saved host activations and reverse VJP, retains memory
checks and logs argument/temp/output/alias/limit/in-use bytes on rejection.
Gradient/Adam correction trees stay small; frozen base stays resident as before.
Teacher KL uses layer-streamed forward; head loss remains32-token tiles.
No process for135 computes a second independent seed or increases batch size.

All old135-142 contracts compare equal to saved engineering plans. Only new GPU
modules and launch docs added/changed. Separate namespace avoids silently migrating
v1 results. Code/source/protocol plan and four launch ASTs validated. No HF writes,
remote deletion, Kaggle launches, or new model-quality results. New physical GPU
memory, runtime and speed await user launch. First1358 accepted steps is the actual
hardware acceptance signal; do not repeat all4 long cloud submissions blindly.

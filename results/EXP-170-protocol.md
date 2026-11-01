# EXP170: clean head-preserving preparation

DESIGN plus tested initializer/alignment primitives. NOT launch-ready. See
EXP-170-clean-start-audit.md. Existing169 and historical code remain unchanged.

## Scientific question

Can a clean teacher-head preparation produce a better composed hybrid before
repeated end-to-end repair? This is a bundled recipe test, not isolation of each
ingredient or a claim of exact reproduction of arianraje's4B model.

## Immutable source and geometry

Original Qwen/Qwen3-1.7B-Base at
ea980cb0a6c2ae4b936e82123acc929f1cec04c1. Original frozen embeddings, final
norm/tied head, residual norms, MLPs and retained GQA. Fresh Mamba from original
Q/K/V/O and Q/K norms. No EXP069/072/111 weights, adapters, folded states or Adam;
do not instantiate legacy Patient for this experiment.

H16,P128,N128,G8,R4,inner2048,RoPE fraction.5. Full pre-gate norm and tanh from
step0; zero dynamic rows remain trainable. Bias1, random-simplex input factors,
one-hot own-head rank selection. Two queries per KV group explicitly adapted to
four rank slots. Random Z is a declared unresolved-surgery choice.

Train a bank of27 independent replacement blocks (indices0..26), using a single
original-teacher forward per batch. Every layer receives original teacher inputs
and original teacher output targets. Original residual/MLP math stays frozen and
differentiable wrt the replacement mixer. Train all mixer/norm parameters.

## Tokens and optimizer

First controlled test may reuse verified FineWeb-Edu600M pack, taking its first
100,007,936 tokens as4096-token sequences with EOS boundaries retained. This pure
Edu stream differs from his web/code/math mix and must be reported explicitly.
No cycling; validation excluded from packing. Discard engineering-pilot state
before the scientific start and charge its extra token presentations.

Target1526 updates x65,536 inputs =100,007,936 inputs;16 sequences4096 per
effective batch. AdamW(.9,.95),eps1e-8,weight decay0,stable GLOBAL clip1,
peakLR1e-3,warm46 updates (ceil3%),cosine to.1 peak. One update after weighted
full-batch gradient accumulation; no individual layer clock/LR resets.

Chosen alignment loss divides by frozen teacher target RMS per layer; record
both full-output relative MSE and mixer-only error to expose residual dominance.
The teacher sees100M input presentations;27 mixers see about2.7B layer-input
presentations. Do not divide100M by27 or call these27 independent text corpora.

Run an identically shaped random-mixer control on a short matched fresh prefix
(approximately2M), then extend the inherited bank to100M. A partial random
control does not establish equal-budget100M superiority. Register exact common
pilot endpoints before execution.

## Composition and measurements

Compose the SAME teacher-aligned bank with two retained-GQA masks:

* Existing4/28:6,13,20,27.
* 7/28 (75% replacements):3,7,11,15,19,23,27.

No second alignment bank is needed; train all27 candidate layers once. Compare
these masks separately, counting their different parameter counts and compute.
Geometry, retention, data and optimizer changes confound unique attribution.

At source and approximately1/2/5/10/25/50/100M: ordinary FULL-MODEL NLL; original
Qwen matched baseline; local layer error vs composed-prefix error; residual,
mixer and MLP scales; gradients/clipping and real update ratios. Keep existing
A-E panels/context256 separate from a standard WikiText2 protocol/context4096.
Never compare arianraje's4k PPL-derived NLL directly against unrelated A-E NLL.
Use a declared validation selector and retain last/final panels; local MSE alone
does not pass the composition test. NLL3 is not a cross-dataset launch gate.

## Integration required before launch

Implement clean original-Qwen loader without old preparation dependencies;
full-model forward using this layout; source hashes and frozen-parameter parity;
fused/BF16 numerical checks; fixed-batch weighted gradients; immutable source vs
own CURRENT; exact fresh-data cursor/optimizer restore; composition evaluations;
matched pilot separation; pruned checkpoints/HF durability and quiet notices.

Benchmark physical batches by+1 on the ACTUAL teacher+27-layer alignment workload.
Keep effective65,536 tokens; uneven final microbatches require actual-token
gradient weights, never padding/repetition or a silent optimizer-batch change.
Reuse isolated TPU probes, real per-chip memory accounting and phase timings.

100M/1675 inputs/s is16.6h;100M/6400 is4.34h BEFORE evaluation and saves. Those
rates came from a different model/workload and are not forecasts for this one.
Use caller-specified shared time limit and18min final reserve. Resume toward the
token target without repeats. Honor server disk limits and TPU verified-RAM-only
large artifacts. No accelerator run until these integrations pass their checks.

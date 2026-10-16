# EXP130–134: competitor-informed recovery suite

PREPARED,16 October2026. T4 sessions7/7/7h; TPU4/9h;129 separate10h T4.
One seed123 exploratory search, exposed panels. No new accelerator run locally.

## Reference and exact scope

arianraje/qwen3-4b-mamba3-hybrid-stage2b-kd-bias1, source revision
b946069890837bbe9b5dd75e2d118a7e98d2d193, Apache-2.0 metadata; modeling file SHA
6ca2cee4665ac88b28cb81470179f635287e983a28c4079afe2b085ceaf55f77.
Reference: https://huggingface.co/arianraje/qwen3-4b-mamba3-hybrid-stage2a-kd-bias1
Training summary: https://huggingface.co/arianraje/qwen3-4b-mamba3-hybrid-stage2a-kd-bias1/resolve/main/stage2a_kd_summary.json

Confirmed: internal per-head/per-rank RMSNorm before gate/reduction; tanh on raw
angle; zero scalar/dynamics rows at init; dt log-uniform.001–.1 via softplus
inverse; random simplex value factors; full-vocab forwardKL; mixer/rest LR
2e-4/2e-5; Adam(.9,.95), clip1, warmup fraction.03, minimum LR fraction.1.
Summary has no temperature knob or scheduler curve: plainT1 and cosine-to-.1
are explicit choices HERE, not recovered trainer source. No remote Python
execution or competitor weight downloads. Independent JAX formula implementation.
Pre-gate normalization is a standard upstream component, not a new invention.

Continuation of our1.7B patient (24/28 Mamba,head64,state64,groups1,expand.75,
MIMO4), NOT a replication of competitor4B geometry,27/36 retention,Q/K/V mapping,
own-rank selector,100M alignment/600M+294M KD, huge batches or4k/32k context.
Our geometry cannot inherit his Q/K/V mapping verbatim.130 resets dynamics and
simplex on already-learned rank streams: NOT clean teacher initialization or
an exact function-preserving simplex substitution. A negative continuation does
not reject the competitor recipe. Future faithful replication needs geometry
and teacher-based initialization as separate work.

## Three T4x2 sessions

Shared source111 seed123 MAIN32768, restored protectedR32, freshAdam .9/.999
 eps1e-8, stableclip1,256warmup. LR MambaR32=3e-4,norm/MLP=1e-4. Frozen BF16
base, FP32 corrections/optimizer. No output gains or residual caps.

NORM: RMS over P of C·state+D·value, separately per token/head/rank,eps1e-5,
learned shared H,P weight, BEFORE SiLU and rank reduction. TANH on raw angle
BEFORE pi*dt accumulation. Identity-to-full2048: geometric norm denominator,
linear angle transition. Legacy executable at strength0, full chain rule.

130/GPU0 NORM+TANH, protectedR32.
130/GPU1 same plus combined reset: zero dt/A/trap/angle in_proj columns;
new log-uniform dt_bias; new softmax-over-R mimo_x, seed130123+layer. Smooth2048
transition. Other learned geometry retained. Dynamics coordinates remain frozen
AFTER the reset in130;131 tests learning them. Record full-strength FIT8 shock.
130/GPU0 versus129 norm-only measures extra tanh across sessions;130 pair tests
combined reset, not each ingredient separately.

131/GPU0 NORM+TANH with unprotected in_projR32, including dynamics columns.
Outside-protection coordinates explicitly zeroed before unlocking to preserve
initial effective kernels.131/GPU1 additionally MLP-down LoRA32 in ALL28 layers.
Comparison130/GPU0→131/GPU0 tests unlocking; within131 tests MLP capacity.
This still freezes dt_bias,D,MIMO factors themselves;134 trains those too.

132/GPU0 repeats131/GPU1 CE as a paired control.
132/GPU1 identical but PURE exact full-vocab forwardKL,T1, original Qwen teacher
on same input tokens. No top-k or CE mixture. CE and KL logged separately; NLL
always truth CE. Teacher token passes counted separately. Equal steps not equal
FLOPs: KD teacher cost reported. Repeated control is intentional.

## Data, measurements and interpretation

Shared real token hashes checked against114/115. A64 validation offset122880;
B32 offset180224; C64 offset196608; FIT16384 windows train offset74973184.
GPU/TPU133 batch1,256 tokens/255 targets, permutation(step*7919+120123)%16384.
TPU134 packs the same pool into4096 contiguous1024-token blocks, permutation
(step*7919+120123)%4096, all1023 next-token targets including256 boundaries. Train pool
4,194,304 distinct tokens. GPU horizon32768=8,388,608 presentations/arm; TPU133
horizon16384=4,194,304/arm; TPU13416384=16,777,216/arm. Actual horizon depends on measured throughput; do not
promise completion. Old preparation and teacher passes accounted separately.

Source A/B/C/context/profile; GPU full-strength initial FIT8/profile. A16 before
full ramp; A64 alone selects best at registered full-strength2048/2560/3072/
3584/4096/4608/5120/6144/8192/12288/16384/24576/32768 within horizon. B32/C64 and
profiles at those same endpoints, never selection.128/512/1024 monitor. Dense
endpoints catch transient126-like peaks. Arbitrary final last evaluated separately,
not retroactively a fixed selection point. Interrupted cursor evaluation retried
before next update on resume. Compare LAST COMMON registered endpoint across arms.

Source/hook tolerances.03/.01 per-window GPU. TPU111 source .03, TPU126 source
.03 on EACH A/B/C window before133 updates. Profile/ordinary relative.02.
Per-layer residual/mixer/MLP RMS/peak, nonzero addition absorption, internal
readout pre/post norm and gate RMS, weight ranges. Last/best panels and same
last127 targets at contexts128/256/512/1024. One seed/reused panels: no independent
confirmation, unique plateau diagnosis or total-token-efficiency claim.

## EXP133: retain126 improvement, TPU4h

Exact source levkremlev888/extent revision68b6879f73207f23b440f3dfc68df153a09a4a0b;
experiments/exp126-t4/worker-1/states/MAIN/checkpoint.json. Metadata7523 contains
separate best parameters4096. Pin, full reference panels and manifest in
results/EXP-133-source-pin.json. Parameter-only reader checks all chunk SHA and
best_step. Locally verified96 leaves/7,145,472 finite coordinates. Best Adam
at4096 is NOT claimed: FRESH Adam all3 branches.

Full-strength fixed relative cap.5 plus residual RMS100 after BOTH additions,
no pre-gate norm or new ramp. Replay A7.01017/B6.99240/C6.69832 within.03 EACH
window before updates. Permutation starts4096, not zero. Source0 eligible best
prevents bad continuation replacing the selected baseline in best reports.

Arm0 MambaR32 LR3e-4; arm1 LR3e-5; arm2 LR3e-5 +zero-effect MLP32 LR1e-5.
All .9/.999 Adam,256warmup, same next data. Separates LR and MLP under a shared
optimizer reset; does not isolate ORIGINAL Adam-trajectory degradation alone.
Alternate128-update blocks favor least-advanced branch. Practical aim sustain
or improve source gain, not a promise to reach a prespecified NLL.

## EXP134: all Mamba parameters trainable, TPU9h

Pinned111 start, paired norm+tanh+MLP32 CE versus pure full-vocab forwardKL T1.
Every Mamba weight trained directly with FP32 master: in/out,dynamics,B/C norms
and biases,dt_bias,D,MIMO factors. Fold old R32 kernels into dense initialization;
no correction-rank restriction. Frozen GQA,embedding/readout,MLP base; train
MLP-down LoRA32 in ALL28 blocks. Broader than132, narrower than competitor full
model fine-tuning; do not label full-decoder replication.

LR Mamba2e-4,norm/MLP2e-5,Adam .9/.95 eps1e-8,clip1,492warmup
(ceil.03*16384),cosine decay to.1 peak at16384. Restore exact schedule cursor.
All dynamics have live gradients. Train at1024 context, nearer the competitor long-context setup but below4k.
Compare134 CE/KL at equal steps; against132 capacity is confounded with changed
Adam/LR and context. All quality panels still explicitly report256 and context probes. Existing dimensional limits remain.

Decoder rematerialization,32-position exact vocabulary tiles, all8 TPU devices
in notebook kernel. No subprocess TPU. Nested multi_transform Adam moments
explicitly sharded like parameters, empty coordinate nodes retained. Legacy
initializer only recognizes top-level Adam and would replicate nested moments.
Tiny full-chain gradients/virtual8 updates verified;1.7B physical HBM/runtime
and quality still pending.

## Wall, storage and persistence

| Experiment | Available | Runner wall INCLUDING reserve | Reserve | New HF cap TOTAL/invocation |
| --- | --- | --- | --- | --- |
|130/131/132 each|7h T4x2|6.6h|45min|4GB each, divided between workers|
|133|4h TPUv5e8|3.6h|45min|4GB|
|134|9h TPUv5e8|8.5h|60min|24GB|

24/30min margins cover notebook setup; reduce wall if setup takes longer.
GPU horizons deliberately long: run until budget/horizon/numerical/storage stop,
not an early one-hour finite search. Repeat same cell to continue saved work.

Current+Adam+best+best_step/cursor local128. GPU remote2048 OR120min and exit.
TPU133 remote120min,134240min and exit. Dense134 state roughly4.23GB/arm,
8.45GB/paired save, about17GB for two full paired uploads;24GB cap leaves margin.
Frozen full-model weights NOT copied. Cap explicit storage_partial, no deletion.
Save training state BEFORE long final diagnostics. Source/current/best and
step/durable_step distinct; external kill may lose unsynced progress. No hardware
outcome or guaranteed throughput from local virtual CPU checks.

Telegram start,30min progress,end/failure/pending. Source/code/data/runtime pin
contracts; separate namespaces and aggregate output for each experiment. Existing
124–129 contracts untouched. One aggregate JSON per session, no manual checkpoints.

Five launch files EXP-130-launch.md through EXP-134-launch.md include last cells.
Local scope and test evidence: EXP-130-134-local-validation.md.

Primary discovery comparison: A64 NLL at last common registered full-strength
endpoint, source delta separately. Practical signal0.1NLL; B/C direction and
trajectory stability reported. This is a screening criterion, not confirmation.

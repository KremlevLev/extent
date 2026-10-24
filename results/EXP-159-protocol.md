# EXP159 — output distillation versus output + intermediate distillation

Prepared24October2026 for one12h2×T4 session while independent TPU157 runs.
No physical GPU result is claimed by this preparation.

## Question

Does adding internal teacher targets help a partially recovered Mamba3MIMO hybrid,
beyond a matched output-distribution distillation recipe?

Both workers start from immutable EXP155 CURRENT118304, HF revision
5e0d5026d03da0320c72162f5216215b137e43c6 (EXP-157-source-pin.json),
with unchanged protected rank256 coordinates, pregate normalization and tanh.
No new folding, memory reset, rank growth or full-decoder unfreezing.
Fresh Adam in BOTH, warmup512, mixerLR3e-4/pregateLR1e-4, global stable clip1.
Same cyclic FIT windows, global source cursor118304, sequence256, batch1.

GPU0: L=0.5 CE(true next token)+0.5 KL(Qwen||student), temperature1.
GPU1: identical +0.1 mean(normalized residual-stream MSE at four layer outputs).
Targets are outputs after blocks7/14/21/28 (zero-based6/13/20/27), on the same
input prefixes. Each token vector is RMS-normalized in FP32 with scale-safe
reduction; average over255prediction positions, feature dimension and four
selected layers. No matching of the different-dimensional Mamba recurrent state.
Teacher target is detached. No teacher outputs are fed into student inputs.

The teacher is original pinned Qwen3-1.7B-Base from the already validated mapper.
Frozen embedding and final norm/head identity are checked against the student
backbone before updates. Teacher block weights remain host arrays; a single
block is placed on the visible GPU at a time. No persistent second full-model
GPU allocation or full-dataset hidden/logit cache. Teacher compute/tokens are
reported separately from accepted student training tokens.

Auxiliary hidden cotangents are injected at the selected block OUTPUT before
that block's VJP, then propagate through the entire preceding student prefix.
Both output KD and hidden objectives are trained in the assembled hybrid.
Bounded-memory layerwise forward/VJP uses original BF16 source arithmetic and
FP32 head/correction operations. No FP16 conversion. Tiny BF16 tests compare
actual streamed gradients against a monolithic combined objective, including
nonzero grown-rank source factors and intermediate (not merely final) targets.
These CPU tests do not establish full1.7B GPU HBM or speed.

## Controls and evaluation

Only hidden_beta/name differ between arms. No matched CE-only GPU arm in this
session; TPU157 CE evidence is cross-hardware context, not a causal KD control.
A benefit here establishes an exploratory advantage of this hidden coefficient
under this recipe, not superiority to tuned CE or all other distillation methods.
The fixed coefficient0.1 and LR are not claimed optimal.

Mandatory160 source-window replay<=.03 BEFORE updates, runtime/source code pins,
finite loss/parameters/optimizer/gradient gates. First8updates are the short
engineering stage; no8192early stopping;262144new updates is a high ceiling.
Record ordinary A/B/C NLL at0/128/256/512/1024/2048/3072/4096/6144/8192 etc.
Fresh156 EN/RU first16windows at source and every1024updates; full128 at final
if reserve permits. Log CE, forwardKL, raw/weighted hidden loss, per-target-layer
loss, norm/finite checks, step latency and student/teacher token costs.
Compare shared registered new-step points and time-to-quality separately;
workers may finish at different steps. A selects best, fresh panels descriptive.
Fresh156 panels have already been inspected and are diagnostic validation now,
not untouched final tests. They are NEVER added to FIT or used as teacher training
inputs. No claim of downstream recovery follows from NLL alone; a winner needs
another locked benchmark and later seed replication.

## Budget, durability and notifications

Default11.8h total (12min margin within12h platform session),18min final reserve,
so work deadline11.5h after parent starts; includes setup/downloads/compiles/evals.
Both GPUs run concurrently. Local adapter+Adam+best every256updates; remote about
every4hours and exit,8GB upload cap PER worker, parent reports2GB cap. Obsolete
local generations pruned by atomic store; frozen donor model not uploaded.
Downloads retry transient/429 errors using existing deadline-aware policy.
Report retries honor numeric Retry-After with backoff. Final sync errors/pending
checkpoints remain explicit, never reported as successful durability.
Parent start+finish notifications only; worker errors terminate the peer and
print saved log tails. Existing own worker namespaces support exact resume;
changed source/objective/code contracts reject incompatible resume.

One file to return: extent-m3q-hidden-distill-159.json. Contains both workers,
per-arm trajectories, health, source gates and final NLL. No cloud job launched
by local preparation.

# Singularity Technical Report and Experiment Ledger

**Working title:** *Singularity: Compute-Efficient Transplantation of Qwen3-14B into a Mamba-3/MLA Hybrid*

**Status:** research prototype; no quality or inference claims are established yet.

**Last updated:** 2026-08-17

This document is the source of truth for the paper. Every number must be marked as one of:

- **MEASURED:** produced by a recorded experiment;
- **DERIVED:** exactly computed from a configuration or measured values;
- **TARGET:** an acceptance criterion, not a result;
- **HYPOTHESIS:** a claim still requiring an experiment.

## 1. Research objective

Convert a pretrained Qwen3-14B Transformer into a deployable hybrid language model without full pretraining:

- preserve the Qwen embeddings, MLPs, normalization layers, and as much learned behavior as possible;
- replace approximately 85% of GQA sequence mixers with Mamba-3 MIMO blocks;
- convert the retained approximately 15% GQA layers into MLA;
- recover from the architecture-conversion “shock” using a severely constrained distillation/adaptation budget;
- preserve or extend useful long-context behavior;
- demonstrate an actual quality/latency/memory Pareto improvement with matched implementations and hardware.

The paper is not about merely combining Mamba and attention. The intended contribution is a **compute-efficient, MIMO-aware cross-architecture transplantation and recovery method**, evaluated jointly with GQA-to-MLA conversion.

## 2. Precise source model and target architecture

### 2.1 Source checkpoint

The source checkpoint is **Qwen/Qwen3-14B** (the released post-trained thinking/non-thinking model, not the separate `Qwen3-14B-Base` repository):

| Field | Value |
|---|---:|
| Layers | 40 |
| Hidden size | 5120 |
| MLP intermediate size | 17408 |
| Query heads | 40 |
| KV heads | 8 |
| Head dimension | 128 |
| Attention Q/K normalization | per-head RMSNorm |
| Vocabulary | 151936 |
| RoPE theta | 1,000,000 |
| Published maximum positions | 40960 |

The immutable source revision is `40c069824f4251a91eefaf281ebe4c544efd3e18`. Architecture recovery targets this exact checkpoint; any later additional reasoning tuning remains a separate stage.

### 2.2 Target ratio

The scientific target is exactly 15% MLA attention and 85% Mamba-3 MIMO. With 40 layers:

- 6 MLA layers = 15%;
- 34 Mamba-3 layers = 85%.

The production configuration uses six uniformly interleaved MLA layers at zero-based indices `[5, 12, 19, 25, 32, 39]`. Earlier 48-layer configurations were based on the wrong Qwen generation and are void for scientific claims.

The main retained-layer schedule is fixed to the uniform indices above. Alternative layer placements, if tested, are small-scale allocation ablations and do not restore the removed 25% architecture.

### 2.3 Long context

The production configuration now matches Qwen3 at `max_position_embeddings=40960`; tiny tests use shorter bring-up lengths. This field alone does not establish long-context support. Long-context claims require:

- a defined target (32K, 64K, or 128K);
- an MLA positional strategy compatible with the source checkpoint;
- training examples at sufficient lengths or a justified extrapolation method;
- RULER/retrieval/state-tracking evaluation across increasing lengths;
- peak-memory and latency measurements at the same lengths.

## 3. Candidate scientific contributions

### C1. MIMO-aware attention-to-Mamba-3 transplant

Develop a principled mapping from pretrained GQA projections into the Mamba-3 MIMO parameterization, including the MIMO channels, complex/data-dependent rotation components, exponential-trapezoidal recurrence, decay/time-step parameters, and output projection.

The contribution is valid only if it beats both random initialization and the strongest direct port of prior attention-to-Mamba initialization under the same token and compute budget.

### C2. Joint conversion shock and compute-efficient recovery

Measure and reduce the compounded damage caused by:

1. replacing most GQA mixers with Mamba-3 MIMO; and
2. converting the retained GQA mixers to MLA.

Compare simultaneous conversion with staged alternatives. A publishable result would show that a proposed initialization/recovery schedule reaches a fixed quality threshold with materially fewer tokens or accelerator-hours.

### C3. Quality-efficiency Pareto at approximately 15% attention

Show how attention retention, MLA latent rank, Mamba state size, and MIMO rank trade quality against:

- prefill time;
- decode time per token / tokens per second;
- KV-cache bytes per generated token;
- recurrent-state bytes per sequence;
- peak HBM;
- long-context retrieval and state tracking.

No inference speed claim is valid with the current reference `lax.scan` implementation. End-to-end speed requires optimized TPU kernels (likely Pallas) and a matched JAX baseline.

## 4. Required controlled ablations

Each ablation changes one factor, uses identical data order, optimizer budget, evaluation code, and as many seeds as affordable. Small-scale experiments establish causality before the 14B confirmation run.

### 4.1 Initialization/transplant ablation

| ID | Preserved Qwen components | New mixer initialization | Purpose |
|---|---|---|---|
| INIT-A | embeddings, MLPs, norms | random Mamba-3/MLA | lower baseline |
| INIT-B | embeddings, MLPs, norms, output projections where compatible | otherwise random | projection-only baseline |
| INIT-C | same | strongest Mamba-in-the-Llama-style Q/K/V/O reuse port | prior-method baseline |
| INIT-D | same | proposed Mamba-3 SISO transplant | isolate Mamba-3 mapping |
| INIT-E | same | proposed Mamba-3 MIMO-aware transplant | main method |
| INIT-F | same | INIT-E without complex/rotation initialization | complex-state ablation |
| INIT-G | same | INIT-E with copied rather than decomposed MIMO channels | MIMO allocation ablation |

Primary immediate-shock metrics: validation NLL/perplexity before recovery, KL to teacher, layer-output normalized MSE, hidden-state cosine similarity, and logit agreement.

Primary recovery metrics: tokens and TPU-hours needed to recover 90%, 95%, and 99% of the teacher-to-shocked-student quality gap.

### 4.2 Joint conversion ablation

| ID | Replaced layers | Retained attention | Conversion schedule |
|---|---|---|---|
| JOINT-A | none | original GQA | teacher/control |
| JOINT-B | none | MLA | GQA-to-MLA only |
| JOINT-C | Mamba-3 | original GQA | Mamba conversion only |
| JOINT-D | Mamba-3 | MLA | simultaneous conversion |
| JOINT-E | Mamba-3 | MLA | Mamba recovery, then MLA conversion |
| JOINT-F | Mamba-3 | MLA | MLA recovery, then Mamba conversion |

This table is essential: without it, quality loss cannot be attributed to the Mamba transplant, MLA conversion, or their interaction.

### 4.3 Architecture/efficiency ablation

- The full-scale architecture is fixed at exactly 15% attention (6/40); no 25% model is retained as a baseline.
- Retained-layer allocation: uniform, late-layer biased, and sensitivity-ranked.
- Mamba-3: SISO vs MIMO; MIMO rank; state sizes 32/64/128; real vs complex; Euler vs trapezoidal where implementation permits.
- MLA: latent KV rank, query rank, partial-RoPE dimensions, and SVD/transplant method.
- Recovery: frozen vs trainable MLPs; mixer-only vs progressive unfreezing; hard-label CE vs teacher KL vs hidden-state matching; staged vs joint recovery.

### 4.4 Minimum credible scaling strategy

Run broad ablations at a smaller Qwen3 scale using the same conversion code, then promote only the best few variants to 14B. A single 14B run cannot establish causality. The 14B experiment should confirm the trend and demonstrate feasibility, not carry the entire ablation matrix.

### 4.5 Available recovery-compute envelope

- **USER-PROVIDED PLANNING ENVELOPE:** one TPU v5e-8 slice may be available for approximately 3–6 months of recovery training if the scientific result requires it.
- **HYPOTHESIS:** this horizon may correspond to roughly 25–50B processed tokens. Pairing 25B with 90 days and 50B with 180 days implies approximately 3,215 training tokens/s sustained end to end.
- **Not yet measured:** compiled 14B step throughput, utilization, input-pipeline stalls, checkpoint/evaluation downtime, preemptions, and actual accelerator availability. Therefore 25–50B is not yet a promised budget or result.
- **Decision rule:** first measure tokens/s for short, medium, and target sequence lengths with the real sharded model. Convert accelerator-hours into a defensible token budget only after that benchmark.
- **Experimental use:** the larger envelope permits recovery scaling curves rather than one terminal run. Mandatory checkpoints should include logarithmically spaced early points and fixed anchors such as 0.1B, 0.3B, 1B, 3B, 6B, 12B, 25B, and—only if still improving—50B tokens.
- **Paper framing:** compute efficiency remains central. The main comparison is tokens and TPU-hours needed to cross frozen recovery thresholds, not merely the best quality after spending the entire envelope.

## 5. Evaluation protocol

### 5.1 Quality and recovery

- Held-out language-model NLL/perplexity on a frozen, contamination-aware corpus slice.
- Teacher-student KL and top-k logit agreement.
- General tasks using a pinned lm-evaluation-harness revision and exact prompts.
- Reasoning/code evaluations only after base capability recovery.
- Report mean and spread over seeds for small-scale causal experiments.

### 5.2 Long context

- RULER and needle/retrieval tests at geometrically increasing lengths.
- State-tracking tasks relevant to Mamba-3's claimed advantage.
- Quality plotted against context length, not a single maximum-context score.
- Separate trained-length performance from extrapolation.

### 5.3 Inference

Compare against the unchanged Qwen3-14B GQA teacher and useful intermediate baselines on the same TPU slice, JAX version, BF16 precision, batch sizes, prompt lengths, generated lengths, compilation state, and sampling settings.

Record:

- time to first token (TTFT);
- inter-token latency (TPOT) and tokens/s;
- prefill latency;
- peak HBM;
- KV/recurrent cache bytes per sequence and per token;
- compilation time separately from steady-state runtime;
- batch-size and context-length sweeps.

The 15% MLA hybrid still has quadratic attention prefill in retained layers. Claims must describe the measured end-to-end behavior, not only asymptotic Mamba complexity.

### 5.4 Provisional success gates (targets, not results)

- **TARGET:** proposed MIMO-aware transplant materially reduces immediate conversion shock against INIT-C and INIT-A.
- **TARGET:** at least 2x fewer recovery tokens or TPU-hours to reach a fixed quality threshold than random initialization.
- **TARGET:** recovered model retains a defensible fraction of teacher quality across both perplexity and downstream tasks, with no benchmark cherry-picking.
- **TARGET:** optimized hybrid establishes a Pareto improvement in measured long-context decode memory/latency, not merely an analytical cache reduction.
- **TARGET:** key causal effects reproduce at small scale and are confirmed by selected 14B runs.

Thresholds will be frozen before final experiments after pilot variance is known.

## 6. Experiment ledger

### EXP-000 — Local reference and unit tests

- **Date reported:** 2026-08-17
- **Hardware:** local CPU for final development run; earlier 2x NVIDIA T4 diagnostics
- **Commit:** `896b292`
- **Result (MEASURED):** 9 local tests passed after adding the full-model Lion optimizer HBM probe.
- **T4 diagnostic (MEASURED):** sharded BF16 backward produced non-finite gradients in several MLP/output leaves; forcing FP32 compute produced finite gradients (`loss=4.8757`, `grad_norm=4.8774`, `max_abs_grad=0.6868`).
- **Interpretation:** old T4 BF16 behavior is treated as an engineering-platform limitation, not a model-quality result. TPU BF16 is the target path.

### EXP-001 — Tiny hybrid BF16 train step on TPU v5e-1

- **Hardware:** 1x TPU v5e device
- **Model:** 128,784-parameter tiny configuration
- **Result (MEASURED):** 7 tests passed in 31.94 s.
- **Sequence length 8, one step (MEASURED):** `loss=4.8756`, `grad_norm=4.8684`, `max_abs_grad=0.6875`, `grads_finite=True`.
- **Sequence length 128, three steps (MEASURED):** losses `4.8529, 4.8529, 4.8324`; gradient norms approximately `1.103`; no non-finite gradient leaves.
- **Interpretation:** BF16 forward/backward and Lion update are numerically viable on TPU for the tiny reference model. This does not test multi-device partitioning.

### EXP-002 — Tiny distributed BF16 train step on TPU v5e-8

- **Hardware:** one TPU v5e-8 slice; 8 devices visible
- **Mesh:** `data=1, fsdp=4, tensor=2`
- **Model:** 128,784 parameters
- **Result (MEASURED):** `partitioned_param_arrays=20/47`; `partitioned_train_state_arrays=40/97`.
- **Step 0 (MEASURED):** `loss=4.8750`, `grad_norm=4.8699`, `max_abs_grad=0.6875`, `grads_finite=True`, `nonfinite_grad_leaves=0`.
- **Interpretation:** the intended 8-device mesh, BF16 path, parameter sharding, Lion-state sharding, forward pass, backward pass, and update execute together on the tiny model.

### EXP-003 — VOID: wrong-generation bring-up shape audit and parameter allocation

- **Hardware:** one TPU v5e-8 slice for allocation; shape audit is hardware-independent
- **Configuration:** superseded 12 MLA / 36 Mamba bring-up prototype; removed from active configs and not used as a paper baseline
- **Commit:** `7c7b725`
- **Exact parameter count (DERIVED):** 14,655,260,384 across 723 tensors.
- **Partitioned tensors (DERIVED):** 278/723.
- **Global BF16 weight size (DERIVED):** 27.298 GiB.
- **Ideal per-device BF16 weights (DERIVED):** 3.418 GiB.
- **Ideal per-device weights + BF16 gradients + BF16 Lion momentum (DERIVED):** 10.253 GiB.
- **Real parameter allocation (MEASURED):** 3.418 GiB on each of TPU_0 through TPU_7.
- **Result (MEASURED):** full parameter initialization passed; optimizer and train step were not created.
- **Interpretation:** the 1x4x2 sharding mechanism worked, but the model used wrong-generation dimensions. No architecture or memory number from this experiment is valid for the Qwen3 target.

### EXP-004 — VOID: wrong-generation Lion persistent-state HBM probe

- **Commit:** `896b292`
- **Hardware:** one TPU v5e-8 slice
- **Mesh:** `data=1, fsdp=4, tensor=2`
- **Status:** VOID for the Qwen3 target; mechanical optimizer-sharding evidence only.
- **Command:** `full_preflight(["--initialize-optimizer"])` in the notebook process.
- **Parameter allocation (MEASURED):** 3.418 GiB on each of eight devices.
- **Lion optimizer allocation (MEASURED):** 3.418 GiB on each of eight devices.
- **Combined persistent allocation (MEASURED):** 6.835 GiB on each of eight devices.
- **Nominal remaining HBM (DERIVED):** 9.165 GiB/device relative to 16 GiB, before gradients, activations, XLA temporaries, executable buffers, and runtime overhead.
- **Result (MEASURED):** `optimizer_initialization=PASS`; no gradients, activations, or train step were created.
- **Interpretation:** the optimizer-sharding mechanism worked, but the model used wrong-generation dimensions. This is not a Qwen3 memory result.

### EXP-005 — VOID: wrong-generation 7/41 target shape audit

- **Hardware:** local shape-only trace; no full arrays allocated
- **Configuration:** obsolete 48-layer Qwen2.5-shaped prototype, not Qwen3-14B
- **Exact parameter count (DERIVED):** 14,698,336,184 across 743 tensors.
- **Partitioned tensors (DERIVED):** 263/743.
- **Global BF16 weight size (DERIVED):** 27.378 GiB.
- **Ideal per-device BF16 weights (DERIVED):** 3.428 GiB.
- **Ideal per-device weights + BF16 gradients + BF16 Lion momentum (DERIVED):** 10.285 GiB.
- **Ideal training-state headroom at 16 GiB (DERIVED):** 5.715 GiB/device before activations and XLA/runtime overhead.
- **Result (DERIVED):** shape-only preflight passed.
- **Status:** VOID for research use. Retained only as an engineering-error log.

### EXP-006 — VOID: Qwen2.5 metadata audit

- **Implementation commit title:** `feat: add pinned Qwen2.5 streaming checkpoint import`
- **Source:** `Qwen/Qwen2.5-14B`
- **Immutable revision:** `97e1e76335b7017d8f67c08a19d103c0504298c9`
- **Mode:** metadata-only; no model shards downloaded
- **Checkpoint index (MEASURED):** 579 tensors, 8 shards, 29,540,067,328 bytes (27.511 GiB).
- **Direct mapping (DERIVED/VALIDATED):** 243 tensors containing 11,749,790,720 parameters.
- **Share of final target parameters directly preserved (DERIVED):** 79.94%.
- **Direct scope:** token embeddings, lm_head, final norm, all input/post-attention norms, and all gate/up/down MLP projections.
- **Result (MEASURED):** pinned config/index validation and target-shape mapping validation passed.
- **Status:** VOID. The audit was internally correct but targeted the wrong model generation and must not be used in the paper.

### EXP-007 — Pinned Qwen3-14B metadata and direct-map audit

- **Implementation commit title:** `fix: migrate Singularity source and target to Qwen3-14B`
- **Source:** `Qwen/Qwen3-14B` (not `Qwen3-14B-Base`)
- **Immutable revision:** `40c069824f4251a91eefaf281ebe4c544efd3e18`
- **Mode:** metadata-only; no model shards downloaded
- **Checkpoint index (MEASURED):** 443 tensors, 8 shards, 29,536,614,400 bytes (27.508 GiB).
- **Direct mapping (DERIVED/VALIDATED):** 203 tensors containing 12,251,714,560 parameters.
- **Share of final target parameters directly preserved (DERIVED):** 83.30%.
- **Direct scope:** token embeddings, lm_head, final norm, all input/post-attention norms, and all gate/up/down MLP projections.
- **Mixer tensors reserved for conversion:** Q/K/V/O projections and Q/K per-head RMSNorm parameters.
- **Result (MEASURED):** pinned Qwen3 config/index validation and target-shape mapping validation passed.
- **Independent reproduction (MEASURED):** the user reproduced the same metadata-only output in a separate runtime: 443 checkpoint tensors and 203 direct tensors / 12,251,714,560 parameters.

### EXP-008 — Qwen3 6/34 target shape audit

- **Configuration:** 6 MLA / 34 Mamba-3 MIMO, MLA indices `[5, 12, 19, 25, 32, 39]`
- **Exact parameter count (DERIVED):** 14,707,402,224 across 631 tensors after adding source-compatible Q/K RMSNorm to the six MLA mixers.
- **Partitioned tensors (DERIVED):** 220/631.
- **Global BF16 weight size (DERIVED):** 27.395 GiB.
- **Ideal per-device BF16 weights (DERIVED):** 3.429 GiB.
- **Ideal per-device weights + BF16 gradients + BF16 Lion momentum (DERIVED):** 10.288 GiB.
- **Ideal headroom at 16 GiB (DERIVED):** 5.712 GiB/device before activations and XLA/runtime overhead.
- **Result (DERIVED):** shape-only preflight passed.
- **Status:** real parameter/Lion allocation for this exact Qwen3 target remains to be measured on v5e-8.

### EXP-009 — Exact JAX Qwen3 teacher parameter-tree contract

- **Implementation commit title:** `feat: add exact Qwen3 GQA teacher parity scaffold`
- **Mode:** local CPU structural tests plus metadata-only pinned checkpoint audit; no model shards downloaded
- **Teacher architecture:** 40-layer Qwen3 GQA with 40 query heads, 8 KV heads, head dimension 128, per-head Q/K RMSNorm, split-half RoPE, SwiGLU, and untied output head.
- **Checkpoint-to-teacher mapping (DERIVED/VALIDATED):** 443 of 443 source tensors mapped bijectively to 443 JAX parameter leaves; 14,768,307,200 parameters; 100% tensor coverage.
- **Tests (MEASURED):** 22 passed in 38.55 s, including BF16 parameter/logit shapes, split-half RoPE fixture, causal-independence test, and complete mapping bijection.
- **Result (MEASURED):** metadata preflight printed `teacher_mapping=PASS tensors=443 parameters=14,768,307,200 coverage=100%`.
- **Independent reproduction (MEASURED):** the user reproduced the complete output in a separate runtime, including 443/443 teacher tensors, 14,768,307,200 parameters, and 100% coverage.
- **Interpretation:** source names and shapes are fully accounted for. This is a prerequisite for parity, not evidence of numerical logits/NLL parity; cross-framework comparison after loading the pinned weights remains pending.

### EXP-010 — Qwen3 decoder-layer cross-framework parity

- **Implementation commit title:** `feat: add cross-framework Qwen3 layer parity harness`
- **Status:** PASS; independently executed in Kaggle on two visible CUDA devices.
- **Frozen source:** `Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18`; official reference implementation pinned to `transformers==4.51.0` as recorded in the source config.
- **Scope:** decoder layer 0, all 11 layer tensors, deterministic FP32 input, sequence length 4, seed 123, eager causal attention.
- **Download scope (DERIVED/VALIDATED):** only `model-00001-of-00008.safetensors` is required for layer 0; HTTP metadata reports 3,841,788,544 bytes (3.578 GiB).
- **Acceptance gates (FROZEN TARGET):** maximum absolute output error <= `5e-3` and relative L2 error <= `5e-4`.
- **Synthetic cross-framework test (MEASURED):** an official Transformers Qwen3 decoder layer and the JAX decoder layer passed with identical small synthetic weights; the three parity-module tests passed in 9.38 s.
- **Regression suite (MEASURED):** 25 tests passed in 46.09 s after adding the harness.
- **Real checkpoint result (MEASURED):** layer 0, sequence length 4, seed 123, FP32; `max_abs=4.0531158447265625e-06`, `mean_abs=6.121665592218051e-08`, `rmse=9.890513461222466e-08`, `relative_l2=9.860404374182388e-08`, `cosine_similarity=0.9999999999999956`.
- **Result:** `PARITY-PASS`; both frozen gates passed with substantial margin. The JAX layer is accepted as the numerical teacher for subsequent conversion-shock experiments.

### EXP-011 — Qwen3 GQA-to-MLA initialization shock

- **Implementation commit title:** `feat: add Qwen3 GQA-to-MLA conversion baselines`
- **Status:** completed for layer 0 on a Kaggle GPU runtime with two visible CUDA devices.
- **Frozen input:** pinned Qwen3 layer 0, sequence length 4, hidden-state seed 123, FP32, SVD seed 0.
- **Common tensors:** source input/post-attention norms and MLP are copied identically in every variant.
- **MLA controls:** `random_mla`; `projection_copy_random_kv`; joint rank-512 SVD with 64 partial-RoPE dimensions.
- **Prior-method baseline:** joint SVD while retaining all eight Qwen GQA RoPE-key heads; analytical cache is 1024 elements/token/layer versus 2048 for source GQA (50% reduction).
- **Aggressive target:** the same joint SVD but averaging eight RoPE-key projections into one shared head; analytical cache is 576 elements/token/layer (71.875% reduction).
- **Primary shock metric:** relative L2 and cosine similarity of the clean mixer output. Full decoder-output metrics are secondary because the residual path can hide mixer damage.
- **Regression suite (MEASURED):** 27 tests passed in 49.98 s; grouped/shared conversion paths, Qwen split-half partial-RoPE indices, shapes, and finite outputs passed.
- **Random MLA (MEASURED):** mixer `relative_l2=1.024080`, `cosine=-0.004875`; decoder `relative_l2=0.042905`, `cosine=0.999085`.
- **Copied Q/O + random KV (MEASURED):** mixer `relative_l2=1.044106`, `cosine=0.013729`; copying compatible projections alone did not improve the random-KV baseline.
- **Joint SVD, eight grouped RoPE heads (MEASURED):** joint reconstruction `relative_l2=0.650992`; mixer `relative_l2=0.647275`, `cosine=0.762263`; decoder `relative_l2=0.027075`, `cosine=0.999635`.
- **Joint SVD, one shared RoPE head (MEASURED):** RoPE-weight aggregation `relative_l2=0.933184`; mixer `relative_l2=0.687968`, `cosine=0.726283`; decoder `relative_l2=0.028789`, `cosine=0.999587`.
- **Interpretation:** rank-512 joint reconstruction is the dominant measured error source. Collapsing eight RoPE heads to one causes an additional mixer penalty of `0.040693` absolute relative-L2 and `0.035980` cosine, but is secondary at this rank. Residual-dominated decoder cosine conceals the mixer damage and is not an acceptable primary shock metric.
- **Raw artifact:** `results/EXP-011-qwen3-mla-shock-layer0.json`.

### EXP-012 — MLA KV-rank shock sweep

- **Implementation commit title:** `feat: add MLA rank shock sweep`
- **Status:** completed on a Kaggle GPU runtime.
- **Frozen factors:** one deterministic maximum-rank joint SVD at partial-RoPE width 64 and SVD seed 0; ranks 256, 512, 1024, and 1536 use nested prefixes of the same factors.
- **Variants:** eight grouped/shared-RoPE combinations across four ranks; teacher input, sequence length, seed, and copied non-mixer tensors match EXP-011.
- **Scientific purpose:** separate low-rank reconstruction damage from partial-RoPE damage and from eight-to-one RoPE-head aggregation.
- **Rank-1536 control:** the joint K-content/V matrix has output width 1536, so rank 1536 removes low-rank reconstruction loss up to numerical precision. Grouped cache returns to 2048 elements/token and intentionally offers no cache saving; shared-RoPE cache remains 1600 elements/token.
- **Regression suite (MEASURED):** 28 tests passed in 50.61 s; reused-factor rank errors were verified monotonic on the small fixture.
- **Factorization time (MEASURED):** 8.003 s for the reusable rank-1536 factorization.
- **Rank 256 (MEASURED):** grouped/shared mixer L2 `0.786286/0.819324`, cosine `0.617904/0.575500`, cache reduction `62.5%/84.375%`.
- **Rank 512 (MEASURED):** grouped/shared mixer L2 `0.588695/0.645381`, cosine `0.808355/0.764634`, cache reduction `50%/71.875%`.
- **Rank 1024 (MEASURED):** grouped/shared mixer L2 `0.225261/0.353762`, cosine `0.974299/0.935828`, cache reduction `25%/46.875%`.
- **Rank 1536 (MEASURED):** grouped reconstruction L2 `6.817e-7`, mixer L2 `0.000202`, cosine `0.999999980`; shared mixer L2 `0.271009`, cosine `0.962960`.
- **Interpretation:** the nested Pareto curve confirms weight-rank loss is dominant at ranks <=1024. With low-rank loss removed, grouped partial-RoPE is nearly exact at length 4 and eight-to-one RoPE aggregation becomes the dominant error. The aggregation penalty grows from `0.033038` mixer L2 at rank 256 to `0.270807` at rank 1536 because it is no longer masked by SVD loss.
- **Method correction:** EXP-012 rank-512 factors come from the optimal nested full-width factorization and outperform the standalone randomized rank-512 factors in EXP-011 (`0.588695` versus `0.647275` grouped mixer L2). Future converters use the nested full-width factorization; EXP-011 remains as a recorded weaker engineering baseline.
- **Raw artifact:** `results/EXP-012-qwen3-mla-rank-sweep-layer0.json`.

### EXP-013 — MLA positional shock versus context length

- **Implementation commit title:** `feat: add MLA context-length shock sweep`
- **Status:** completed on a Kaggle GPU runtime.
- **Lengths:** 4, 64, 256, and 1024 using prefixes of one deterministic Gaussian hidden-state sequence.
- **Variants:** rank 512 and no-SVD-loss rank 1536, each with grouped and shared RoPE keys.
- **Metrics:** mixer-output relative L2/cosine over all tokens and separately over the last 128 tokens.
- **Purpose:** test whether the near-exact rank-1536 grouped result at length 4 persists as omitted low-frequency RoPE dimensions accumulate phase. This is a controlled positional diagnostic, not evidence of natural-language long-context quality.
- **Regression suite (MEASURED):** 29 tests passed in 50.52 s after adding the launcher, switching the default rank-512 converter to nested optimal factors, and avoiding unused MLP device allocation in mixer-only probes.
- **Factorization time (MEASURED):** 8.909 s for the reusable maximum-rank factorization.
- **Rank-512 grouped (MEASURED):** all-token mixer L2 rises monotonically from `0.588695` at length 4 to `0.699936/0.735776/0.768287` at lengths 64/256/1024. At length 1024, tail L2 is `0.804776` and tail cosine is `0.606150`.
- **Rank-512 shared (MEASURED):** all-token mixer L2 rises from `0.645381` to `0.864422/0.921571/0.973569`. At length 1024, tail L2 reaches `1.017279` and tail cosine falls to `0.291382`.
- **No-SVD-loss grouped control (MEASURED):** all-token L2 grows from `0.000202` at length 4 to `0.005093/0.024409/0.098180`. Its length-1024 tail L2 is `0.178582` with cosine `0.984014`, directly exposing accumulated partial-RoPE positional error.
- **No-SVD-loss shared control (MEASURED):** all-token L2 grows from `0.271009` to `0.604397/0.684070/0.753150`; length-1024 tail L2 is `0.829012` and tail cosine is `0.607683`.
- **Interpretation:** every path worsens monotonically with context. Rank loss, omitted RoPE frequencies, and eight-to-one RoPE-head aggregation compound rather than remain independent short-sequence perturbations. The naive shared rank-512 target is therefore not accepted as the production MLA initializer. Wider RoPE or a stronger covariance/frequency-aware converter must be evaluated at a fixed cache budget before recovery training.
- **Compact artifact:** `results/EXP-013-qwen3-mla-context-sweep-layer0.json` retains all primary relative-L2/cosine values; the original Kaggle JSON additionally contains max-absolute, mean-absolute, and RMSE fields.

### EXP-014 — Fixed-cache MLA RoPE-width sweep

- **Implementation commit title:** `feat: add fixed-cache MLA RoPE width sweep`
- **Status:** completed on a Kaggle GPU runtime.
- **Frozen budget:** shared MLA cache is exactly 576 elements/token/layer, matching the EXP-013 rank-512, width-64 target and retaining a 71.875% analytical reduction from Qwen3 GQA.
- **Widths and ranks:** RoPE widths `32/64/96/128` pair with latent ranks `544/512/480/448`, respectively. This keeps `latent rank + shared RoPE width = 576` for every deployable variant.
- **Context:** one deterministic Gaussian sequence of length 1024; primary metrics are computed over all tokens and the final 128 tokens.
- **Controls:** for each width, maximum-rank grouped RoPE isolates frequency-removal error without SVD or head aggregation; maximum-rank shared RoPE adds only eight-to-one RoPE-key aggregation; fixed-cache shared RoPE then adds the budget-constrained rank loss.
- **Scientific question:** at identical inference cache cost, does preserving more positional frequencies compensate for the smaller content/value latent rank? The width-128 case preserves full source RoPE and tests whether eliminating partial-RoPE drift outweighs reducing latent rank to 448.
- **Boundary:** this remains a synthetic single-layer initialization diagnostic. It cannot establish language quality, post-recovery behavior, or end-to-end inference speed.
- **Regression suite (MEASURED):** 30 tests passed in 50.18 s, including an end-to-end width-128 conversion test with zero non-RoPE key dimensions.
- **Factorization times (MEASURED):** `9.395/7.127/4.971/3.665` s for widths `32/64/96/128`.
- **Fixed-cache results (MEASURED):** width/rank `32/544`, `64/512`, `96/480`, and `128/448` produce all-token L2 `0.953267/0.973569/1.027065/1.100851`; tail L2 is `1.002607/1.017279/1.059122/1.130818`; tail cosine is `0.309006/0.291382/0.207086/0.081792`.
- **Grouped no-SVD-loss controls (MEASURED):** all-token L2 falls sharply as more RoPE is retained: `0.516543/0.098180/0.012942/0.00000295` for widths `32/64/96/128`. Thus full RoPE is numerically faithful when heads are not collapsed.
- **Shared no-SVD-loss controls (MEASURED):** all-token L2 instead worsens `0.703080/0.753150/0.960664/1.130436` as the naively averaged shared head covers more dimensions. The corresponding weight aggregation errors remain approximately `0.930–0.935`.
- **Interpretation:** at the fixed 576-element budget, width 32 is the least damaged candidate, but its tail L2 remains above 1 and is not acceptable. Increasing RoPE width cannot repair naive eight-to-one head averaging; it increases the amount of badly aggregated key content. The next controlled baseline must rotate/concentrate information across heads before removing positional components.
- **Raw artifact:** `results/EXP-014-qwen3-mla-rope-width-sweep-layer0.json`.

### EXP-015 — Standard TransMLA RoRoPE positional shock

- **Implementation commit title:** `feat: add TransMLA RoRoPE shock baseline`
- **Status:** harness implemented; Kaggle execution pending.
- **Prior-method baseline:** standard RoRoPE from TransMLA, without FreqFold, BKV-PCA, or latent KV compression. This is explicitly related-work reproduction, not the proposed contribution.
- **Calibration/evaluation split:** independent deterministic Gaussian sequences, both length 1024, with seeds 123 and 124. PCA rotations are fitted only on calibration keys.
- **Mechanism:** for each of Qwen3's 64 split-half RoPE frequencies, fit an orthogonal PCA rotation over the eight normalized KV heads; apply the same rotation to absorbed Q and K components; retain RoPE on the leading `1/2/4/8` components and remove it from the rest.
- **Controls:** retaining all eight components must preserve original GQA attention numerically. One component is the aggressive single-RoPE-head target; intermediate counts expose the positional cache/fidelity curve.
- **Scope:** this experiment isolates RoPE decoupling. All rotated key components and original values remain present, so it does not yet claim KV-cache compression. FreqFold and joint balanced KV compression are subsequent, separately measured stages.
- **Qwen3 caveat:** PCA is fitted after Qwen3 per-head K normalization and applied after Q/K normalization. A later weight-mapping experiment must explicitly test whether this ordering remains compatible with the deployable absorbed MLA path.
- **Regression suite (MEASURED):** 31 tests passed in 50.94 s. The all-component tiny fixture matches ordinary Qwen split-half RoPE GQA attention within `2e-5` absolute/relative tolerance, and PCA eigenvalues are verified in descending order.

## 7. Development milestones

1. **Completed only for the sharding mechanism:** wrong-generation weights + Lion state fit on v5e-8; exact Qwen3 HBM validation remains pending.
2. **Completed:** set the Qwen3 target schedule to exactly 6/40 MLA layers.
3. **Completed:** validate the pinned Qwen3-14B metadata, target shapes, and streaming mapping contracts.
4. **Current:** implement the exact Qwen3 GQA teacher and establish logits/NLL parity before conversion.
5. Implement and test GQA-to-MLA conversion baselines.
6. Implement Mamba-3 transplant variants and single-layer shock tests.
7. Run small-scale ablations and freeze the 14B recovery recipe.
8. Compile the first guarded full-model short-sequence forward/backward.
9. Recovery training, fixed evaluation checkpoints, and failure logging.
10. Optimized inference kernel and matched end-to-end benchmarks.
11. Only after base recovery: separate reasoning SFT study using legally and scientifically documented data.

## 8. Reasoning SFT boundary

“Claude-like reasoning” is not part of the architecture-recovery claim. It should be a later experiment with explicit data provenance, permissions, filtering, and a frozen pre-SFT checkpoint. Otherwise architecture recovery and behavior imitation become confounded. Prefer reproducible/open reasoning datasets or lawfully generated teacher traces, and evaluate reasoning improvements separately from retained base capabilities.

## 9. Related work anchors

- Mamba-3: Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, arXiv:2603.15569. https://arxiv.org/abs/2603.15569
- Wang et al., *The Mamba in the Llama: Distilling and Accelerating Hybrid Models*, arXiv:2408.15237. https://arxiv.org/abs/2408.15237
- Moudgil et al., *Attention to Mamba: A Recipe for Cross-Architecture Distillation*, arXiv:2604.14191. https://arxiv.org/abs/2604.14191
- Ji et al., *Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs*, arXiv:2502.14837. https://arxiv.org/abs/2502.14837
- TransMLA, arXiv:2502.07864. https://arxiv.org/abs/2502.07864
- TransMLA code and RoRoPE/FreqFold baseline: https://github.com/MuLabPKU/TransArch/tree/main/TransMLA_NeurIPS_2025
- Zhou et al., *CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling Multi-Head Latent Attention*, arXiv:2603.17946. https://arxiv.org/abs/2603.17946
- Qwen Team, *Qwen3 Technical Report*, arXiv:2505.09388. https://arxiv.org/abs/2505.09388
- Exact pinned source: https://huggingface.co/Qwen/Qwen3-14B/tree/40c069824f4251a91eefaf281ebe4c544efd3e18

## 10. Rules for future entries

For every experiment record:

- experiment ID, date, commit hash, config hash, checkpoint revision;
- hardware topology and software versions;
- dataset identity/revision, token count, sequence-length distribution, and seed;
- trainable/frozen parameter groups;
- optimizer, learning-rate schedule, batch size, accumulation, precision, and wall time;
- raw metrics and failure traces;
- interpretation separated from observation;
- artifact/checkpoint location and whether it is reproducible.

Failed experiments stay in the ledger. They are evidence for engineering decisions and prevent accidental repetition.

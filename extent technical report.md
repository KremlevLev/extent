# Extent Technical Report and Experiment Ledger

**Working title:** *Extent-14B: Compute-Efficient Transplantation of Qwen3-14B into a Mamba-3/MLA Hybrid*

**Status:** research prototype; no quality or inference claims are established yet.

**Last updated:** 2026-08-20

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
| INIT-D | same | head-pooled Q/K state projections copied across MIMO channels | SISO-copy control |
| INIT-E | same | disjoint Q/K head groups assigned to distinct MIMO channels | MIMO-allocation candidate |
| INIT-F | same | direct INIT-C x/B/C slices RMS-matched to the canonical random base | isolate scale mismatch |
| INIT-G | same | 25% interpolation from random x/B/C toward INIT-F | weak-prior candidate |
| INIT-H | same | 50% interpolation from random x/B/C toward INIT-F | interpolation-strength control |
| INIT-I | same | INIT-G offset over a trainable random base, linearly removed during early updates | transient-prior candidate |

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

- **Implementation commit title:** `fix: migrate Singularity source and target to Qwen3-14B` (historical title, before the Extent rename)
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
- **Status:** completed on a Kaggle GPU runtime.
- **Prior-method baseline:** standard RoRoPE from TransMLA, without FreqFold, BKV-PCA, or latent KV compression. This is explicitly related-work reproduction, not the proposed contribution.
- **Calibration/evaluation split:** independent deterministic Gaussian sequences, both length 1024, with seeds 123 and 124. PCA rotations are fitted only on calibration keys.
- **Mechanism:** for each of Qwen3's 64 split-half RoPE frequencies, fit an orthogonal PCA rotation over the eight normalized KV heads; apply the same rotation to absorbed Q and K components; retain RoPE on the leading `1/2/4/8` components and remove it from the rest.
- **Controls:** retaining all eight components must preserve original GQA attention numerically. One component is the aggressive single-RoPE-head target; intermediate counts expose the positional cache/fidelity curve.
- **Scope:** this experiment isolates RoPE decoupling. All rotated key components and original values remain present, so it does not yet claim KV-cache compression. FreqFold and joint balanced KV compression are subsequent, separately measured stages.
- **Qwen3 caveat:** PCA is fitted after Qwen3 per-head K normalization and applied after Q/K normalization. A later weight-mapping experiment must explicitly test whether this ordering remains compatible with the deployable absorbed MLA path.
- **Regression suite (MEASURED):** 31 tests passed in 50.94 s. The all-component tiny fixture matches ordinary Qwen split-half RoPE GQA attention within `2e-5` absolute/relative tolerance, and PCA eigenvalues are verified in descending order.
- **Exact-invariance control (MEASURED):** retaining all eight RoPE components gives all-token L2 `3.642e-6`, tail L2 `5.380e-6`, and all-token cosine `0.999999999993`. The implementation invariant passes decisively.
- **One-component result (MEASURED):** one 128-element shared RoPE component retains `52.726%` of measured positional key energy; all-token/tail L2 is `0.715780/0.807436` and cosine is `0.744083/0.675598`.
- **Two-component result (MEASURED):** 256 RoPE-cache elements retain `61.491%` energy; all-token/tail L2 is `0.682616/0.775244`.
- **Four-component result (MEASURED):** 512 RoPE-cache elements retain `76.600%` energy; all-token/tail L2 is `0.585000/0.673825`.
- **Comparison with naive aggregation (MEASURED):** at full RoPE width, naive eight-to-one averaging had all/tail L2 `1.130436/1.169610`; one-component RoRoPE improves these to `0.715780/0.807436`, reductions of `36.68%/30.97%`, while all-token/tail cosine rises from `0.176311/0.092147` to `0.744083/0.675598`.
- **Interpretation:** head-aware orthogonal concentration is materially better than averaging and is the valid prior-method direction. Nevertheless, one-component RoRoPE remains a large shock. It is not yet a 128-element cache result because the other seven components remain as NoPE content awaiting joint compression.
- **Raw artifact:** `results/EXP-015-qwen3-rorope-shock-layer0.json`.

### EXP-016 — Fixed-RoPE-cache FreqFold sweep

- **Implementation commit title:** `feat: add fixed-cache RoRoPE FreqFold sweep`
- **Status:** completed on a Kaggle GPU runtime.
- **Frozen RoPE cache:** every variant retains exactly 128 elements/token/layer, equal to one original Qwen3 key head.
- **Variants:** adjacent-frequency fold sizes `1/2/4/8`. Each group jointly fits PCA over `8 × fold` KV-head/frequency features and retains the leading `fold` components, keeping the total retained positional dimensionality fixed.
- **Calibration/evaluation:** independent deterministic Gaussian sequences of length 1024 with seeds 123/124, matching EXP-015.
- **Scientific question:** can nearby-frequency grouping retain more useful positional signal than independent per-frequency PCA without increasing the shared RoPE cache?
- **Implementation boundary:** this is a FreqFold-style activation diagnostic following TransMLA's joint head/frequency PCA principle. It is not yet claimed numerically identical to the official PyTorch converter. Fold 1 is regression-tested to equal standard one-component RoRoPE.
- **Compression boundary:** rotated NoPE keys and values are still uncompressed. Balanced joint KV and covariance-aware compression follow only after selecting the positional transform.
- **Regression suite (MEASURED):** 32 tests passed in 49.68 s. Fold 1 matches the independently implemented standard one-component RoRoPE path within `2e-5` absolute/relative tolerance; a fold-2 shape/finite-output probe also passes.
- **Fold 1 (MEASURED):** retained positional-energy fraction `0.527264`; all/tail mixer L2 `0.715780/0.807436`; all/tail cosine `0.744083/0.675598`. These reproduce EXP-015.
- **Folds 2/4/8 (MEASURED):** retained energy rises monotonically to `0.552234/0.586278/0.628286`, but all-token L2 worsens to `0.727976/0.739489/0.745680` and tail L2 worsens to `0.818274/0.827886/0.838460`.
- **Interpretation:** nearby-frequency mixing increases PCA energy capture but degrades attention fidelity on the held-out sequence. Spectral energy is therefore not a sufficient selection proxy for this Qwen3 layer. Fold 1 is frozen for the next compression diagnostic; folds 2/4/8 remain recorded negative ablations.
- **Raw artifact:** `results/EXP-016-qwen3-freqfold-sweep-layer0.json`.

### EXP-017 — Cache-matched balanced KV compression probe

- **Implementation commit title:** `feat: add cache-matched balanced KV compression probe`
- **Status:** completed on a Kaggle GPU runtime.
- **Frozen positional transform:** standard fold-1 RoRoPE with one 128-element positional component, selected by EXP-015/016.
- **Frozen total cache:** 576 elements/token/layer: 128 RoPE plus rank-448 latent KV, a 71.875% reduction from the 2048-element Qwen3 GQA cache.
- **Compared variants:** uncompressed RoRoPE control; rank-448 uncentered activation PCA; rank-448 TransMLA-style BKV-balanced activation PCA.
- **BKV rule:** measure `alpha = E||K_nope|| / E||V||` on calibration activations, factorize `[K_nope / alpha, V]`, and restore the K scale after reconstruction.
- **Calibration/evaluation:** independent deterministic Gaussian sequences of length 1024 with seeds 123/124. The PCA basis is fitted only on calibration activations.
- **Metrics:** calibration joint-reconstruction L2, held-out K/V reconstruction L2, and mixer all-token/tail parity against the original Qwen3 GQA layer.
- **Boundary:** activation PCA is an oracle-style diagnostic of the compressed activation subspace, not yet a deployable linear weight mapping. It intentionally precedes CARE-style covariance-aware weight mapping and real-text calibration.
- **Regression suite (MEASURED):** 33 tests passed in 53.63 s. Full-rank activation PCA reconstructs its fixture within `2e-5`, BKV scaling equalizes mean tokenwise K/V norms, and the earlier RoRoPE invariants remain green.
- **Uncompressed RoRoPE control (MEASURED):** all/tail mixer L2 `0.715780/0.807436`, exactly reproducing EXP-015/016.
- **Measured imbalance:** `alpha=86.8204`, reflecting the very large norm mismatch between normalized Qwen3 `K_nope` activations and unnormalized V activations.
- **Plain activation PCA (MEASURED):** calibration joint L2 `0.260045`; held-out K/V reconstruction L2 `0.513479/1.127435`; all/tail mixer L2 `1.089895/1.080180`; all/tail cosine `0.103675/0.094275`.
- **BKV-balanced PCA (MEASURED):** calibration joint L2 `0.416602`; held-out K/V reconstruction L2 `0.713854/0.757706`; all/tail mixer L2 `0.951443/0.975588`; all/tail cosine `0.316530/0.240102`.
- **Interpretation:** BKV balancing reduces all-token mixer L2 by `12.70%` relative to plain PCA and prevents the factorization from nearly discarding V. It is retained as a required baseline. However, rank-448 compression still adds substantial damage beyond the uncompressed RoRoPE shock. The Gaussian calibration matrix has only 1024 samples for 1920 features and generalizes poorly, so it cannot decide the deployable mapping.
- **Raw artifact:** `results/EXP-017-qwen3-balanced-kv-probe-layer0.json`.

### EXP-018 — Real-text balanced KV calibration

- **Implementation commit title:** `feat: add real-text MLA calibration probe`
- **Status:** completed on Kaggle GPU; deployable direction frozen for the next implementation stage.
- **Dataset:** pinned `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1` train split.
- **Tokenization/source:** pinned Qwen3-14B tokenizer at the immutable source revision; actual checkpoint embedding rows provide layer-0 hidden inputs.
- **Split:** first 8192 packed real-text tokens for calibration and the following disjoint 1024 tokens for evaluation. Empty records are skipped and EOS separates documents.
- **Frozen method:** fold-1 RoRoPE, one 128-element positional cache, rank-448 latent, plain versus BKV-balanced uncentered activation PCA, total cache 576.
- **Compute path:** randomized PCA runs on the active JAX accelerator rather than CPU NumPy; calibration and evaluation remain FP32 diagnostics.
- **Scientific question:** does the EXP-017 failure primarily reflect under-sampled isotropic Gaussian calibration, and does BKV remain beneficial on real language-token activations?
- **Boundary:** WikiText-2 is only a calibration probe, not the final recovery corpus or evaluation benchmark. Layer 0 alone cannot establish a global rank schedule or downstream quality.
- **Regression suite (MEASURED):** 34 tests passed in 53.49 s. Pinned-text packing, accelerator PCA full-rank reconstruction, standard RoRoPE, and FreqFold regressions pass after factoring out the reusable key rotation.
- **Uncompressed RoRoPE control (MEASURED):** all/tail mixer relative L2 `0.174770/0.179739`; all/tail cosine `0.986864/0.986213`.
- **Plain rank-448 PCA (MEASURED):** all/tail mixer relative L2 `0.327066/0.322150`; all/tail cosine `0.946777/0.948648`.
- **BKV rank-448 PCA (MEASURED):** `alpha=32.170159`; held-out K/V reconstruction relative L2 `0.319477/0.313302`; all/tail mixer relative L2 `0.237558/0.236841`; all/tail cosine `0.973784/0.974106`.
- **Interpretation:** replacing Gaussian calibration with real text changes the conclusion materially. BKV improves all-token mixer relative L2 by approximately `27.37%` over plain activation PCA and clears the post-pilot cosine target. The frozen MLA reference recipe is therefore fold-1 RoRoPE + BKV + rank 448 + one 128-element positional component, for 576 cached elements per token per retained attention layer.
- **Raw artifact:** `results/EXP-018-qwen3-real-text-kv-probe-layer0.json`.

### EXP-019 — Deployable RoRoPE-BKV reference mapping

- **Implementation commit title:** `feat: add deployable Qwen3 RoRoPE-BKV mapping`
- **Status:** completed on the real Qwen3 layer-0 checkpoint on Kaggle GPU; all implementation gates passed.
- **Purpose:** turn the EXP-018 oracle-style activation probe into a normal Flax attention module with frozen Qwen projections, Q/K normalization, learned RoRoPE rotations, BKV scale, and rank-448 joint basis.
- **Explicit cache contract:** `kv_latent [batch, tokens, 448]` plus `k_rope [batch, tokens, 128]`; total 576 elements/token/layer versus 2048 for Qwen3 GQA, a `71.875%` element-count reduction.
- **Correctness invariant:** at full joint rank, the mapped module agrees with the independently implemented uncompressed RoRoPE path within `3e-4` absolute/relative tolerance and emits finite cache tensors of the declared shapes.
- **Real-checkpoint gate:** layer-0 output relative L2 at most `0.30`, cosine at least `0.95`, finite output/cache, and exact cache shapes on the disjoint 1024-token WikiText evaluation prefix.
- **Numerical controls:** FP32 diagnostic execution, high matmul precision, explicit finite checks, and a pinned Qwen3/WikiText source.
- **Boundary:** this is a deployable cache-producing correctness reference. It still computes the original K/V projections and reconstructs K/V before attention. Projection absorption, incremental decoding, and an optimized kernel are separate later experiments; this commit does not claim their speedup.
- **Regression suite (MEASURED):** 35 tests passed in 55.49 s after adding the mapping and cache-shape invariant.
- **Mapping result (MEASURED):** BKV ratio `32.170153`; calibration joint reconstruction relative L2 `0.256885`; source/target cache `2048/576` elements per token, or `71.875%` fewer elements.
- **Mixer result (MEASURED):** all/tail relative L2 `0.237558/0.236841`; all/tail cosine `0.973784/0.974106`; all-token max absolute error `0.752632`.
- **Runtime contract (MEASURED):** `kv_latent=[1,1024,448]`, `k_rope=[1,1024,128]`, finite arrays, exact cache shapes, and the `relative_l2 <= 0.30` / cosine `>= 0.95` engineering gates all passed under JAX `0.7.2` on GPU.
- **Probe-to-module agreement:** deployable all-token relative L2 differs from EXP-018 by approximately `3.8e-8`, showing that the measured BKV behavior survived extraction into the normal module path rather than depending on probe-only arithmetic.
- **Raw artifact:** `results/EXP-019-qwen3-deployable-mla-parity-layer0.json`.

### EXP-020 — Mamba-3 MIMO mathematical reference parity

- **Implementation commit title:** `test: add Mamba-3 reference parity fixtures`
- **Reference:** official `state-spaces/mamba` source at immutable commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`; the independent oracle directly implements the published one-token MIMO recurrence in eager PyTorch rather than calling the JAX implementation.
- **Status:** local and Kaggle cross-framework recurrence parity completed; accelerator-specific parity against the optimized TileLang/CuTe kernels remains a separate test.
- **Discovered defect:** the bring-up JAX path incorrectly applied `tanh` to the predicted angle increment. Official Mamba-3 accumulates `pi * angle_projection * dt` directly. The fixture deliberately includes angle projections outside `[-1,1]` so the old implementation fails decisively.
- **Contract correction:** the module now stores the official inverse-softplus `dt_bias` directly and uses official MIMO parameter shapes `[heads, rank, head_dim]` with names `mimo_x`, `mimo_z`, and `mimo_o`. This is intentionally corrected before any recovery checkpoint exists.
- **Parity result (MEASURED):** deterministic FP32 fixture with batch 2, length 7, 3 heads, MIMO rank 2, state size 8, partial complex rotation, nontrivial trapezoidal gates, decay, skip, and MIMO projections: max absolute error `2.91e-11`, mean absolute error `1.03e-12`, RMSE `3.78e-12`, relative L2 `7.66e-8` against eager PyTorch.
- **Regression suite (MEASURED):** 37 tests passed in 58.46 s after the correction.
- **Kaggle confirmation (MEASURED):** the focused fixture passed `2/2` tests in 11.61 s with exit code 0. The emitted `jupyter_client` UTC deprecation warnings are external notebook warnings and do not affect numerical results.
- **Boundary:** this establishes the recurrence and parameter contract, not numerical equivalence to the fused production kernel, training throughput, long-context stability, or superiority of MIMO. Those require accelerator fixtures and controlled ablations.

### EXP-021 — Controlled Qwen3-to-Mamba-3 initialization shock

- **Implementation commit title:** `feat: add Qwen3-to-Mamba3 transplant baselines`
- **Status:** completed on the real Qwen3 layer-0 checkpoint on Kaggle GPU; all variants executed finitely, but none preserved useful immediate mixer alignment.
- **Input:** pinned real WikiText token prefix represented by actual Qwen3 embedding rows. The default first measurement uses layer 0, sequence length 128, FP32, and seed 123.
- **Shared control:** every variant starts from the same canonical Mamba-3 random parameter tree. Stability/dynamics fields `z`, `dt`, `A`, `trap`, and `angle` remain identical unless a later named ablation explicitly changes one.
- **INIT-A:** fully random Mamba-3 mixer.
- **INIT-B:** INIT-A plus resized Qwen output projection only.
- **INIT-C:** prior-method Q/K/V/O port following the official Mamba-in-the-Llama correspondence `V->x`, `K->B`, `Q->C`, `O->out`, adapted deterministically to the Mamba-3 widths; Q/K norm scales are also preserved.
- **INIT-D:** head-pooled K/Q state projections copied identically into all four MIMO channels. This is the SISO-information/copy baseline.
- **INIT-E:** disjoint source Q/K head groups initialize distinct MIMO channels. INIT-D versus INIT-E isolates MIMO allocation while holding every other field fixed.
- **Metrics:** clean mixer max/mean absolute error, RMSE, relative L2, cosine, output-norm ratio, decoder-layer parity after the shared residual/MLP, and finite checks.
- **Pass criterion:** this first probe only requires finite execution. No quality ranking is predeclared; the measured result decides whether the MIMO mapping deserves subsequent complex-state and recovery ablations.
- **Regression suite (MEASURED):** 38 tests passed in 60.45 s; all five tiny transplant variants also execute through the full Mamba-3 module with finite outputs.
- **Prior-code pin:** `jxiw/MambaInLlama@b03f123152eeba5f2ae9d8694f4a001147e0a14c`.
- **Boundary:** deterministic width adaptation and head grouping are initialization hypotheses, not functional attention-to-SSM equivalences. One layer and 128 tokens cannot establish full-model recovery quality.
- **INIT-A result (MEASURED):** mixer relative L2 `1.000352`, cosine `-0.005826`, norm ratio `0.021356`; decoder relative L2 `1.532735`, cosine `0.614348`.
- **INIT-B result (MEASURED):** mixer relative L2 `0.999703`, cosine `0.024421`, norm ratio `0.023194`; decoder relative L2 `1.531408`, cosine `0.615262`.
- **INIT-C result (MEASURED):** mixer relative L2 `1.000205`, cosine `0.013098`, norm ratio `0.037219`; decoder relative L2 `1.523208`, cosine `0.614811`.
- **INIT-D result (MEASURED):** mixer relative L2 `1.003062`, cosine `0.010646`, norm ratio `0.089688`; decoder relative L2 `1.460898`, cosine `0.616681`.
- **INIT-E result (MEASURED):** mixer relative L2 `1.000674`, cosine `0.010512`, norm ratio `0.048717`; decoder relative L2 `1.515896`, cosine `0.614798`.
- **Interpretation:** the best mixer cosine is only `0.024421` and every candidate is severely under-scaled. INIT-D's larger norm gives the best decoder relative L2, but its mixer direction remains uncorrelated with the teacher. INIT-E does not beat INIT-D consistently, so distinct MIMO allocation is not supported by this initialization alone. Complex-angle ablation is deferred: changing recurrence details cannot be interpreted while the base features/readout carry almost no teacher-aligned signal.
- **Decision:** next run a frozen-feature calibrated-readout probe on disjoint real-text calibration/evaluation tokens. This separates “the recurrence features contain no transferable signal” from “the copied output projection cannot read that signal.” Only a held-out improvement justifies deeper transplant refinements.
- **Raw artifact:** `results/EXP-021-qwen3-mamba3-shock-layer0.json`.

### EXP-022 — Frozen-feature calibrated Mamba-3 readout

- **Implementation commit title:** `feat: add Mamba-3 calibrated readout probe`
- **Status:** completed on the real Qwen3 layer-0 checkpoint on Kaggle GPU; frozen features are readout-recoverable, but the proposed static transplants underperform random Mamba features and the raw-hidden control.
- **Question:** do frozen Mamba-3 recurrent features contain teacher-aligned information that the copied Qwen output projection simply cannot decode, or is the information absent before the readout?
- **Protocol:** process one contiguous real-text prefix, fit only an uncentered linear readout on the first 256 tokens, and evaluate on the next disjoint 128 tokens. Every recurrent parameter remains frozen.
- **Regularization selection:** for each feature variant, fit candidate relative ridge values `1e-2/1e-3/1e-4` on the first 192 calibration tokens, select using the following 64 calibration tokens, then refit on all 256. The final 128 evaluation tokens never influence selection.
- **Variants:** all EXP-021 INIT-A through INIT-E recurrent features, plus normalized raw hidden states as a linear-control representation.
- **Primary metrics:** original versus calibrated held-out mixer relative L2 and cosine. Calibration-fit error is diagnostic only and cannot establish generalization.
- **Decision rule:** a material held-out improvement shows that readout alignment, not merely recurrence construction, is a bottleneck. Failure despite near-zero calibration error indicates feature overfit/no transferable signal and motivates short layerwise distillation rather than more static Q/K/V slicing.
- **Regression suite (MEASURED):** 39 tests passed in 67.24 s. A synthetic linear system is recovered on held-out data, and existing Mamba parity/transplant/full-model tests remain green.
- **Boundary:** 256 calibration tokens intentionally test few-shot recoverability; they are not a proposed final calibration budget. Ridge readout is a diagnostic and not yet the recovery-training method.
- **Regularization result (MEASURED):** every Mamba variant selected relative ridge `1e-2` on the internal validation split. Lower ridge values drove calibration error toward zero while degrading validation, confirming substantial overfit pressure and validating the nested selection protocol.
- **INIT-A/B result (MEASURED):** the two variants have identical recurrent features and therefore identical calibrated output: held-out relative L2 `0.495559`, cosine `0.876858`, versus original relative L2 approximately `1.0006/1.0003` and cosine approximately `-0.0050/0.0056`.
- **INIT-C result (MEASURED):** calibrated held-out relative L2 `0.584426`, cosine `0.838907`.
- **INIT-D result (MEASURED):** calibrated held-out relative L2 `0.587043`, cosine `0.839339`.
- **INIT-E result (MEASURED):** calibrated held-out relative L2 `0.583205`, cosine `0.840404`.
- **Raw-hidden control (MEASURED):** held-out relative L2 `0.388996`, cosine `0.921752`, outperforming every frozen Mamba representation.
- **Interpretation:** a learned readout recovers substantial held-out teacher alignment from random Mamba features, proving that the copied output projection was a major failure point in EXP-021. However, every Q/K/V transplant degrades recoverability relative to random features, so INIT-C/D/E are rejected in their current form. The stronger raw-hidden control means this experiment alone does not prove that the recurrence captures useful contextual information; much of layer-0 attention output is linearly predictable from the current token representation.
- **Decision:** before layerwise distillation, test incremental contextual value by fitting a raw-hidden baseline and then fitting frozen Mamba features only to its residual on disjoint tokens. Mamba is useful only if the combined held-out prediction beats the raw-hidden control. This prevents mistaking a random-feature readout for successful attention transplantation.
- **Raw artifact:** `results/EXP-022-qwen3-mamba3-readout-probe-layer0.json`.

### EXP-023 — Residualized Mamba-3 contextual-value probe

- **Implementation commit title:** `feat: add residualized Mamba-3 context-value probe`
- **Status:** completed on Kaggle GPU; frozen/static transplant branch rejected under this protocol.
- **Question:** after fitting the strongest raw-hidden linear baseline, can frozen Mamba recurrence features predict additional held-out teacher residual?
- **Nested protocol:** select the raw readout ridge using only the internal calibration validation suffix. For each Mamba variant, fit a context readout to the residual left by the raw model, select its ridge on that same held-in validation protocol, then refit both stages on all 256 calibration tokens. The final 128 tokens remain untouched.
- **Comparison:** raw-hidden held-out metrics versus `raw prediction + Mamba residual prediction` for INIT-A through INIT-E. Improvement must occur on held-out relative L2 and cosine, not merely calibration error.
- **Decision rule:** if no variant beats raw hidden, frozen recurrence adds no demonstrated contextual value and the next justified step is short layerwise distillation of recurrent parameters. If a variant wins, retain it as the initialization for that distillation ablation.
- **Regression suite (MEASURED):** 40 tests passed in 66.70 s. A synthetic two-source target verifies that the residualized procedure improves held-out prediction when context features contain complementary signal.
- **Boundary:** sequential residual fitting is deliberately interpretable but is not guaranteed to equal a jointly optimized two-block ridge model. It measures incremental signal under the declared fitting rule.
- **Raw-hidden baseline (MEASURED):** held-out relative L2 `0.388996`, cosine `0.921752`, max absolute error `0.999713`, RMSE `0.0357064`.
- **INIT-A/B (MEASURED):** both select context ridge `1e-2` and produce identical raw-plus-Mamba held-out relative L2 `0.412084`, cosine `0.911153`. Their recurrent features are identical; only the unused original output projection differs.
- **INIT-C (MEASURED):** raw-plus-Mamba held-out relative L2 `0.416555`, cosine `0.909114`.
- **INIT-D (MEASURED):** raw-plus-Mamba held-out relative L2 `0.421239`, cosine `0.906976`.
- **INIT-E (MEASURED):** raw-plus-Mamba held-out relative L2 `0.420634`, cosine `0.907251`.
- **Result:** no frozen Mamba variant improves both held-out criteria; every variant is worse than raw hidden. The least harmful random features still increase relative L2 by `0.023088` and reduce cosine by `0.010599`. Current Q/K/V transplants degrade the residual predictor further.
- **Interpretation:** under one layer-0 WikiText prefix with 256 calibration and 128 held-out tokens, frozen Mamba recurrence features demonstrate no incremental contextual signal beyond the raw-token linear control. This rejects the tested static mapping/probe, not attention-to-Mamba transplantation in general.
- **Decision:** stop static slicing, complex-angle, and frozen-readout tuning. Proceed to a controlled short layerwise distillation pilot in which recurrent parameters are trainable. Retain random initialization and prior QKVO initialization as mandatory baselines under identical data, optimizer, and token budgets.
- **Raw artifact:** `results/EXP-023-qwen3-mamba3-context-probe-layer0.json`.

## 7. Development milestones

1. **Completed only for the sharding mechanism:** wrong-generation weights + Lion state fit on v5e-8; exact Qwen3 HBM validation remains pending.
2. **Completed:** set the Qwen3 target schedule to exactly 6/40 MLA layers.
3. **Completed:** validate the pinned Qwen3-14B metadata, target shapes, and streaming mapping contracts.
4. **Completed:** implement the exact Qwen3 GQA teacher and establish decoder-layer parity against the official PyTorch implementation.
5. **Completed for the frozen reference direction:** establish fold-1 RoRoPE + BKV rank-448 conversion diagnostics and a cache-producing Flax mapping; optimized decode remains pending.
6. **Completed for the mathematical reference path:** establish cross-framework Mamba-3 MIMO recurrence parity and align the module parameter contract with the official implementation.
7. **Completed:** implement controlled attention-to-Mamba-3 transplant variants and establish that direct Q/K/V/O reuse alone does not preserve layer-0 mixer alignment.
8. **Completed:** show that frozen random Mamba features are readout-recoverable, while current Q/K/V transplants hurt and raw hidden states remain the strongest linear control.
9. **Completed:** frozen residualized Mamba features add no held-out signal beyond raw hidden under EXP-023; the static transplant branch is closed.
10. **Current:** run a short, controlled layerwise Mamba-3 distillation pilot before attempting a full-model training step.
11. Recovery training, fixed evaluation checkpoints, and failure logging.
12. Optimized inference kernel and matched end-to-end benchmarks.
13. Only after base recovery: separate reasoning SFT study using legally and scientifically documented data.

### EXP-024 — Trainable Mamba-3 layerwise distillation pilot

- **Implementation commit title:** `feat: add Mamba-3 layerwise distillation pilot`
- **Status:** completed on Kaggle 2xT4 (single-device pilot execution), FP32.
- **Question:** does a small, fixed token budget of gradient-based mixer distillation recover held-out Qwen3 attention behavior, and does the prior QKVO transplant improve recovery speed over random recurrent initialization?
- **Controlled comparison:** `INIT-A-random` versus `INIT-C-prior-qkvo-port`. Both receive their own ridge-calibrated output projection, then all Mamba parameters are trained with identical Lion hyperparameters, data windows, seed, and token budget.
- **Leakage control:** calibration, training, and held-out evaluation use disjoint fixed-length WikiText windows. Evaluation targets never select the readout, learning rate, or checkpoint.
- **Primary endpoint:** change in held-out mixer relative L2 from pre-distillation to the final fixed step. Cosine similarity and the complete loss/gradient-health curve are secondary diagnostics.
- **Numerics:** auto dtype uses FP32 on T4 because its BF16 backward path was empirically non-finite; TPU v5e uses BF16 parameters, compute, gradients, and Lion momentum.
- **Boundary:** this is a one-layer feasibility and initialization-ranking experiment, not evidence that the 14B hybrid has recovered. Positive results justify a larger multi-layer/token-budget sweep; negative results trigger objective/architecture diagnosis before full-model training.
- **Budget (MEASURED):** layer 0, sequence length 32, 256 calibration tokens, 640 disjoint training tokens per variant, 128 held-out tokens, 20 Lion steps at learning rate `3e-5`, seed 123.
- **INIT-A random recurrence (MEASURED):** held-out relative L2 improves `0.886174 -> 0.783387` (11.60% relative reduction), while cosine increases `0.561092 -> 0.624184`. All gradients are finite.
- **INIT-C prior QKVO recurrence (MEASURED):** held-out relative L2 improves `0.903962 -> 0.836949` (7.41% relative reduction), while cosine increases `0.508162 -> 0.582952`. All gradients are finite.
- **Controlled ranking:** random recurrence is already better after readout calibration and remains better after the identical training budget: post-training L2 advantage `0.053561`, post-training cosine advantage `0.041232`. Thus the tested V-to-x, K-to-B, Q-to-C port provides negative transfer rather than faster recovery.
- **Interpretation:** this result rejects only the current direct QKVO-to-Mamba mapping at layer 0 under a 640-token pilot. It does not establish that all Mamba-3 initializations should be random, nor that the rest of Qwen should be discarded. Qwen embeddings, norms, MLPs, residual stream, and retained attention layers remain directly reusable; the unresolved question is how to initialize and recover the new recurrent mixers.
- **Decision:** retain random recurrence as the mandatory control and current leading initializer. Before committing the 14B recovery run, test whether the ranking persists across multiple layers, seeds, and longer fixed token budgets; any proposed learned/transformed transplant must beat this control.
- **Raw artifact:** `results/EXP-024-qwen3-mamba3-distill-pilot-layer0.json`.

### EXP-025 — Paired seed and token-budget distillation sweep

- **Implementation commit title:** `feat: add paired Mamba-3 distillation sweep`
- **Status:** layer-0 measurement completed in a Kaggle 2xT4 allocation; experiment code is unsharded and the reported backend is GPU/FP32.
- **Question:** is the EXP-024 random-over-QKVO ranking reproducible across paired initialization seeds, and does it persist as the distillation budget grows?
- **Primary design:** three paired seeds (`123,456,789`), random recurrence versus prior QKVO recurrence, held-out measurements at steps 0, 20, and 80 on the same evaluation windows. Both variants receive independent ridge-calibrated output projections and identical Lion schedules.
- **Fixed-evaluation rule:** the evaluation set begins after the maximum 80-step training region, so every checkpoint is compared on exactly the same untouched tokens. Step 20 is an intermediate checkpoint of the 80-step schedule rather than a separately tuned run.
- **Primary endpoint:** paired per-seed relative-L2 difference at step 80, supported by mean, sample standard deviation, cosine, and win count. Individual seed measurements remain in the artifact.
- **Execution order:** establish replication on layer 0 first. Only then run the unchanged protocol on representative middle and late Mamba layers, avoiding a costly broad sweep of a failed setup.
- **Failure safety:** a partial JSON is updated after every completed seed/variant pair.
- **Kaggle artifact policy:** every partial and final JSON is written both to the requested path and to `/kaggle/working/output/` for explicit notebook-output collection.
- **Random recurrence aggregate (MEASURED):** mean held-out relative L2 is `0.771632 +/- 0.041263` at step 0, `0.675836 +/- 0.039105` at step 20, and `0.597025 +/- 0.010472` at step 80. Mean cosine at step 80 is `0.803737 +/- 0.006396`; mean relative-L2 reduction from step 0 is 22.52%.
- **Prior QKVO aggregate (MEASURED):** mean held-out relative L2 is `0.830772 +/- 0.037966` at step 0, `0.766275 +/- 0.026702` at step 20, and `0.687220 +/- 0.025226` at step 80. Mean cosine at step 80 is `0.741243 +/- 0.013787`; mean relative-L2 reduction from step 0 is 17.24%.
- **Paired result (MEASURED):** random recurrence wins all three seeds at steps 0, 20, and 80. Mean random-minus-QKVO relative-L2 differences are `-0.059140`, `-0.090439`, and `-0.090195`, respectively. The transplant does not catch up with the larger pilot budget; its disadvantage grows by step 20 and remains through step 80.
- **Numerical result (MEASURED):** all six 80-step runs complete with finite gradients and outputs. The pass flag is true.
- **Decision:** reject the current direct V-to-x, K-to-B, Q-to-C transplant as the recovery initializer for layer 0. Random recurrence becomes the leading control, but `0.597025` mean relative L2 after 2,560 tokens is still far from layer recovery and is not a deployable result.
- **Scientific boundary:** the evidence is strong for this layer-0 protocol but does not establish a universal random-initialization rule. Middle/late-layer experiments require true Qwen residual-stream activations from all preceding layers; raw embeddings are invalid there.
- **Next method step:** implement a real teacher-activation cache for arbitrary layers, then use activation-driven layerwise distillation as the transplant mechanism. Direct matrix reuse remains a negative ablation rather than the proposed method.
- **Raw artifact:** `results/EXP-025-qwen3-mamba3-distill-sweep-layer0.json`.

### EXP-026 — Streaming Qwen3 teacher-activation cache

- **Implementation commit title:** `feat: add streaming Qwen3 activation cache`
- **Status:** layer-0 cache validation completed on TPU v5e-8; deeper-layer execution remains gated on cached-input consumer validation.
- **Purpose:** create scientifically valid mixer inputs for arbitrary Qwen3 layers by streaming the frozen teacher through every preceding decoder layer. This removes the invalid assumption that raw embeddings can serve as middle/late-layer residual inputs.
- **Cached contract:** independent-window token IDs, residual input immediately before the target decoder layer, input-RMSNorm output consumed by its attention mixer, and the exact frozen attention output target. Shapes, dtypes, byte counts, SHA-256 hashes, source revisions, split boundaries, and visible devices are recorded in a manifest.
- **Memory strategy:** only one teacher layer is materialized on-device at a time; activations move through bounded host/device microbatches. A partial manifest and residual checkpoint are updated after every completed decoder layer.
- **Disk strategy:** `--prune-consumed-shards` may delete only exact safetensors shard files whose final required layer has completed. This is optional for early layers and intended for deep targets under Kaggle disk limits.
- **Numerical policy:** T4 uses FP32 teacher compute and FP32 cache storage by default. The cache performs inference only and never updates teacher parameters.
- **Execution gate:** first reproduce a finite layer-0 cache with the EXP-025 window layout. Only after that artifact passes will layer 18 be propagated and used for a cached-input distillation experiment.
- **Layer-0 result (MEASURED):** `passed=true`; compute dtype BF16, storage dtype FP32, microbatch 4, split windows calibration/training/evaluation `8/80/4`, total 92, sequence length 32. All eight TPU v5e devices are visible, while the current correctness path executes unsharded on one device.
- **Artifact contract (MEASURED):** token IDs are `[92,32]`; residual input, normalized input, and attention target are each `[92,32,5120]` FP32 and 60,293,248 bytes including the NPY header. Each artifact has a distinct recorded SHA-256 digest.
- **Interpretation:** the cache producer can materialize the exact layer-0 mixer dataset without training and persist it independently of the checkpoint session. The next gate is consuming these artifacts in the paired distillation sweep and reproducing the qualitative EXP-025 ranking without an on-the-fly teacher forward.
- **Raw artifact:** `results/EXP-026-qwen3-layer0-activation-cache-manifest.json`.

### EXP-027 — Portable cached-activation distillation consumer

- **Implementation commit title:** `feat: add cached-activation Mamba-3 distillation`
- **Status:** layer-0 cache-consumer validation completed on Kaggle GPU/FP32.
- **Purpose:** run the paired Mamba-3 seed/budget sweep from persisted teacher activations without recomputing embeddings or the teacher attention forward. The same consumer will accept layer-18 caches once their true residual stream exists.
- **Integrity gate:** resolve artifacts either from their recorded paths, an explicit cache directory, or the manifest directory; require the producer pass flag; verify every SHA-256, shape, and dtype; and reject mismatched source revision, layer index, sequence length, or split layout.
- **Portability:** manifests produced in `/kaggle/working/output` remain usable after Kaggle publishes them under a different `/kaggle/input/...` directory because artifacts are also resolved by manifest-sibling basename.
- **Validation target:** cached BF16-teacher/FP32-storage layer-0 data should reproduce the qualitative EXP-025 result: finite training and random recurrence beating the direct QKVO port across paired seeds. Exact metrics need not equal the earlier T4/FP32-teacher run.
- **Producer used for the measured consumer run:** FP32 teacher compute and FP32 cache storage. The manifest/artifact paths resolve from `/kaggle/working/output`; all integrity checks pass before training.
- **Reproduction result (MEASURED):** cached-input metrics reproduce EXP-025 nearly numerically. Across all 18 seed/variant/checkpoint comparisons, maximum absolute relative-L2 difference is `2.25e-6` and maximum cosine difference is `3.88e-6`.
- **Scientific result (MEASURED):** random recurrence again wins 3/3 seeds at steps 0, 20, and 80. At step 80, random versus QKVO mean relative L2 is `0.597026` versus `0.687221`, with cosine `0.803737` versus `0.741241`; all runs are finite.
- **Decision:** portable cached activations are accepted as a faithful replacement for on-the-fly layer-0 teacher computation. Deeper-layer distillation may now use this interface, provided the producer first propagates the true residual stream through all preceding frozen layers.
- **Raw artifact:** `results/EXP-027-qwen3-mamba3-cached-distill-sweep-layer0.json`.

### EXP-028 — Data-parallel teacher activation propagation

- **Implementation commit title:** `feat: add data-parallel Qwen3 activation caching`
- **Status:** layer-0 multi-device smoke completed on Kaggle 2xT4; layer-18 propagation authorized on the same topology.
- **Purpose:** use every visible local TPU/GPU for the expensive frozen-teacher propagation while preserving the exact per-window cache contract validated in EXP-026/027.
- **Parallel rule:** parameters are replicated, windows are sharded across the leading `pmap` device axis, and only the final global batch is zero-padded before valid outputs are restored in original order. `--per-device-windows` controls bounded activation memory.
- **Correctness boundary:** this changes throughput only. Each window remains an independent causal sequence, the teacher is frozen, and cache storage/hashing are unchanged.
- **Execution gate:** validate target layer 0 with `--data-parallel` on the available accelerator topology, then propagate target layer 18 with shard pruning enabled.
- **Hardware result (MEASURED):** JAX GPU backend, two visible T4 devices, FP32 compute/storage, `data_parallel=true`, two windows per device, and `execution=data-parallel pmap across 2 devices`.
- **Correctness result (MEASURED):** `passed=true`; split remains `8/80/4`, all artifact shapes match EXP-026, and token/residual SHA-256 values match the prior FP32 source inputs. Normalized/target artifacts are finite and independently hashed.
- **Decision:** the two-device propagation path is accepted for the next layer-18 cache run. TPU-v5e-8 pmap remains a separate topology smoke if that accelerator is chosen instead.
- **Raw artifact:** `results/EXP-028-qwen3-layer0-dp-activation-cache-manifest.json`.

### EXP-029 — True residual-stream activation cache at layer 18

- **Implementation basis:** `feat: add streaming Qwen3 activation cache` plus `feat: add data-parallel Qwen3 activation caching`.
- **Status:** completed on Kaggle 2xT4; cached layer-18 distillation pending.
- **Execution (MEASURED):** frozen Qwen3 layers 0 through 17 are propagated in FP32 using data-parallel `pmap` across two GPUs with two windows/device. Target layer is 18; split is `8/80/4` independent windows of length 32.
- **Artifact result (MEASURED):** `passed=true`; token IDs are `[92,32]`, while residual input, normalized input, and layer-18 attention target are each `[92,32,5120]` FP32 with independent SHA-256 digests.
- **Disk result (MEASURED):** pruning is enabled and exactly checkpoint shards 1 through 4 of 8 are removed after their final required layer. No recursive deletion is used.
- **Scientific meaning:** unlike the earlier layer-0 experiments, the mixer input now contains the true Qwen residual stream after 18 frozen decoder transformations. This makes the next random-versus-QKVO comparison a valid middle-layer transplant ablation.
- **Boundary:** this is still a 2,944-token, windowed activation dataset and not a full-model or long-context recovery result.
- **Raw artifact:** `results/EXP-029-qwen3-layer18-dp-activation-cache-manifest.json`.

### EXP-030 — Layer-18 cached-activation Mamba-3 distillation

- **Status:** completed on Kaggle GPU/FP32 using the EXP-029 portable activation cache.
- **Protocol:** true layer-18 normalized residual inputs and attention targets; paired seeds `123/456/789`; random recurrence versus prior QKVO port; checkpoints 0/20/80; 2,560 unique training tokens per seed/variant; identical Lion settings and held-out windows.
- **Random recurrence (MEASURED):** mean held-out relative L2 `0.945156 +/- 0.031969` at step 0, `0.812879 +/- 0.007577` at step 20, and `0.766070 +/- 0.009172` at step 80. Step-80 cosine is `0.643446 +/- 0.011054`; relative-L2 reduction is 18.91%.
- **Prior QKVO port (MEASURED):** mean held-out relative L2 `0.982541 +/- 0.030771` at step 0, `0.901970 +/- 0.048148` at step 20, and `0.788970 +/- 0.014350` at step 80. Step-80 cosine is `0.617078 +/- 0.017457`; relative-L2 reduction is 19.68%.
- **Paired ranking (MEASURED):** random wins 3/3 seeds at every checkpoint. Mean random-minus-QKVO L2 is `-0.037385` at step 0, widens to `-0.089092` at step 20, then contracts to `-0.022901` at step 80.
- **Depth comparison:** the direct port remains decisively harmful at layer 0 through step 80, but at layer 18 it recovers faster late in training and nearly closes the gap. This supports a depth-dependent delayed-transfer hypothesis rather than a universal random-initialization conclusion.
- **Decision:** do not select either initializer for full recovery yet. Run a longer layer-18 budget with fixed untouched evaluation data to determine whether QKVO crosses random or merely converges to a worse asymptote. Both current endpoints (`0.766/0.789` L2) remain far from recovered attention behavior.
- **Raw artifact:** `results/EXP-030-qwen3-mamba3-cached-distill-sweep-layer18.json`.

### EXP-031 — Extended layer-18 delayed-transfer test

- **Status:** completed on Kaggle TPU/BF16 with an independently produced FP32 activation cache.
- **Protocol:** true layer-18 residual inputs; 160 unique training windows (5,120 tokens) plus disjoint `8` calibration and `4` evaluation windows; checkpoints 0/20/80/160; paired seeds `123/456/789`; identical Lion schedule per variant.
- **Random recurrence (MEASURED):** mean held-out relative L2 `1.001585`, `0.947928`, `0.844232`, and `0.812722` at steps 0/20/80/160. Step-160 cosine is `0.595412 +/- 0.011582`; relative-L2 reduction is 18.82%.
- **Prior QKVO port (MEASURED):** mean held-out relative L2 `1.104935`, `1.050423`, `0.930425`, and `0.885929`. Step-160 cosine is `0.541207 +/- 0.025396`; relative-L2 reduction is 19.75%.
- **Paired ranking (MEASURED):** random wins 3/3 seeds at every checkpoint. Mean random-minus-QKVO L2 contracts monotonically from `-0.103351` at step 0 to `-0.073207` at step 160, but no crossover occurs.
- **Interpretation:** QKVO shows a small relative convergence-rate advantage only because it starts from a materially worse representation. At the fixed 5,120-token budget it remains worse in absolute L2, cosine, and variance. The delayed-transfer hypothesis is not supported strongly enough to justify the direct full-strength port as a compute-efficient initializer.
- **Cross-experiment boundary:** absolute metrics cannot be directly compared with EXP-030 because this run changes the held-out prefix, maximum-step learning-rate schedule, teacher compute path, and token budget. The paired within-run conclusion is valid.
- **Decision:** close `INIT-C-prior-qkvo-port` as the proposed initializer while retaining it as a negative baseline. The next initializer study must reduce transplantation shock, for example by variance matching and controlled interpolation with the canonical random base, and must beat random under the same cache/budget protocol.
- **Raw artifact:** `results/EXP-031-qwen3-mamba3-cached-distill-sweep-layer18-160step.json`.

### EXP-032 — Shock-matched QKVO initializer screen

- **Implementation commit title:** `feat: add shock-matched Qwen-to-Mamba initializers`
- **Status:** one-seed layer-0 screen completed on TPU/FP32; paired confirmation pending.
- **Question:** does the direct QKVO port fail because resized Q/K/V projections have an incompatible scale, or because their directions are intrinsically unhelpful to the new Mamba-3 recurrence?
- **Controlled arms:** `INIT-A-random`; rejected full-strength `INIT-C-prior-qkvo-port`; `INIT-F-variance-matched-qkvo`, whose x/B/C slices independently match the RMS of the same seed's canonical random slices; and 25%/50% random-to-matched interpolation arms `INIT-G`/`INIT-H`.
- **Invariant parameters:** all arms start from the same seed-specific random Mamba base. Only x/B/C slices differ; z, dt, A, trap, angle, convolution, and other recurrence parameters remain identical. Output projection calibration and optimizer protocol remain shared by the distillation harness.
- **Protocol (MEASURED):** seed `123`, layer 0, sequence length 32, eight calibration windows, 20 disjoint training windows (640 tokens), and four fixed held-out windows. Five arms use the same random base; all runs are finite and `passed=true`.
- **Step-0 ranking (MEASURED, relative L2 / cosine):** 25% blend `0.865031 / 0.583476`; random `0.886175 / 0.561088`; 50% blend `0.891340 / 0.547838`; variance-matched full port `0.903903 / 0.508205`; direct port `0.903960 / 0.508160`.
- **Step-20 ranking (MEASURED, relative L2 / cosine):** 25% blend `0.747730 / 0.664355`; 50% blend `0.782923 / 0.625394`; random `0.783389 / 0.624181`; variance-matched full port `0.809520 / 0.603549`; direct port `0.836946 / 0.582954`.
- **Screening signal:** the 25% blend is 4.55% lower in held-out relative L2 than random at step 20 and improves its own L2 by 13.56% from step 0, versus 11.60% for random. The 50% arm is effectively tied with random at step 20; RMS matching alone improves the direct port but remains worse than random.
- **Numerical caution:** the winning 25% arm has the largest single observed gradient-norm spike (`35.29` at step 5), although every gradient leaf remains finite. Confirmation must retain finite-gradient diagnostics.
- **Decision:** advance only `INIT-G-vm-qkvo-blend-0.25` against `INIT-A-random` to the pre-registered three-seed, 80-step layer-0 confirmation. Do not advance INIT-F/H or the direct port.
- **Interpretation boundary:** interpolation is a targeted mechanism ablation, not automatic layer allocation and not evidence of full-model recovery.
- **Artifact note:** the producer's free-text notes incorrectly say “both variants” and refer to an 80-step schedule; the structured fields correctly record five variants and a 20-step maximum. Metrics and ranking are unaffected, and the generator is corrected before confirmation.
- **Raw artifact:** `results/EXP-032-qwen3-mamba3-shock-matched-screen-layer0.json`.

### EXP-033 — Three-seed confirmation of the 25% QKVO blend

- **Status:** completed on TPU/FP32; the 25% blend is rejected as the final layer-0 initializer at the primary 80-step endpoint.
- **Protocol:** paired `INIT-A-random` versus `INIT-G-vm-qkvo-blend-0.25`; seeds `123/456/789`; checkpoints 0/20/80; 80 independent training windows (2,560 tokens) and four fixed held-out windows after the maximum-budget region. All runs are finite and `passed=true`.
- **Random aggregate (MEASURED):** mean held-out relative L2 is `0.771630 +/- 0.041263`, `0.675836 +/- 0.039106`, and `0.597028 +/- 0.010470` at steps 0/20/80. Step-80 cosine is `0.803735 +/- 0.006395`; relative-L2 reduction is 22.52%.
- **25% blend aggregate (MEASURED):** mean held-out relative L2 is `0.791486 +/- 0.029547`, `0.658893 +/- 0.028574`, and `0.604759 +/- 0.000882` at steps 0/20/80. Step-80 cosine is `0.796979 +/- 0.000785`; relative-L2 reduction is 23.52%.
- **Paired trajectory (MEASURED):** random wins `3/3` seeds at step 0; the 25% blend wins `3/3` at step 20 with mean random-minus-blend L2 `+0.016943`; random wins `2/3` at step 80 with mean difference `-0.007731`.
- **Interpretation:** a weak QKVO directional prior reproducibly accelerates early recovery, but the advantage is transient under this schedule. By the pre-registered primary endpoint, it is 1.30% worse than random in mean relative L2 and also worse in cosine. Lower across-seed variance for the blend does not compensate for its worse mean endpoint.
- **Numerical observation:** every gradient remains finite, but the blend reaches larger maximum per-run gradient norms for seeds 123 and 789 (`35.29` and `36.26`) than their random controls (`20.00` and `24.74`).
- **Decision:** retain the 25% blend as an early-recovery ablation, not the production initializer. Do not escalate it directly to full-model recovery. The next transplant method must optimize a fixed-budget endpoint rather than only the first 20 updates; learning a gate or staged decay of the prior is a separate hypothesis requiring a new protocol.
- **Boundary:** this result covers layer 0, sequence length 32, and 2,560 adaptation tokens per seed. It does not establish the behavior of middle layers, long context, or the complete hybrid.
- **Raw artifact:** `results/EXP-033-qwen3-mamba3-shock-matched-confirm-layer0.json`.

### EXP-034 — Pre-registered long-horizon 25% blend crossover test

- **Status:** completed on TPU/FP32; the 25% blend failed the pre-registered primary endpoint and the fixed-interpolation family is closed.
- **Question:** is the step-20 benefit and step-80 loss of the 25% QKVO blend transient, or does the blend regain an absolute advantage with a four-times-larger layerwise adaptation budget?
- **Arms and pairing:** `INIT-A-random` versus `INIT-G-vm-qkvo-blend-0.25`, paired within seeds `123/456/789` from the same canonical random base.
- **Data protocol:** layer 0, sequence length 32, eight calibration windows, 320 unique training windows, and 16 fixed held-out windows placed after the complete training region. This is 10,240 training tokens per arm/seed and 61,440 optimizer-visible tokens across all six runs.
- **Optimization protocol:** FP32 compute, Lion learning rate `3e-5`, zero weight decay, checkpoints 0/20/80/160/320, one shared 320-step schedule per run, and unchanged readout ridge `1e-2`.
- **Primary endpoint:** paired held-out relative L2 at step 320. Advancement requires INIT-G to have at least 1% lower aggregate relative L2 than random and win at least two of three paired seeds, with all gradients finite.
- **Secondary endpoints:** cosine similarity, paired differences at earlier checkpoints, relative recovery from step 0, across-seed variance, and maximum gradient norm. Earlier checkpoints are trajectory diagnostics and cannot override a failed primary endpoint.
- **Random trajectory (MEASURED):** mean relative L2 at steps 0/20/80/160/320 is `0.863258`, `0.682863`, `0.594978`, `0.567076`, and `0.541133`. Final cosine is `0.841112 +/- 0.001733`; relative recovery from step 0 is 37.23%.
- **25% blend trajectory (MEASURED):** mean relative L2 is `0.876450`, `0.676399`, `0.655294`, `0.606816`, and `0.572112`. Final cosine is `0.821638 +/- 0.001944`; relative recovery is 34.70%.
- **Paired result (MEASURED):** the blend wins two of three seeds at step 20 by a small mean margin (`random-minus-blend = +0.006464`). Random then wins all three seeds at steps 80, 160, and 320. At the primary endpoint, `random-minus-blend = -0.030980`; blend L2 is 5.72% worse than random.
- **Numerical result (MEASURED):** all six runs reach step 320 with finite gradients and `passed=true`. The largest recorded gradient norm is `36.26` for the blend versus `24.74` for its corresponding random control.
- **Cross-experiment boundary:** checkpoints 20/80 from this 320-step optimizer schedule and later held-out prefix are not numerically interchangeable with EXP-033. Only within-EXP-034 paired comparisons are causal.
- **Decision:** the pre-registered gate fails decisively: INIT-G is neither 1% better nor a two-seed winner. Close fixed 25% QKVO interpolation as a production initializer and retain it only as evidence that a weak prior can briefly accelerate the earliest updates. Do not spend a layer-18 cache regeneration on this arm.
- **Scale boundary:** 10,240 tokens per run is four times EXP-033 but remains a layerwise diagnostic, not evidence about billion-token Extent-14B recovery.
- **Raw artifact:** `results/EXP-034-qwen3-mamba3-long-horizon-layer0.json`.

### EXP-035 — Transient QKVO prior screen

- **Implementation commit title:** `feat: add decaying QKVO transplant prior`
- **Status:** one-seed TPU/FP32 screen completed; the pre-registered gate failed and the transient schedule is closed.
- **Hypothesis:** the 25% QKVO direction contains useful information during the first updates, but retaining it in the parameter basin harms later recovery. A decaying external offset may preserve the early benefit while allowing the trained base to converge like canonical random initialization.
- **INIT-I parameter contract:** the trainable parameter tree starts from `INIT-A-random`. The difference `INIT-G - INIT-A` is applied only to x/B/C input-projection regions and copied B/C norm scales; `out_proj` is explicitly excluded and independently ridge-calibrated. The offset scale decays linearly from 1 to 0 over the first 20 completed updates and is exactly zero thereafter.
- **Screen protocol:** seed `123`; layer 0; sequence length 32; eight calibration, 160 unique training, and 16 fixed held-out windows; checkpoints 0/20/80/160; FP32; arms INIT-A random, static INIT-G, and transient INIT-I on the same random base.
- **Primary gate:** at step 160, INIT-I must have at least 1% lower held-out relative L2 than INIT-A with finite gradients. Static INIT-G is a negative mechanism control. Earlier checkpoints describe the trajectory but cannot override the primary endpoint.
- **Random result (MEASURED):** relative L2 at steps 0/20/80/160 is `0.843208`, `0.691656`, `0.613662`, and `0.587294`; final cosine is `0.809420`.
- **Static blend result (MEASURED):** relative L2 is `0.847944`, `0.683095`, `0.655684`, and `0.623839`; final cosine is `0.781934`.
- **Transient result (MEASURED):** relative L2 is `0.847944`, `0.722410`, `0.615610`, and `0.590025`; final cosine is `0.807443`. Recorded prior scale is exactly `1.0` at step 0 and `0.0` at steps 20/80/160.
- **Mechanism result:** removing the offset eliminates most of the static blend's long-horizon damage: at step 160 transient improves L2 over static by `0.033814`. It does not create a benefit over random; transient remains `0.002732` (0.47%) worse in L2 and also worse in cosine.
- **Numerical result:** all three runs are finite and `passed=true`. Maximum gradient norms are `20.00` random, `35.28` static, and `28.84` transient.
- **Decision:** INIT-I fails the frozen 1%-improvement gate. Do not run the three-seed/320-step promotion and do not tune decay duration post hoc. Close direct, variance-matched, fixed-blend, and transient QKV-to-Mamba parameter priors as production initializer candidates. Retain their early-step behavior as negative/mechanistic ablations.
- **Next method boundary:** subsequent transplantation work must derive Mamba parameters or checkpoints from teacher activation matching, not another hand-designed raw Q/K/V matrix correspondence.
- **Boundary:** this is a parameter-schedule ablation for one replacement mixer. It neither initializes recurrent dynamics from Qwen nor establishes full-model recovery.
- **Raw artifact:** `results/EXP-035-qwen3-mamba3-transient-prior-screen-layer0.json`.

### EXP-036 — Resumable offline Mamba-3 activation distillation

- **Implementation commit title:** `feat: add resumable offline Mamba distillation`
- **Status:** accelerator resume smoke and the frozen 1,024-step endpoint completed successfully on Colab TPU v5e-1.
- **Purpose:** separate frozen-Qwen inference from Mamba optimization so the 14B teacher is never resident beside the trainable student and Kaggle/Colab sessions can stop and resume safely.
- **Producer contract:** the existing streaming activation-cache producer writes normalized mixer inputs and exact Qwen attention targets with shapes, dtypes, split bounds, source revision, and SHA-256 hashes. Teacher weights are needed only in this producer session.
- **Consumer contract:** `scripts/offline_mamba_distill.py` opens verified NPY artifacts with memory mapping, initializes only one Mamba-3 mixer, calibrates its readout on the frozen calibration split, and trains against cached targets. It does not instantiate Qwen attention or download a Qwen checkpoint.
- **Resume contract:** every atomic checkpoint contains Mamba parameters, complete Lion optimizer state, completed update count, payload byte size/hash, and an exact compatibility record covering data hashes, pinned source, target layer, Mamba config, dtype, seeds, batch size, schedule, and learning rate. Incompatible resumes fail before training.
- **Data-order contract:** each optimizer step is mapped to a deterministic shuffled per-epoch cache stream using only the completed step and fixed data seed. No implicit NumPy RNG state is required, so a restarted process consumes the exact next batch.
- **Local verification:** checkpoint round-trip and corruption/compatibility gates pass; a four-step Lion trajectory is exactly identical between uninterrupted execution and a two-step plus save/reload plus two-step execution.
- **First accelerator gate:** create a modest layer-0 cache, release the teacher state, then run the student through multiple bounded invocations with `--resume`. Acceptance requires finite metrics, monotonically advancing checkpoint steps, identical checkpoint compatibility, and a final `status=complete` result.
- **Scientific boundary:** this experiment validates infrastructure and resumability, not recovery quality. A later pre-registered data-scale experiment must use disjoint held-out text, report unique and optimizer-visible tokens separately, and compare recovery against the established random-initialization curve.
- **Accelerator protocol (MEASURED):** Colab TPU v5e-1, BF16 teacher and student compute, FP16 cache storage, layer 0, sequence length 32, 1,024 training windows (32,768 unique tokens), 16 disjoint evaluation windows, and a fixed 1,024-step Lion schedule. The first two 64-step invocations exercise the resume path before a final continuation from step 128 to 1,024.
- **Final trajectory (MEASURED):** held-out relative L2 is `0.785725`, `0.636486`, `0.599482`, `0.566117`, `0.547762`, `0.537660`, and `0.537300` at steps 0/64/128/320/512/896/1024. Cosine rises from `0.684613` to `0.843673`; relative L2 falls by 31.62% over exactly one pass through 32,768 unique training tokens.
- **Numerical result (MEASURED):** `status=complete`, `passed=true`, and every recorded gradient is finite. Final training loss is `0.254569`, gradient norm `0.557311`, and maximum absolute gradient `0.059570`. The final Mamba-plus-Lion checkpoint is 250,738,397 bytes with SHA-256 `d0ab8d317cb2a0130d02ab49da7bfe8ed0b2f673f9222eaff12ece617017aea7`.
- **Plateau observation:** improvement slows materially after step 512. Relative L2 improves only 1.91% from step 512 to 1,024 and 0.067% from step 896 to 1,024 under the decaying fixed schedule.
- **Interpretation:** offline producer/consumer separation, checkpoint integrity, Lion restoration, deterministic continuation, and a complete student-only TPU run are validated. The final `0.537300` mixer relative L2 is a meaningful reduction but not functional equivalence to Qwen attention; scaling the identical sequence-32 recipe without a downstream replacement test is not justified.
- **Next gate:** preserve the completed mixer checkpoint and inject it into a frozen Qwen decoder-layer evaluation. Compare original attention, canonical random Mamba, readout-calibrated step-0 Mamba, and the step-1,024 offline checkpoint on decoder-output error and language-model loss shock. This determines whether lower mixer L2 transfers through the residual/MLP path before spending compute on longer context or more layers.
- **Raw artifact:** `results/EXP-036-qwen3-mamba3-offline-distill-resume-smoke-layer0.json`.

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

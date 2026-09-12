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

### EXP-037 — Downstream decoder shock from the recovered Mamba mixer

- **Implementation commit title:** `feat: add recovered Mamba decoder shock evaluation`
- **Status:** completed on Colab TPU v5e-1; recovered Mamba wins both controls, but the pre-registered 20% primary-improvement gate fails.
- **Question:** does the step-1,024 offline mixer checkpoint reduce functional error after Qwen's residual addition, post-attention RMSNorm, and frozen SwiGLU MLP, or is its mixer-level improvement absorbed or distorted by the decoder tail?
- **Frozen evaluation data:** the 16 EXP-036 evaluation windows, disjoint from its eight readout-calibration and 1,024 training windows. The cache source revision, normalized-input hash, attention-target hash, sequence length, layer index, dtype, and checkpoint compatibility contract must all match before evaluation.
- **Controlled arms:** `INIT-A-random` is the exact seed-123 canonical Mamba initialization; `INIT-A-readout-calibrated-step0` differs only by the ridge-fitted output projection used at the start of EXP-036; `OFFLINE-recovered-step1024` restores the complete final mixer parameters. All arms share the exact cached residual inputs and identical frozen Qwen post-attention norm/MLP weights.
- **Targets:** mixer output is compared with cached frozen-Qwen attention output. Decoder output is reconstructed by applying the same frozen Qwen residual/MLP tail once to the teacher attention target and once to each replacement output.
- **Primary endpoint:** held-out decoder-output relative L2. The recovered checkpoint must beat both controls and improve decoder L2 by at least 20% relative to the readout-calibrated step-0 arm, with finite outputs. Mixer relative L2 must reproduce the EXP-036 step-0/final ordering within the same cache/dtype path.
- **Secondary endpoints:** decoder cosine similarity, mixer relative L2/cosine, and the fraction of mixer improvement retained after the decoder tail. No language-model-quality claim is permitted from this single-layer diagnostic.
- **Decision rule:** passing justifies preserving the activation-trained checkpoint as the current layer-0 transplant and advancing to an end-to-end loss-shock evaluation. Failure means mixer MSE is an inadequate recovery objective and must be changed before scaling tokens, context, or layer count.
- **Mixer result (MEASURED):** relative L2 / cosine is `1.000087 / 0.002387` for canonical random, `0.785725 / 0.684613` for readout-calibrated step 0, and `0.537300 / 0.843673` for the recovered step-1,024 checkpoint. The recovered metrics exactly reproduce the EXP-036 endpoint on the same held-out cache.
- **Decoder result (MEASURED):** relative L2 / cosine is `1.457133 / 0.648357` random, `0.730949 / 0.726604` calibrated step 0, and `0.609322 / 0.806318` recovered. Every output is finite and the execution pass flag is true.
- **Primary-gate result:** recovered decoder L2 improves 16.64% over calibrated step 0 and 58.18% over canonical random. It beats both controls but misses the frozen minimum 20% improvement over calibrated step 0; therefore the scientific gate fails.
- **Mechanism interpretation:** approximately 52.6% of the relative mixer-L2 improvement over calibrated step 0 survives after the nonlinear frozen decoder tail. Activation matching transfers downstream, but less strongly than required, and final decoder L2 `0.609322` remains far from functional equivalence.
- **Decision:** retain the step-1,024 checkpoint as evidence that trainable activation transfer works, not as an accepted production transplant. Before end-to-end or multi-layer scaling, change the objective to include the frozen decoder-tail output (mixer loss plus decoder-output loss) and compare it against mixer-only distillation under the same cache and token budget.
- **Raw artifact:** `results/EXP-037-qwen3-mamba3-decoder-shock-layer0.json`.

### EXP-038 — Decoder-aware offline Mamba distillation

- **Implementation commit title:** `feat: add decoder-aware Mamba distillation`
- **Status:** completed on Colab TPU v5e-1; the pre-registered primary gate passes.
- **Question:** can training through the frozen Qwen decoder tail recover more downstream behavior than mixer-only activation matching under exactly the same initialization, data order, optimizer, and token budget?
- **Controlled arms:** `MIXER-ONLY` minimizes mixer relative MSE and exactly reproduces the EXP-036 objective. `JOINT-MIXER-DECODER` minimizes the normalized mean of mixer relative MSE and full decoder-output relative MSE with decoder-loss weight 1.0. Both start from the same seed-123 readout-calibrated parameters and consume the same deterministic window at every update.
- **Frozen path:** cached Qwen attention output supplies the offline mixer target. Teacher and replacement mixer outputs pass through identical frozen Qwen residual addition, post-attention RMSNorm, and SwiGLU MLP weights. Only Mamba parameters receive gradients.
- **Data and schedule:** reuse the verified EXP-036 layer-0 cache: eight calibration, 1,024 training, and 16 evaluation windows of length 32; one pass (32,768 unique tokens per arm); BF16; Lion learning rate `3e-5`; checkpoints 0/128/256/512/1024; seeds 123 and 20260820.
- **Primary endpoint:** held-out decoder-output relative L2 at step 1,024. The joint arm must improve at least 10% over mixer-only, all values must remain finite, and joint mixer-output relative L2 may degrade by no more than 10% versus mixer-only.
- **Secondary endpoints:** decoder cosine, mixer relative L2/cosine, trajectory crossover, maximum gradient norm, and component training losses. Step-0 outputs must be identical across arms.
- **Decision rule:** passing promotes decoder-aware activation training as the new layerwise transplant objective and justifies an end-to-end loss-shock experiment. Failure closes this equal-weight formulation; any different weight or schedule requires a new pre-registered ablation rather than post-hoc tuning.
- **Control reproduction (MEASURED):** mixer-only exactly reproduces EXP-036/037 at the shared checkpoints. Its step-1,024 mixer relative L2 / cosine is `0.537300 / 0.843673`, and decoder relative L2 / cosine is `0.609322 / 0.806318`.
- **Joint result (MEASURED):** step-1,024 mixer relative L2 / cosine is `0.551648 / 0.836725`, while decoder relative L2 / cosine is `0.545319 / 0.846451`. Relative to the identical step-0 state, joint training reduces decoder L2 by 25.40% and mixer L2 by 29.79%.
- **Primary-gate result:** joint decoder L2 is 10.504% lower than mixer-only, narrowly exceeding the frozen 10% requirement. Joint mixer L2 is 2.670% worse than mixer-only, safely inside the maximum 10% degradation. All gradients and outputs are finite; `scientific_gate_passed=true` and `passed=true`.
- **Trajectory (MEASURED):** joint decoder-L2 improvements over mixer-only grow from 5.64% at step 128 to 8.76%, 9.85%, and 10.50% at steps 256/512/1024. The advantage is therefore sustained and increases with the fixed budget rather than arising from a single early checkpoint.
- **Numerical observation:** maximum gradient norm is `9.1107` joint versus `15.8252` mixer-only, a 42.43% reduction. Final component training losses remain balanced (`0.260418` mixer and `0.271634` decoder), with combined loss `0.266026`.
- **Decision:** accept equal-weight decoder-aware activation matching as the current layer-0 transplant method. Do not tune its coefficient on this evaluation split. Advance to a frozen, end-to-end Qwen language-model loss-shock experiment that compares original attention, readout-calibrated step 0, mixer-only step 1,024, and joint step 1,024.
- **Boundary:** this is one layer, one seed, length 32, and a single 32,768-token pass. Passing establishes a useful objective, not full-model recovery, long-context quality, or inference speed.
- **Raw artifact:** `results/EXP-038-qwen3-mamba3-decoder-aware-layer0.json`.

### EXP-039 — Three-seed decoder-aware confirmation

- **Implementation commit title:** `feat: add multi-seed decoder-aware confirmation`
- **Status:** completed on Colab TPU v5e-1; the pre-registered multi-seed confirmation gate passes.
- **Question:** is the EXP-038 decoder-aware advantage reproducible across initialization seeds, or was its narrow 10.504% gate pass specific to seed 123?
- **Pairing:** seeds `123/456/789`. Within each seed, mixer-only and equal-weight joint arms start from the identical seed-specific random recurrence plus independently fitted readout, receive the exact same deterministic training-window order, and share all frozen Qwen tail parameters. Across seeds, data, optimizer schedule, and evaluation windows remain fixed.
- **Data and compute:** the verified EXP-038 cache with eight calibration, 1,024 training, and 16 evaluation windows of length 32; one 32,768-token pass per arm; six optimizer runs and 196,608 optimizer-visible tokens total; BF16; Lion `3e-5`; checkpoints 0/128/256/512/1024.
- **Primary confirmation gate:** mean paired decoder-L2 improvement of joint over mixer-only at step 1,024 must be at least 10%; joint must win at least two of three paired seeds; mixer-output L2 degradation must not exceed 10% in any seed; step-0 paired outputs must be identical; and every gradient/output must be finite.
- **Secondary endpoints:** per-seed trajectories, mean and population standard deviation of endpoint decoder L2 and paired improvement, decoder cosine, mixer degradation, and maximum gradient norm. EXP-038 seed 123 is rerun inside this harness rather than copied into the aggregate.
- **Decision rule:** passing establishes the equal-weight decoder-aware objective as the reproducible layer-0 method and authorizes a streamed end-to-end language-model loss-shock experiment. Failure keeps EXP-038 as a one-seed positive result and blocks full-model escalation until a newly pre-registered objective is tested.
- **Per-seed endpoint (MEASURED):** mixer-only versus joint decoder relative L2 is `0.609322 → 0.545319` for seed 123, `0.612050 → 0.540556` for seed 456, and `0.608451 → 0.544558` for seed 789. Paired improvements are 10.504%, 11.681%, and 10.501%, respectively; joint wins all three seeds.
- **Aggregate endpoint (MEASURED):** mixer-only decoder L2 is `0.609941 +/- 0.001534`; joint is `0.543478 +/- 0.002089`. Mean paired improvement is 10.895% with population standard deviation 0.556 percentage points, exceeding the frozen 10% requirement.
- **Mixer tradeoff (MEASURED):** joint mixer-L2 degradation is 2.670%, 3.222%, and 3.889% across seeds; the maximum 3.889% is well below the per-seed 10% limit. The mean degradation is 3.260%.
- **Correctness result:** step-0 paired outputs are identical for all seeds; every gradient and evaluation output is finite; all three single-seed execution flags pass; aggregate `scientific_gate_passed=true` and `passed=true`.
- **Decision:** equal-weight decoder-aware activation matching is accepted as the reproducible layer-0 transplant objective. The next experiment must measure whether this local advantage survives frozen propagation through the complete Qwen3-14B network and improves end-to-end next-token loss relative to mixer-only and calibrated step-0 controls.
- **Scale boundary:** confirmation covers initialization variance but still uses one data sample, layer 0, sequence length 32, and one 32,768-token pass per arm. It does not establish robustness across layers, corpora, context lengths, or full-model recovery.
- **Raw artifact:** `results/EXP-039-qwen3-mamba3-multiseed-decoder-aware-layer0.json`.

### EXP-040 — Streamed end-to-end language-model loss shock

- **Implementation commit title:** `feat: add streamed end-to-end Qwen loss shock`
- **Status:** completed on Kaggle TPU v5e-8; the pre-registered end-to-end gate passes.
- **Question:** does the reproducible local advantage of equal-weight decoder-aware Mamba distillation survive the remaining 39 frozen Qwen3-14B decoder layers and reduce end-to-end next-token loss?
- **Frozen branch protocol:** four layer-0 outputs are evaluated on the same 16 untouched windows: `ORIGINAL-CACHED-QWEN`, reconstructed from the cached exact Qwen attention target and frozen layer-0 tail; `CALIBRATED-STEP0`; `MIXER-ONLY-STEP1024`; and `JOINT-STEP1024`. The two trained arms start from the same seed-123 calibrated parameters and consume the same 1,024-window order used by EXP-038.
- **Data and optimization:** pinned Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18`; layer 0; sequence length 32; eight readout-calibration, 1,024 training, and 16 held-out evaluation windows; 32,768 unique training tokens per arm; Lion learning rate `3e-5`; BF16 compute; seeds 123 and 20260820; checkpoints 0/512/1024.
- **End-to-end evaluator:** concatenate all four branches, then stream them through identical frozen Qwen decoder layers 1–39 one layer at a time. Materialize only the current decoder layer, transfer its output back to bounded host storage, release its arrays, and optionally delete checkpoint shards after their exact last use. Apply the pinned final RMSNorm and language-model head on device and return only summed NLL, label count, and top-1 hits; full vocabulary logits never return to the host.
- **Primary endpoint:** mean within-window next-token NLL over 496 held-out labels per branch. The joint arm must have lower NLL than both mixer-only step 1,024 and calibrated step 0, and must recover at least 10% of mixer-only excess NLL above the original branch. Mixer-only excess NLL must be positive and all branch statistics must be finite.
- **Secondary endpoints:** perplexity and top-1 accuracy; relative hidden-state L2 to the original branch after layers 0, 9, 19, 29, and 39; exact evaluated-label count; and bounded-memory/shard-pruning execution diagnostics. Secondary metrics cannot override the primary gate.
- **Decision rule:** passing promotes the joint decoder-aware objective from a layerwise transplant metric to the current end-to-end layer-0 recovery method and justifies multi-seed/full-depth confirmation before multi-layer replacement. Failure means the local decoder-tail gain does not survive full frozen propagation at this budget; longer training or a different objective must be separately pre-registered rather than inferred post hoc.
- **Interpretation boundary:** the original branch is a deterministic reconstruction from the stored Qwen layer-0 attention output, not a second independent full-model forward pass. All branches nevertheless share the exact same cached inputs, frozen tail, downstream layers, final norm, tokens, and loss implementation. This experiment covers one replaced layer, one seed, length 32, and one corpus prefix; it cannot establish the final 85% Mamba hybrid quality or long-context behavior.
- **Execution result (MEASURED):** all four branches remain finite through frozen layers 1–39 and the final language-model head; `passed=true`. The evaluator uses all eight TPU v5e devices in data parallel, returns exactly 496 labels per branch, and removes all eight Qwen checkpoint shards after their final use.
- **Language-model result (MEASURED):** mean NLL / perplexity / top-1 accuracy is `3.835861 / 46.33 / 37.30%` for original cached Qwen, `6.973308 / 1067.75 / 13.10%` for calibrated step 0, `5.510824 / 247.35 / 20.16%` for mixer-only step 1,024, and `4.833501 / 125.65 / 26.01%` for joint step 1,024.
- **Primary-gate result:** mixer-only excess NLL above original is `1.674963`; joint excess is `0.997641`. Joint therefore recovers 40.438% of mixer-only excess NLL, over four times the frozen 10% requirement, while also beating calibrated step 0. Its absolute NLL is 12.291% lower than mixer-only and its perplexity is 49.202% lower; `scientific_gate_passed=true`.
- **Depth trajectory (MEASURED):** relative hidden L2 to original for mixer-only versus joint is `0.609214 / 0.545200` after layer 0, `0.588317 / 0.543131` after layer 9, `0.572599 / 0.526512` after layer 19, `0.568440 / 0.521153` after layer 29, and `0.623004 / 0.521189` after layer 39. Joint remains better at every checkpoint and avoids most of mixer-only's late-depth error amplification.
- **Interpretation:** the decoder-aware advantage is not confined to the local frozen MLP used by its training objective. It persists through the complete downstream Qwen stack and materially improves token prediction. The remaining gap to original Qwen is still large: joint NLL is `0.997641` above original and top-1 accuracy remains 11.29 percentage points lower.
- **Decision:** accept equal-weight decoder-aware activation matching as the current end-to-end layer-0 transplant method. The next confirmatory experiment must evaluate the full-depth NLL advantage across seeds 123/456/789 on the same untouched labels before replacing multiple layers. It should stream all seed branches together so the frozen Qwen layers are downloaded and executed once per combined batch rather than rerunning three independent full-model experiments.
- **Raw artifact:** `results/EXP-040-qwen3-streamed-end-to-end-layer0.json`.

### EXP-041 — Three-seed streamed end-to-end confirmation

- **Implementation commit title:** `feat: add multi-seed streamed end-to-end confirmation`
- **Status:** completed on Kaggle TPU v5e-8; the pre-registered multi-seed full-depth gate passes.
- **Question:** is EXP-040's full-depth language-model advantage reproducible across Mamba initialization seeds, or is its 40.438% excess-NLL recovery specific to seed 123?
- **Paired training protocol:** seeds `123/456/789`; for every seed, calibrated step 0, mixer-only step 1,024, and equal-weight joint step 1,024 use the same seed-specific base, cache, deterministic data order, Lion schedule, and frozen Qwen layer-0 tail. Training reuses the EXP-039 protocol with 32,768 unique tokens per arm and 196,608 optimizer-visible tokens over six trained arms.
- **Combined full-depth protocol:** construct one shared original-Qwen branch plus three branches per seed, concatenate all ten branch/window groups, and stream them together through identical frozen Qwen layers 1–39, final RMSNorm, and language-model head. This downloads each pinned Qwen checkpoint shard once and evaluates every seed in one layerwise pass. Each branch uses the same 16 untouched length-32 windows and 496 within-window labels.
- **Correctness control:** seed 123 must reproduce all four archived EXP-040 mean-NLL values within absolute tolerance `0.02`. The archived `results/EXP-040-qwen3-streamed-end-to-end-layer0.json` is the frozen reference; failure blocks the scientific gate even if the new aggregate appears favorable.
- **Primary confirmation gate:** mixer-only excess NLL above the shared original branch must be positive for every seed; joint must beat mixer-only in at least two of three seeds; joint must beat its calibrated step-0 control in at least two of three seeds; mean paired recovery of mixer-only excess NLL must be at least 10%; the EXP-040 reproduction control and the underlying EXP-039 local multi-seed gate must pass; and all training and inference values must remain finite.
- **Secondary endpoints:** per-seed NLL, perplexity, top-1 accuracy, excess-NLL recovery, and hidden relative L2 after layers 0/9/19/29/39; population standard deviation of paired recovery; maximum gradient norms; and exact shard-pruning diagnostics. Secondary endpoints cannot override the primary gate.
- **Decision rule:** passing establishes the decoder-aware objective as robust to initialization at both local decoder output and complete frozen-language-model depth, authorizing the first controlled multi-layer replacement study. Failure limits EXP-040 to a one-seed result and requires diagnosing the failed seed trajectories before any 15/85 scaling.
- **Interpretation boundary:** this confirmation reuses the same held-out corpus prefix as EXP-038–040. It tests initialization robustness and evaluator reproducibility, not robustness across text samples, layers, context lengths, or datasets. A pass is necessary but not sufficient evidence for an 85% Mamba hybrid.
- **Correctness result (MEASURED):** seed 123 reproduces all four archived EXP-040 NLL values inside the frozen `0.02` tolerance. Absolute differences are `0.002329` original, `0.000916` calibrated, `0.003833` mixer-only, and `0.016553` joint. The larger but passing joint difference is consistent with the changed TPU batch shape and must remain visible rather than being rounded away.
- **Per-seed language-model result (MEASURED):** calibrated / mixer-only / joint mean NLL is `6.974224 / 5.514657 / 4.850055` for seed 123, `7.103628 / 5.480069 / 4.816119` for seed 456, and `7.674675 / 5.434410 / 4.685474` for seed 789. Joint beats both controls for every seed.
- **Primary-gate result:** paired recovery of mixer-only excess NLL is 39.533%, 40.324%, and 46.783% for seeds 123/456/789. Mean recovery is `42.213% +/- 3.247%`, over four times the frozen 10% requirement; joint wins mixer-only `3/3` and calibrated step 0 `3/3`. Every mixer-only excess is positive, all values are finite, the local EXP-039 training confirmation passes again, and `scientific_gate_passed=true`.
- **Aggregate quality (MEASURED):** mixer-only NLL is `5.476379 +/- 0.032864`; joint is `4.783883 +/- 0.070951`. Joint reduces absolute NLL relative to mixer-only by 12.052%, 12.116%, and 13.781% across the three seeds. Mean top-1 accuracy increases from 20.833% to 27.285%.
- **Depth result (MEASURED):** joint has lower hidden relative L2 than mixer-only for all three seeds at every recorded layer. At layer 39, mixer-only / joint L2 is `0.623349 / 0.521281`, `0.631794 / 0.495491`, and `0.642598 / 0.493671` for seeds 123/456/789. Thus the advantage persists through the entire frozen tail and the joint endpoint is also less seed-sensitive at depth.
- **Execution result:** all ten branches complete the shared streamed pass over 39 frozen layers on eight TPU devices; all eight checkpoint shards are removed after their final use. Each branch contributes exactly 496 labels.
- **Decision:** accept decoder-aware activation matching as initialization-robust on the current layer-0 split. Before any multi-layer replacement, run one locked fresh-text confirmation with a materially larger evaluation set that was not used to select the objective or thresholds. Passing that gate authorizes a controlled multi-layer composition study; do not jump directly from one replaced layer to 34 replaced layers.
- **Raw artifact:** `results/EXP-041-qwen3-multiseed-streamed-end-to-end-layer0.json`.

### EXP-042 — Locked fresh-text end-to-end confirmation

- **Implementation commit title:** `feat: add locked fresh-text end-to-end confirmation`
- **Status:** completed on Kaggle TPU v5e-8; numerical execution and the primary full-depth NLL condition pass, but the combined pre-registered gate fails narrowly on the required local decoder-output confirmation.
- **Question:** does the three-seed decoder-aware advantage generalize to a substantially larger text region that was never used to select the initializer, objective, checkpoint, or EXP-040/041 thresholds?
- **Frozen training data:** preserve the exact layer-0 calibration/training prefix and protocol: eight calibration windows followed by 1,024 training windows of length 32, one 32,768-token pass per arm, seeds `123/456/789`, data seed `20260820`, BF16, Lion `3e-5`, readout ridge `1e-2`, and equal decoder-loss weight 1.0. A one-window placeholder may remain in the training-cache manifest but is not used for evaluation when the external cache is supplied.
- **Locked fresh evaluation data:** pinned WikiText-2 train token slice `[65536, 69632)` under the pinned Qwen3 tokenizer, divided into 128 independent length-32 windows. This yields 4,096 input tokens and exactly 3,968 within-window next-token labels per branch, eight times EXP-041's label count. The cache is explicitly marked evaluation-only with empty calibration/training ranges and must be hash-verified.
- **Disjointness gate:** both manifests must match source revision, dataset revision, target layer, and sequence length; both must record token ranges; the fresh range must be disjoint from the complete training-cache range; token offset must equal `65536`; and evaluation window count must equal `128`. Any mismatch stops execution before training.
- **Branches and execution:** retain one shared original-Qwen branch plus calibrated, mixer-only step 1,024, and joint step 1,024 branches for each seed. All ten branches stream together through frozen Qwen layers 1–39 and the final LM head. The EXP-040 numerical-reproduction check is intentionally disabled because the token sample differs.
- **Primary gate:** mixer-only excess NLL above original must be positive for all seeds; joint must beat mixer-only in at least two of three seeds and calibrated step 0 in at least two of three; mean paired recovery of mixer-only excess NLL must be at least **25%**; the fresh-cache local three-seed decoder-output gate must pass; and every value must remain finite. The 25% threshold is frozen before observing this token range and requires retention of a substantial fraction of EXP-041's 42.213% effect.
- **Secondary endpoints:** per-seed and aggregate NLL, perplexity, top-1 accuracy, excess-NLL recovery, population standard deviation, local decoder/mixer L2, hidden relative L2 at layers 0/9/19/29/39, and shard-pruning diagnostics. Secondary metrics cannot rescue a failed primary gate.
- **Decision rule:** passing establishes initialization and fresh-prefix robustness for the layer-0 method and authorizes the controlled multi-layer composition experiment. Failure blocks multi-layer escalation and requires distinguishing local-objective failure from downstream distribution shift without tuning on this locked range.
- **Boundary:** the new range is unseen but remains one contiguous slice of WikiText-2 train and uses short independent contexts. A pass does not establish cross-corpus or long-context generalization; those remain later validation axes.
- **Execution and provenance (MEASURED):** the external manifest records the exact disjoint range `[65536, 69632)`, 128 windows, and 3,968 labels per branch. All ten branches complete on eight TPU devices, every value is finite, all eight checkpoint shards are pruned after final use, and `passed=true`.
- **Local decoder result (MEASURED):** joint improves decoder-output L2 over mixer-only by 10.046%, 10.160%, and 9.777% for seeds 123/456/789; it wins all three seeds. The mean is `9.9945% +/- 0.1606%`, missing the inherited 10.0000% requirement by 0.0055 percentage points. Mixer-L2 degradation is `3.427%` mean and `3.553%` maximum, safely inside the 10% cap. Consequently the fresh local confirmation reports `scientific_gate_passed=false`.
- **Full-depth result (MEASURED):** original mean NLL is `4.320217`. Calibrated / mixer-only / joint NLL is `7.718380 / 5.884605 / 5.375257` for seed 123, `7.568387 / 5.881467 / 5.651797` for seed 456, and `7.942847 / 6.040012 / 5.433232` for seed 789. Joint beats both controls `3/3`.
- **Primary NLL threshold result:** joint recovers 32.559%, 14.711%, and 35.282% of mixer-only excess NLL. Mean recovery is `27.517% +/- 9.124%`, exceeding the frozen 25% requirement. Aggregate NLL falls from `5.935361 +/- 0.074010` mixer-only to `5.486762 +/- 0.119073` joint, a 7.558% mean-NLL reduction. Mean top-1 accuracy rises from 20.153% to 23.429%.
- **Depth result (MEASURED):** joint hidden L2 is below mixer-only for every seed at layers 0 and 39. At layer 39, mixer-only / joint is `0.626723 / 0.525167`, `0.609984 / 0.554094`, and `0.633074 / 0.553972`. The end-to-end benefit therefore persists on fresh text despite the marginal local-gate miss.
- **Formal interpretation:** `aggregate.scientific_gate_passed=true` for the full-depth endpoint, but top-level `scientific_gate_passed=false` because the protocol explicitly required both local and full-depth gates. Do not round `9.9945%` to 10% or retroactively remove the local condition. This is a boundary failure of the combined rule alongside strong positive fresh-text evidence, not a numerical failure and not evidence that joint training is worse.
- **Decision:** honor the pre-registration and do not start multi-layer replacement yet. The next adjudication must use a new unseen split and make end-to-end NLL the sole primary quality endpoint, with local L2 retained as a mechanism diagnostic rather than a veto. Its threshold and sample size must be frozen before reading the new split; EXP-042 remains a failed combined gate regardless of that later outcome.
- **Raw artifact:** `results/EXP-042-qwen3-locked-fresh-text-layer0.json`.

### EXP-043 — Cross-split end-to-end NLL adjudication

- **Implementation commit title:** `feat: add cross-split NLL adjudication`
- **Status:** completed; primary scientific gate passed on Kaggle TPU v5e-8.
- **Question:** does decoder-aware Mamba retain a statistically supported end-to-end language-model advantage on the official WikiText-2 validation split, independently of the train-prefix samples used throughout EXP-036–042?
- **Frozen training protocol:** unchanged layer-0 training on the pinned WikiText-2 train prefix: eight calibration and 1,024 training windows of length 32, seeds `123/456/789`, data seed `20260820`, one 32,768-token pass per arm, BF16, Lion `3e-5`, readout ridge `1e-2`, and decoder-loss weight 1.0. No parameter or objective is tuned using validation data.
- **Cross-split evaluation:** pinned file `wikitext-2-raw-v1/validation-00000-of-00001.parquet` at dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`, token offset 0, 256 independent length-32 windows, 8,192 input tokens, and exactly 7,936 next-token labels per branch. The producer records `dataset_split=validation`; the evaluator requires the training manifest to record `dataset_split=train` and explicitly authorizes only this frozen cross-split transition.
- **Branches:** one shared original-Qwen branch plus calibrated step 0, mixer-only step 1,024, and joint step 1,024 for each seed. All ten branches stream together through frozen Qwen layers 1–39, final RMSNorm, and LM head. Per-window NLL is reduced on device and retained for paired uncertainty estimation; full vocabulary logits never return to host.
- **Bootstrap protocol:** paired percentile bootstrap over the 256 window indices, resampling the same indices for original, mixer-only, and joint branches across all seeds; 2,000 samples; seed `20260823`; 95% interval. Each resample computes per-seed excess-NLL recovery and then the three-seed mean. This procedure and seed are frozen before reading validation outputs.
- **Primary gate:** all mixer-only excess NLL values must be positive; joint must beat mixer-only in at least two of three seeds and calibrated step 0 in at least two of three; point-estimate mean excess-NLL recovery must be at least **20%**; the lower bound of the paired 95% bootstrap interval must be at least **10%**; and all training/inference values must remain finite. The full-depth endpoint is the sole quality gate.
- **Local diagnostic boundary:** decoder-output and mixer-output L2, cosine, gradient norms, and the inherited EXP-039 local gate remain fully reported but cannot veto or rescue EXP-043. This change does not reclassify EXP-042, whose combined gate remains failed; it defines a new criterion on unseen data before observation.
- **Secondary endpoints:** per-seed/aggregate NLL, perplexity, top-1 accuracy, recovery variance, full bootstrap distribution summary, hidden relative L2 at layers 0/9/19/29/39, and shard-pruning diagnostics.
- **Decision rule:** passing authorizes the controlled two-layer composition experiment. Failure blocks multi-layer escalation and requires revisiting the distillation objective or data budget without tuning on WikiText validation. The WikiText test split remains untouched for later final validation.
- **Boundary:** validation is an official disjoint split but belongs to the same WikiText-2 corpus and still uses short independent contexts. Cross-corpus and long-context claims remain out of scope.
- **Execution and provenance (MEASURED):** the run used the pinned Qwen3-14B revision, the frozen official validation range `[0, 8192)`, 256 length-32 windows and exactly 7,936 next-token labels per branch. BF16 execution completed over eight TPU v5e devices; all three seeds and all ten end-to-end branches were finite, and all eight downloaded Qwen shards were removed after use.
- **Per-seed primary result (MEASURED):** calibrated / mixer-only / joint NLL is `7.976333 / 5.717690 / 5.145281` for seed 123, `7.921865 / 5.689392 / 5.166175` for seed 456, and `8.287977 / 5.749470 / 5.129610` for seed 789. Against original-Qwen NLL `3.974949`, joint recovers `32.845%`, `30.518%`, and `34.931%` of mixer-only excess NLL. Joint beats mixer-only and calibrated step 0 in all three seeds.
- **Aggregate primary result (MEASURED):** mixer-only NLL is `5.718851 +/- 0.024540`, joint NLL is `5.147022 +/- 0.014978`, and mean excess-NLL recovery is `32.765% +/- 1.802%`. The frozen 20% point-estimate threshold is passed.
- **Paired uncertainty result (MEASURED):** the pre-registered 2,000-sample paired percentile bootstrap gives mean recovery `32.787%` and a 95% interval of `[30.247%, 35.399%]`. Every resample is finite and the lower bound is far above the frozen 10% requirement. Therefore `scientific_gate_passed=true` without relying on the local diagnostic.
- **Secondary end-to-end result (MEASURED):** mean next-token top-1 accuracy rises from approximately `21.904%` for mixer-only to `26.390%` for joint; original Qwen is approximately `37.929%`. At layer 39, joint hidden relative L2 is `0.535632`, `0.538397`, and `0.540660`, below mixer-only `0.635478`, `0.633172`, and `0.635458` for every seed. The decoder-aware benefit therefore survives the full frozen decoder on an official disjoint split.
- **Mechanism diagnostic (MEASURED; non-vetoing):** local decoder-output improvement is `9.590%`, `9.901%`, and `9.523%`, with mean `9.672% +/- 0.165%`; it does not meet the inherited 10% local threshold. This is reported as a diagnostic exactly as pre-registered and does not alter the EXP-043 gate. Mixer-output degradation remains modest at approximately `3.297%` mean and `3.348%` maximum.
- **Interpretation:** the result resolves the EXP-042 boundary case in favor of a real end-to-end benefit: the decoder-aware objective improves held-out validation NLL for every seed, with a narrow paired confidence interval. It does not prove cross-corpus or long-context generalization, and it does not make the current layer-level Mamba replacement competitive with original Qwen; substantial recovery remains necessary.
- **Decision:** EXP-043 passes every frozen primary condition and authorizes controlled two-layer composition. The next experiment must compare original Qwen, layer-0-only, layer-18-only, and simultaneous layer-0+18 replacement so that non-additive error and input-distribution shift at the second replaced layer are measured before any broad 85% rollout.
- **Raw artifact:** `results/EXP-043-qwen3-cross-split-nll-layer0.json`.

### EXP-044 — Controlled layer-0 + layer-18 Mamba composition

- **Implementation commit title:** `feat: add two-layer composition experiment`
- **Status:** completed on the resumed TPU run. Numerical execution passes, but the frozen scientific gate is `FAIL`: the requirement that both standalone excess-NLL values be positive is violated because the layer-18-only replacement slightly *improves* NLL for every seed. This classification is preserved exactly; the favorable reason for failure does not permit relabeling the pre-registered gate.
- **Question:** when independently recovered Mamba-3 mixers replace Qwen attention at layers 0 and 18, is the end-to-end loss shock approximately the sum of the two standalone shocks, or does the shifted layer-18 input cause a materially super-additive failure?
- **Frozen replacement locations:** layers 0 and 18. They are deliberately separated so that layer 18 receives activations propagated through seventeen frozen Qwen layers after the early replacement. No layer selection is changed after observing EXP-044.
- **Frozen training protocol:** independently train decoder-aware Mamba endpoints for layers 0 and 18 using the pinned WikiText-2 train prefix. Each layer uses eight calibration windows, 1,024 training windows, sequence length 32, seeds `123/456/789`, data seed `20260820`, BF16, Lion `3e-5`, readout ridge `1e-2`, decoder-loss weight 1.0, and exactly one 32,768-token optimizer-visible pass per arm. Mixer-only controls remain recorded, but composition uses the joint endpoint for each layer and seed.
- **Frozen evaluation protocol:** official WikiText-2 validation split, token range `[0, 8192)`, 256 independent length-32 windows and 7,936 next-token labels per branch. Layer-0 and layer-18 caches must contain identical token IDs and matching provenance. The streamed original branch must reproduce the cached layer-18 residual input within relative L2 `0.01` before composition is evaluated.
- **Branches:** one shared original-Qwen branch plus `layer0-only`, `layer18-only`, and `layer0+layer18` for each seed, for ten branches total. At layer 18 the Mamba input RMSNorm is recomputed from each branch's current residual; the composed branch is never evaluated using cached original-Qwen normalized activations.
- **Primary quantity:** for each seed define `E0 = NLL(layer0-only) - NLL(original)`, `E18 = NLL(layer18-only) - NLL(original)`, `Eboth = NLL(layer0+18) - NLL(original)`, and composition inflation `I = Eboth / (E0 + E18)`. `I=1` is additive shock, `I<1` is sub-additive, and `I>1` is super-additive. Interaction NLL `Eboth - E0 - E18` and the marginal amplification of layer 18 after layer 0 are also reported.
- **Uncertainty protocol:** paired percentile bootstrap over the same 256 validation windows for all four branch types and all three seeds; 2,000 samples; seed `20260824`; 95% interval over the three-seed mean composition inflation.
- **Primary gate:** all values must be finite; both standalone excess NLL values must be positive for every seed; mean composition inflation must be at most `1.25`; at least two of three per-seed inflation ratios must be at most `1.50`; and the upper bound of the paired 95% bootstrap interval must be at most `1.50`. These thresholds are frozen before execution.
- **Secondary endpoints:** branch NLL/perplexity/top-1, per-seed interaction NLL, layer-18 marginal amplification, hidden relative L2 at layers 0/17/18/29/39, local training diagnostics for both replacement layers, cache/stream parity, and shard-pruning diagnostics. Secondary metrics cannot rescue a failed primary gate.
- **Decision rule:** passing authorizes a small progressive replacement schedule with input-distribution-aware recovery. Failure blocks broad replacement and requires training the downstream Mamba on upstream-shifted activations or jointly optimizing the two replaced layers before attempting additional layers.
- **Boundary:** EXP-044 tests only one widely separated pair on short WikiText-2 contexts. A pass is evidence that independently trained replacements compose at this depth pair, not proof that 34 Mamba layers will compose, that the best 15% attention allocation is known, or that long-context behavior is preserved.
- **Interrupted-run provenance (MEASURED):** both layer-specific training stages completed on TPU in BF16 for all seeds and both arms, using the frozen 1,024-step protocol and 256 validation windows. Every recorded value is finite. The run then constructed the ten composition branches, applied the layer-18 replacement, and streamed successfully through layer 23 before the remote session stopped progressing. No final LM-head NLL, top-1, interaction ratio, or bootstrap result exists.
- **Layer-0 training replication (MEASURED):** joint decoder-aware training beats mixer-only decoder L2 for all three seeds. Mixer-only / joint decoder L2 is `0.602791 / 0.544963`, `0.602776 / 0.543109`, and `0.602929 / 0.545497`, corresponding to `9.593%`, `9.899%`, and `9.525%` improvement. Mean improvement is `9.6726% +/- 0.1623%`; joint mixer-L2 degradation is `3.2978%` mean and `3.3476%` maximum. The inherited local 10% gate is narrowly missed, consistently with EXP-043, while numerical execution passes.
- **Layer-18 training result (MEASURED):** decoder-aware training does not beat mixer-only for any seed. Mixer-only / joint decoder L2 is `0.00532873 / 0.00534341`, `0.00531660 / 0.00533170`, and `0.00533464 / 0.00534746`, for improvements of `-0.2756%`, `-0.2838%`, and `-0.2402%`; mean improvement is `-0.2666% +/- 0.0189%`. Mixer-only / joint mixer L2 is `0.675836 / 0.677717`, `0.674315 / 0.676206`, and `0.676987 / 0.678491`; joint degradation is only `0.2604%` mean and `0.2805%` maximum. Both arms train stably, but the decoder-aware arm is consistently slightly worse at this layer.
- **Mechanistic interpretation:** the absolute layer-18 decoder-output relative L2 is only about `0.0053` while mixer-output relative L2 remains about `0.675`. The full residual/MLP output is therefore dominated by the inherited residual stream and makes the current decoder-relative-MSE term weakly discriminative for a deep mixer. This is evidence that the layer-0 decoder-aware objective is not depth-invariant; it is not evidence that the layer-18 mixer has recovered end-to-end language-model behavior.
- **Interruption boundary:** endpoint Mamba parameters were held only in process memory and were not serialized, so the streamed pass cannot resume from layer 23 and the completed training cannot be reused from the JSON artifacts alone. The four activation-cache manifests are provenance records but are not sufficient without their large array artifacts.
- **Decision after interruption:** preserve the original EXP-044 thresholds and do not infer a composition outcome. Before rerunning, add atomic endpoint-weight checkpoints plus a stage manifest that separates cache production, replacement training, and streamed evaluation. The rerun must either execute the frozen joint-endpoint EXP-044 unchanged or receive a new experiment ID if the layer-18 objective/endpoint selection is altered.
- **Partial raw artifacts:** `results/EXP-044-layer0-training-interrupted-run.json` and `results/EXP-044-layer18-training-interrupted-run.json`.
- **Recovery implementation:** the follow-up implementation stores each completed layer's multi-seed mixer-only and joint endpoint parameters in an atomic, SHA-256-verified checkpoint containing no optimizer state. Its compatibility contract pins source revision, target layer, train/evaluation manifest hashes, seeds, schedule, dtype, and objective hyperparameters. A durable stage manifest records cache, layer-training, streamed-evaluation, and final-result boundaries. The one-shot runner now verifies and reuses all complete caches and endpoint bundles by default; after a streaming interruption, repeating the same command reruns only streamed evaluation. This is an engineering recovery change and does not alter the frozen EXP-044 scientific protocol or reclassify the interrupted run.
- **Successful-run provenance (MEASURED):** the resumed run completed all 39 streamed decoder layers and the LM-head evaluation on eight TPU devices in BF16. It evaluated 256 validation windows and 7,936 labels per branch. Original-Qwen mean NLL is `3.97742154` (perplexity `53.3792`, top-1 `0.378276`). Cache-to-stream parity at the layer-18 input is `0.00359158` relative L2, below the frozen `0.01` bound. All metrics are finite and all eight downloaded Qwen shards were removed after use.
- **Primary composition result (MEASURED):** mean composition inflation is `0.99532570 +/- 0.00138833`. Per-seed inflation is `0.99664708`, `0.99592263`, and `0.99340737`, so all three seeds satisfy the `1.50` ratio bound. The frozen 2,000-sample paired bootstrap gives mean `0.99536352` and 95% interval `[0.99084652, 1.00003355]`; its upper endpoint is far below `1.50`. There is therefore no evidence of super-additive error for this layer pair.
- **Per-seed decomposition (MEASURED):** for seeds `123/456/789`, layer-0 excess NLL is `1.16282563 / 1.19485953 / 1.14742776`; layer-18 excess NLL is `-0.00854411 / -0.00829536 / -0.00902493`; and joint excess NLL is `1.15041130 / 1.18172612 / 1.13089777`. Interaction NLL is negative in every seed (`-0.00387021 / -0.00483806 / -0.00750507`). Adding the layer-18 replacement after layer 0 changes NLL by `-0.01241432 / -0.01313341 / -0.01652999`, so the composed branch is slightly better than layer-0-only in every seed rather than being amplified by the shifted input distribution.
- **Frozen-gate accounting:** finiteness passes; mean-inflation passes; per-seed inflation passes `3/3`; and the bootstrap-upper-bound condition passes. `all_single_layer_excess_nll_positive` fails because all three layer-18 excess-NLL values are negative. Consequently `scientific_gate_passed=false` even though the experiment answers its mechanistic danger question favorably. Secondary metrics cannot and do not override this outcome.
- **Depth diagnostic (MEASURED):** layer-18-only hidden relative L2 grows from approximately `0.00537` immediately after replacement to approximately `0.0378` at layer 39. Nevertheless, at layer 39 the composed branch has lower hidden relative L2 than layer-0-only for every seed (`0.53072 < 0.53549`, `0.53472 < 0.54227`, and `0.53378 < 0.53918`). This agrees with the negative interaction-NLL result.
- **Exploratory uncertainty diagnostic (POST HOC; non-vetoing):** a separate 10,000-sample paired bootstrap with seed `20260825` estimates the mean standalone layer-18 NLL change as `-0.00862146`, 95% interval `[-0.01227904, -0.00507210]`, and its conditional change after layer 0 as `-0.01402591`, interval `[-0.01889874, -0.00928889]`. These intervals quantify sampling uncertainty on the reused validation windows only; they were not pre-registered, do not cover systematic implementation uncertainty, and do not alter the primary gate. Top-1 is mixed across seeds, so this is not claimed as broad capability improvement.
- **Scientific decision:** reject the hypothesized catastrophic two-layer interaction at the tested early/middle pair, but do not authorize an immediate 85% rollout. The formal gate failed because its positive-standalone-shock assumption was wrong, the current deep-layer decoder-aware objective was weakly discriminative, and only one short-context layer pair has been tested. The next experiment receives a new ID and must compare mixer-only, current decoder-aware matching, and a depth-aware counterfactual-contribution objective at both layers 0 and 18 using fixed held-out end-to-end NLL. This tests whether the transplant objective should change with residual-stream depth before progressive multi-layer scaling.
- **Completed raw artifact:** `results/EXP-044-qwen3-two-layer-composition.json`.

### EXP-045 — Depth-aware counterfactual contribution objective

- **Implementation commit title:** `feat: add depth-aware transplant objective experiment`
- **Status:** completed numerically on Kaggle TPU v5e-8; the pre-registered scientific gate fails at both layers. Earlier interrupted attempts remain recorded below as engineering provenance.
- **Question:** does removing the residual-stream baseline produce a transplant objective that predicts and improves held-out end-to-end language-model behavior at both an early and a deep decoder layer?
- **Frozen layers and controls:** layers 0 and 18; seeds `123/456/789`. Each seed begins from the same seed-specific ridge-calibrated Mamba parameters and consumes the same deterministic batch order in three arms: `MIXER-ONLY`, `JOINT-MIXER-DECODER`, and `COUNTERFACTUAL-CONTRIBUTION`.
- **New objective:** let `T(r,m)` be the frozen Qwen decoder tail applied to residual input `r` and mixer output `m`. The teacher contribution is `T(r,m_teacher) - T(r,0)` and the student contribution is `T(r,m_mamba) - T(r,0)`. The counterfactual arm minimizes the normalized mean of contribution relative MSE and mixer-output relative MSE, with weight 1.0 on each term. No Qwen norm, MLP, embedding, or LM-head parameter is trainable.
- **Rationale:** full decoder-output relative error is depth-dependent because the inherited residual stream can dominate `T(r,m)`. Subtracting the same zero-mixer intervention isolates the causal contribution carried through the attention/Mamba path while preserving nonlinear interaction with the frozen decoder tail.
- **Frozen training protocol:** pinned Qwen3-14B revision; WikiText-2 train prefix; sequence length 32; eight calibration windows; 1,024 training windows; exactly 1,024 Lion steps at batch one (`32,768` optimizer-visible tokens per arm); BF16; learning rate `3e-5`; ridge `1e-2`; data seed `20260820`. No arm receives additional tokens or tuning.
- **Frozen evaluation protocol:** official WikiText-2 validation token range `[0,8192)`, 256 length-32 windows and 7,936 labels per branch. For each layer, one original-Qwen branch and all nine objective/seed branches start from the same cached residual input and stream through the identical frozen remainder of Qwen, final norm, and LM head. Layer 18 is evaluated as a standalone replacement, not composed with layer 0.
- **Primary estimands:** paired end-to-end NLL differences `NLL(contribution)-NLL(mixer-only)` and `NLL(contribution)-NLL(decoder-aware)`, separately at layers 0 and 18. Absolute differences are used instead of shock ratios because EXP-044 showed that deep-layer excess NLL can cross zero.
- **Uncertainty protocol:** paired percentile bootstrap over the 256 shared validation windows, averaged across the three paired seeds; 2,000 samples; base seed `20260826` plus target-layer index; 95% intervals reported separately against both controls at each depth.
- **Per-layer scientific gate:** all values finite; the counterfactual arm has lower mean NLL than each control; it beats each control in at least two of three seeds; and the upper endpoint of each paired 95% confidence interval is below zero. The overall EXP-045 gate passes only if every condition passes independently at both layers 0 and 18. A local mixer/decoder relative-L2 result cannot rescue a failed end-to-end gate.
- **Engineering protocol:** the one-shot TPU v5e-8 runner builds or verifies all four caches, executes layer 18 first, then layer 0, and mirrors JSON artifacts into Kaggle output. Completed per-layer results are restart boundaries. Multi-seed endpoint bundles are stored atomically with SHA-256 and compatibility metadata, so an interruption after training requires only streamed evaluation rather than retraining. Large checkpoints do not need to be downloaded to the user's computer.
- **Decision rule:** a pass supports a depth-invariant counterfactual transplant objective and authorizes a progressive multi-layer schedule using it. A layer-0-only pass implies that the residual-subtraction idea does not solve deep recovery. A layer-18-only pass implies that it should be used selectively at depth. Failure at both layers rejects this formulation and motivates direct downstream-logit/Jacobian matching rather than further loss-weight tuning on the same validation range.
- **Boundary:** this experiment selects a recovery objective on short WikiText contexts. It does not establish long-context retention, the final 15% attention allocation, full 34-layer composition, throughput, or downstream capability recovery. The validation split is used for this objective adjudication and cannot later serve as untouched final evidence for the full model.
- **Interrupted-run provenance (MEASURED):** the four activation caches completed. Layer 18 then completed all `3 objectives x 3 seeds x 1,024 steps` in TPU BF16, consuming 294,912 optimizer-visible tokens in total. Every arm remained finite. The atomic endpoint bundle is `1,128,322,561` bytes with SHA-256 `7a037406c88463d00f3a7c441f23ea3e58198f40ad415d9ee55e2df2bca0c847`. Streaming subsequently completed layers 19 and 20 and stopped while acquiring weights for layer 21. No layer-18 LM-head NLL or bootstrap result exists, so the EXP-045 scientific gate has not failed or passed.
- **Layer-18 local diagnostic (MEASURED; non-vetoing):** counterfactual contribution matching lowers held-out contribution relative L2 versus mixer-only by `0.3306% / 0.3115% / 0.3322%` across seeds `123/456/789`, and versus decoder-aware matching by `0.6045% / 0.5936% / 0.5710%`. It also lowers full decoder-output relative L2 in all three seeds, while degrading direct mixer-output relative L2 versus mixer-only by `1.3919% / 1.3895% / 1.3606%`. This is directionally consistent with the mechanism but is not the pre-registered end-to-end endpoint.
- **Legacy local-gate note:** the printed `DECODER-AWARE-CONFIRMATION-GATE-FAIL` applies the inherited EXP-039 test of decoder-aware versus mixer-only matching. It is expected at layer 18 and is not the EXP-045 scientific decision, which compares all three objectives using streamed LM NLL.
- **Disk-failure correction:** the original runner retained earlier Qwen shards during the layer-18 stream in an attempt to reuse them for layer 0. The corrected runner prunes every shard after its last use during cache construction and both end-to-end passes. In `auto` mode it additionally places transient Qwen shards under `/dev/shm` only when that mounted RAM filesystem reports at least 36 GiB free; otherwise it uses bounded disk streaming. Endpoint and stage manifests are now recorded before streamed evaluation begins.
- **Second campaign attempt (ENGINEERING FAILURE; 2026-08-24):** the resilient campaign completed all nine layer-18 LM-head evaluations (`MIXER-ONLY`, `JOINT`, and `CONTRIBUTION` for seeds `123/456/789`) and then failed during Python result aggregation with `KeyError: 'SEED-123-CALIBRATED-STEP0'`. The objective protocol intentionally does not evaluate the legacy calibrated branch; its collector incorrectly retained an unconditional lookup from EXP-041--043. This occurred after numerical inference, not inside TPU training or the model, and therefore provides no scientific pass/fail evidence. The process had not yet written the LM metrics to JSON, so their numerical values are not recoverable from the supplied log. Telegram failure reporting executed successfully (`message_id=108`).
- **Aggregation and durability correction:** objective protocols now collect exactly the branches they execute, while legacy protocols retain the calibrated control. Regression tests cover both schemas. The audit also found and corrected a latent EXP-046/047 failure: every bootstrapped objective protocol now retains per-window NLL. Finally, a complete raw LM-metrics JSON is written immediately after the last branch evaluation and before aggregation, bootstrap, or scientific-gate code, so any later reporting failure leaves the expensive measurements recoverable.
- **Correction commit title:** `fix: align campaign metric aggregation with objective branches`
- **Final layer-0 result (MEASURED):** contribution minus mixer-only mean NLL is `-0.36568988` with paired 95% interval `[-0.40984153, -0.32404156]` and wins `3/3`, but contribution minus joint is `+0.20121843`, interval `[+0.17199723, +0.23020454]`, and wins `0/3`. Per-seed mixer / joint / contribution NLL is `5.713451 / 5.140247 / 5.345404`, `5.688465 / 5.172281 / 5.343346`, and `5.736187 / 5.124849 / 5.352282`. Joint is decisively the best tested layer-0 objective.
- **Final layer-18 result (MEASURED):** contribution is worse than both controls: mean deltas are `+0.00090061` versus mixer-only and `+0.00126629` versus joint, with intervals `[+0.00023200, +0.00152141]` and `[+0.00055571, +0.00199274]`, and `0/3` wins against each. All three replacement arms slightly improve absolute NLL versus the original Qwen branch, so this result concerns objective ranking rather than shock recovery.
- **Scientific decision:** reject counterfactual-contribution matching as a universal transplant objective. At layer 0 it improves strongly over mixer-only but loses strongly to joint matching; at layer 18 it loses significantly to both. `passed=true`, `scientific_gate_passed=false`.
- **Partial raw artifacts:** `results/EXP-045-stage-manifest-interrupted.json`, `results/EXP-045-layer18-endpoint-checkpoint.json`, and `results/EXP-045-layer18-training-interrupted.json`.

### EXP-046 — Additional-depth objective generalization

- **Implementation commit title:** `feat: add resilient TPU experiment campaign`
- **Status:** completed numerically on Kaggle TPU v5e-8; the pre-registered scientific gate fails at both added depths.
- **Question:** does the EXP-045 objective ranking generalize beyond the originally selected early/middle pair, or is its behavior specific to layers 0 and 18?
- **Frozen layers:** layers 9 and 29. Together with EXP-045 layers 0 and 18, these form a four-point depth profile spanning early, early-middle, late-middle, and late decoder positions. These layers are not selected after inspecting new objective results.
- **Training protocol:** identical to EXP-045: sequence length 32, eight calibration windows, 1,024 training windows, seeds `123/456/789`, one 32,768-token pass per arm, BF16 Lion at `3e-5`, ridge `1e-2`, and paired `MIXER-ONLY`, `JOINT-MIXER-DECODER`, and `COUNTERFACTUAL-CONTRIBUTION` arms.
- **Evaluation protocol:** official WikiText-2 validation range `[0,8192)`, 256 length-32 windows and 7,936 next-token labels per branch. Every replacement is evaluated independently from its original-Qwen residual input and streams through the identical frozen suffix, final norm, and LM head.
- **Per-layer gate:** identical to EXP-045: counterfactual matching must have lower mean NLL than both controls, win against each control in at least two of three seeds, and have a paired-bootstrap 95% upper bound below zero against each control. All numerical and provenance checks must pass. Overall EXP-046 passes only if both layers 9 and 29 pass.
- **Decision rule:** a joint EXP-045/046 pass supports a depth-invariant objective. Mixed results motivate an explicit depth-conditioned allocation rule. Failure at the added depths rejects a universal counterfactual objective even if the original pair passes.
- **Layer-9 result (MEASURED):** contribution wins the mean comparison against mixer-only and joint in all three seeds, with mean deltas `-0.00082666` and `-0.00091892`. However, the paired intervals are `[-0.00168275, +0.00002834]` and `[-0.00180149, +0.00001103]`; both upper endpoints narrowly cross zero, so the frozen uncertainty gate fails.
- **Layer-29 result (MEASURED):** contribution wins only `1/3` seeds against each control and is worse on average by `+0.00018787` versus mixer-only and `+0.00015353` versus joint. Both paired intervals cross zero. The layer gate fails.
- **Scientific decision:** the depth sweep rejects a depth-invariant counterfactual objective. Its possible layer-9 advantage is a weak hypothesis for future independent validation, not a confirmed effect. `passed=true`, `scientific_gate_passed=false`.

### EXP-047 — Zero-shot context transfer of recovered endpoints

- **Implementation commit title:** `feat: add resilient TPU experiment campaign`
- **Status:** completed numerically on Kaggle TPU v5e-8; the pre-registered context-transfer gate fails.
- **Question:** do endpoints trained exclusively on length-32 windows preserve their relative objective ranking when evaluated at longer contexts without any additional optimization?
- **Frozen scope:** EXP-045 layers 0 and 18; sequence lengths `64`, `128`, and `256`; all three objective arms and seeds `123/456/789`. EXP-046 layers are excluded to keep the campaign within the cloud-session budget while retaining an early/deep comparison.
- **Token-control protocol:** every context length covers the identical official WikiText-2 validation token range `[0,8192)`: 128 windows at length 64, 64 windows at length 128, and 32 windows at length 256. Thus token count and text are fixed while segmentation and recurrent/attention horizon change. Endpoints receive no context-specific fitting.
- **Primary cell estimands:** mean paired NLL differences of counterfactual matching versus mixer-only and decoder-aware matching for each of the six `layer x context` cells. Counterfactual regret is `NLL(counterfactual) - min(NLL(mixer-only), NLL(decoder-aware))`.
- **Scientific gate:** all six cells must be finite; counterfactual matching must be the best arm in at least five of six cells; and maximum counterfactual regret over all cells must not exceed `0.02` NLL. Per-cell paired bootstrap intervals are reported as uncertainty diagnostics. This gate measures ranking transfer, not absolute equivalence to original Qwen.
- **Boundary:** all contexts remain short relative to the intended final long-context model and reuse the objective-selection validation split. EXP-047 can reject immediate context transfer but cannot establish 32K+ context quality or serve as untouched final evaluation.
- **Result (MEASURED):** contribution is best in only `2/6` required cells versus a threshold of `5/6`; maximum regret is `0.26433590` NLL versus the frozen maximum `0.02`. At layer 0 it loses to joint at lengths `64/128/256` by `+0.24550454 / +0.26433590 / +0.22591893` NLL. At layer 18 all effects are tiny: contribution is best at lengths 128 and 256, but not at 64.
- **Scientific decision:** objective ranking does not transfer uniformly with context segmentation. In particular, longer context does not rescue contribution matching at layer 0. `passed=true`, `scientific_gate_passed=false`.

### EXP-048 — Paired long-horizon transplant scaling

- **Implementation commit title:** `feat: add long-horizon transplant scaling campaign`
- **Status:** completed numerically on Kaggle TPU v5e-8 in `3.504` hours; layer 0 passes its scaling gate, layer 18 fails in the opposite direction, so the overall pre-registered scientific gate fails.
- **Question:** does the ranking or absolute recovery of the three transplant objectives change when unique one-pass recovery data is increased fourfold, rather than merely repeating short 1,024-step probes?
- **Frozen design:** fresh paired runs at layers 0 and 18, seeds `123/456/789`, and arms `MIXER-ONLY`, `JOINT-MIXER-DECODER`, and `COUNTERFACTUAL-CONTRIBUTION`. The short budget uses 1,024 unique length-32 windows (`32,768` tokens per arm); the long budget uses 4,096 nested-prefix windows (`131,072` tokens per arm). Both use BF16 Lion at `3e-5`, ridge `1e-2`, batch one, one pass, and the same pinned Qwen/data revisions. Checkpoints are `0/512/1024` and `0/1024/2048/4096`, respectively.
- **Evaluation:** every final branch is streamed through the complete frozen Qwen suffix on the same official validation range `[0,8192)`, yielding 7,936 labels per branch. Window-level NLL is retained before aggregation. Short and long runs execute in the same TPU allocation to avoid cross-session evaluator drift.
- **Primary arm and estimand:** `JOINT` is selected before EXP-048 because EXP-045 established it as the best tested layer-0 objective and it is never materially worse than contribution at layer 18. The primary estimand is `NLL(long JOINT) - NLL(short JOINT)` separately at layers 0 and 18. Objective-ranking crossover for mixer-only and contribution is secondary and cannot rescue the primary gate.
- **Per-layer gate:** long JOINT must have negative mean paired NLL change, beat short JOINT in at least two of three seeds, and have a paired 2,000-sample window-bootstrap 95% upper endpoint below zero. The experiment passes only if both layers pass and every numerical/provenance check succeeds.
- **Interpretation boundary:** the treatment changes both unique-token budget and the pre-registered Lion schedule tied to total steps, so it estimates recovery-budget scaling rather than a token-only causal effect. A pass supports longer offline recovery; it does not yet justify full 34-layer conversion.
- **Resource protocol:** run the fresh short and long comparisons sequentially in one estimated 4--8 hour TPU v5e-8 campaign. Each budget result and each layer result is a restart boundary. Regenerable NPY caches and endpoint payloads are deleted only after the corresponding complete JSON exists. The runner requires 5 GiB free before short and 12 GiB before long, uses `/dev/shm` for transient Qwen shards when available, saves raw LM metrics before aggregation, and sends Telegram start/completion/failure notifications.
- **Layer-0 primary result (MEASURED):** long-minus-short JOINT NLL is `-0.29570831`, paired 95% interval `[-0.32887481, -0.25825480]`, with `3/3` seed wins. Mixer-only changes by `-0.04214377`, interval `[-0.09960257, +0.01727137]`, with `2/3` wins; contribution changes by `-0.21488645`, interval `[-0.25864546, -0.16860392]`, with `3/3` wins. The layer-0 scaling gate passes decisively and JOINT remains the best tested objective.
- **Layer-18 primary result (MEASURED):** all three arms become worse with the 4,096-step budget. Long-minus-short NLL is `+0.00268675` for mixer-only, `+0.00308665` for JOINT, and `+0.00292799` for contribution; every arm has `0/3` long-budget wins and a confidence interval entirely above zero. The layer-18 scaling gate fails decisively.
- **Scientific decision:** recovery budget is depth-dependent. More unique recovery data materially helps the early high-shock replacement but harms the already-low-shock layer-18 replacement. Reject any uniform per-layer recovery budget for the eventual hybrid; early stopping or automatic depth-conditioned budgeting is now a central method hypothesis. Overall `passed=true`, `scientific_gate_passed=false`.
- **Execution provenance:** `2,949,120` optimizer-visible tokens across all paired arms; source artifact SHA-256 `305592006b6c21b0b1535d056442ff0d00fb7041219d3e57b3ded9c35f31c025`.

### EXP-049 — Extended-horizon depth-dependent scaling

- **Implementation commit title:** `feat: add extended-horizon TPU campaign and compact summaries`
- **Status:** completed numerically on Kaggle TPU v5e-8 in `4.357` hours. Layer 0 passes; layer 18 has zero long-budget wins and fails, so the overall frozen gate fails while reproducing the depth-dependent direction from EXP-048.
- **Question:** does the opposite scaling sign at layers 0 and 18 persist when the one-pass recovery budgets are moved to 2,048 versus 8,192 unique windows, or was EXP-048 specific to its 1,024/4,096 comparison?
- **Frozen design:** layers `0/18`, seeds `123/456/789`, and all three existing objective arms. The short arm uses 2,048 unique length-32 windows (`65,536` tokens per objective/seed/layer); the long arm uses 8,192 windows (`262,144` tokens). Total optimizer-visible exposure is `5,898,240` tokens. Both budgets retain BF16 Lion at `3e-5`, batch one, ridge `1e-2`, one pass, pinned source/data, and their total-step-dependent frozen schedules.
- **Primary estimand and gate:** JOINT remains the pre-selected primary arm. At each layer compute paired `NLL(8192)-NLL(2048)` on the same 256 validation windows. Long must have negative mean delta, win at least `2/3` seeds, and have a paired 2,000-sample bootstrap 95% upper endpoint below zero. Overall EXP-049 passes only if both layers pass; based on EXP-048, a layer-0 pass/layer-18 fail is an explicitly anticipated scientific outcome, not a post-hoc relabeling of the overall gate.
- **Secondary endpoints:** identical scale deltas, seed wins, and intervals for mixer-only and contribution; final objective rankings; local checkpoint L2 at `0/1024/2048` or `0/2048/4096/8192`; absolute excess NLL versus original Qwen; and numerical stability. Secondary endpoints cannot rescue the primary gate.
- **Resource and failure protocol:** target duration is approximately 6--7 hours. Each `budget x layer` cell is processed sequentially and becomes a durable JSON restart boundary before its NPY cache and endpoint payload are deleted. This bounds working-disk demand despite the 8,192-window cache. Require 6 GiB free before a 2,048-step layer and 11 GiB before an 8,192-step layer; keep transient Qwen shards in `/dev/shm` when available; send Telegram start/completion/failure notifications.
- **Artifact protocol:** preserve the complete machine JSON, but automatically emit `extent-extended-horizon-campaign-summary.json` and `.md` containing only provenance, runtime, token budget, layer/arm deltas, confidence intervals, wins, and gates. These compact artifacts are the default human/LLM handoff; the full JSON is consulted only for audits.
- **Layer-0 result (MEASURED):** 8,192-minus-2,048-step NLL is `-0.02945809` for mixer-only (CI `[-0.06385350, +0.00896162]`, `2/3` wins), `-0.15481465` for JOINT (CI `[-0.18828086, -0.12088480]`, `3/3`), and `-0.15252120` for contribution (CI `[-0.17957339, -0.12783189]`, `3/3`). The primary JOINT gate passes decisively, confirming that the early replacement remains data-hungry beyond 4,096 steps.
- **Layer-18 result (MEASURED):** all long-minus-short means remain positive: `+0.00082100` mixer-only, `+0.00076168` JOINT, and `+0.00042287` contribution, with `0/3` long-budget wins for every arm. Unlike EXP-048, each interval now crosses zero, so the evidence supports no long-budget benefit but does not establish significant harm at this larger budget pair.
- **Replication decision:** the qualitative depth interaction reproduces twice: layers 0 and 18 have opposite seed-win patterns in both EXP-048 and EXP-049. Treat depth-conditioned recovery budgeting as the supported method direction. Overall `passed=true`, `scientific_gate_passed=false`; the false gate is expected because a uniform long budget is precisely what the data reject.
- **Execution provenance:** `5,898,240` optimizer-visible tokens; source compact-summary SHA-256 `341f03b5a8ef0c8040c03374340be42d45c9332fe5ab8852b84bc6f7d3e2285a`. The observed `4.357`-hour runtime was below the planned 5--8-hour utilization target, so the next campaign increases measured work by 1.5x rather than relying on the earlier timing estimate.

### EXP-050 — Multi-depth recovery-budget scaling atlas

- **Implementation commit title:** `feat: add depth-scaling atlas campaign`
- **Status:** completed numerically on Kaggle TPU v5e-8 in `7.940` hours; the frozen scientific gate fails. The null result rules out the pre-registered smooth depth-scaling hypothesis at the tested internal depths and rejects a uniform 8,192-step recovery budget.
- **Question:** does the benefit of a long recovery budget decay systematically with decoder depth, rather than being an idiosyncratic difference between layers 0 and 18?
- **Frozen layers:** `6`, `12`, and `29`. Together with independently measured layers 0 and 18, these positions form an early-to-late depth profile. Layer 6 tests whether strong early-layer scaling persists; layer 12 localizes the transition before layer 18; layer 29 tests a late position already used in objective generalization but never in a budget comparison.
- **Frozen training design:** fresh paired 2,048-step and 8,192-step one-pass runs; seeds `123/456/789`; arms mixer-only, JOINT, and contribution; length 32; BF16 Lion `3e-5`; ridge `1e-2`; identical pinned model/data and validation protocol. Total optimizer-visible exposure is `8,847,360` tokens, exactly 1.5x EXP-049.
- **Primary trend estimand:** for JOINT, compute `NLL(8192)-NLL(2048)` per layer and fit an unweighted linear slope against layer index across `6/12/29`. A positive slope means long-budget benefit becomes weaker or turns harmful with depth. Use 2,000 paired window-bootstrap samples, shared across depths within each draw, to form a 95% slope interval.
- **Scientific gate:** all values finite; the layer-6 JOINT long budget must beat short by negative mean, at least `2/3` seeds, and a paired CI upper endpoint below zero; and the JOINT depth-slope 95% lower endpoint must be above zero. The gate does not require layer 29 to be significantly harmful, avoiding an unjustified effect-size assumption. Mixer-only and contribution slopes are secondary mechanism diagnostics.
- **Resource protocol:** process each of six `budget x layer` cells sequentially, persist its complete JSON, then delete only its verified cache arrays and endpoint payload. Require 6/11 GiB free for short/long cells, use RAM-backed transient Qwen weights, and retain Telegram plus raw-LM-metric failure containment. Based on the measured EXP-049 runtime and 1.5x workload, target runtime is approximately `6.5` hours within the requested 5--8-hour range.
- **Artifact protocol:** automatically emit `extent-depth-scaling-atlas-summary.md` and `.json`. In addition to per-layer arm tables, the summary reports each arm's NLL-delta-per-layer slope, confidence interval, and the primary depth-trend gate.
- **Layer-6 result (MEASURED):** 8,192-minus-2,048-step NLL is `+0.00032148` for mixer-only (CI `[-0.00091376, +0.00162655]`, `1/3` wins), `+0.00013563` for JOINT (CI `[-0.00111447, +0.00144412]`, `1/3`), and `+0.00056037` for contribution (CI `[-0.00067942, +0.00185838]`, `1/3`). The primary layer-6 gate fails: the long budget has no detectable benefit.
- **Layer-12 result (MEASURED):** the corresponding deltas are `-0.00031532` mixer-only (CI `[-0.00138176, +0.00076337]`, `2/3`), `+0.00028036` JOINT (CI `[-0.00071205, +0.00133492]`, `1/3`), and `-0.00036048` contribution (CI `[-0.00146448, +0.00075419]`, `3/3`). Contribution's seed count is a weak secondary signal only; its interval crosses zero and cannot establish a long-budget win.
- **Layer-29 result (MEASURED):** the deltas are `-0.00001689` mixer-only (CI `[-0.00106405, +0.00107696]`, `2/3`), `+0.00010956` JOINT (CI `[-0.00090793, +0.00113451]`, `1/3`), and `+0.00085660` contribution (CI `[-0.00019233, +0.00198592]`, `0/3`). No arm has a statistically resolved long-budget advantage.
- **Depth-trend result (MEASURED):** NLL-delta-per-layer slopes are `-0.00000765` for mixer-only (CI `[-0.00006799, +0.00005741]`), `-0.00000309` for JOINT (CI `[-0.00006796, +0.00006404]`), and `+0.00002574` for contribution (CI `[-0.00004118, +0.00009174]`). Every interval contains zero, so the pre-registered positive JOINT slope and the depth-trend gate both fail.
- **Scientific decision:** EXP-050 does not support a smooth recovery-budget gradient across layers 6/12/29. Combined with EXP-048/049, the supported description is instead a sharp layer-0 exception: layer 0 gains strongly from prolonged recovery, whereas every tested internal layer (6, 12, 18, and 29) is consistent with saturation by 2,048 steps at the current evaluation resolution. This is evidence against uniformly training all transplanted Mamba layers for 8,192 steps, not evidence that the transplant itself fails. The next architecture-scale test should use an asymmetric recovery schedule—long for layer 0 and short or early-stopped for internal layers—and measure progressive multi-layer composition rather than another isolated-layer budget sweep.
- **Execution provenance:** `8,847,360` optimizer-visible tokens; source compact-summary SHA-256 `886d1456d84ed684ae0eb3f72ea4ab3f8ebdd5f854441377896b32f88356db32`.

### EXP-051 — Asymmetric-budget progressive Mamba composition

- **Implementation commit title:** `feat: add progressive composition campaign`
- **Status:** completed numerically and scientifically on Kaggle TPU v5e-8 in `3.211` hours. The primary eight-layer composition gate passes, with statistically sub-additive rather than amplified cross-layer shock.
- **Question:** do independently recovered Mamba-3 transplants compose approximately additively as the model moves from two to four to eight simultaneous replacements, or does cross-layer interaction amplify architectural shock nonlinearly?
- **Frozen nested replacement sets:** `2={0,18}`, `4={0,12,18,29}`, and `8={0,6,12,18,23,29,34,39}`. The first two layers reproduce and extend the earlier two-layer composition path; the eight-layer set spans both decoder boundaries and the full depth. New layers 23, 34, and 39 are included before their standalone results are observed. This is a composition-scaling probe, not yet the final 34-layer allocation.
- **Frozen recovery schedule:** layer 0 receives 8,192 unique length-32 windows, while every other target receives 2,048. Seeds are `123/456/789`; BF16 Lion remains `3e-5`, ridge is `1e-2`, and mixer-only, JOINT, and contribution controls are trained for every standalone cell. Only the pre-selected `JOINT-MIXER-DECODER` endpoint enters the composition branches. Total optimizer-visible training exposure is `6,488,064` tokens.
- **Evaluation:** each standalone endpoint and each nested composition is evaluated on the same 256 official WikiText-2 validation windows. The final progressive pass streams original Qwen and nine composition branches through the same frozen decoder, final norm, and LM head. Original-branch mean NLL must reproduce every standalone evaluator within `0.01`.
- **Primary estimand:** at each replacement count, additive expected excess NLL is the sum of the corresponding standalone `NLL(replacement)-NLL(original)` values. Observed composed excess is measured directly. Composition inflation is `observed composed excess / additive expected excess`; interaction NLL is their difference. The eight-layer stage is primary, while two- and four-layer values are secondary scaling diagnostics.
- **Scientific gate:** all values and all paired-bootstrap ratio draws must be finite; every seed's additive expected excess must be positive; mean eight-layer inflation must be at most `1.25`; at least `2/3` seeds must have inflation at most `1.50`; and the paired 2,000-sample bootstrap 95% upper endpoint must be at most `1.50`. A false gate is a valid scientific outcome and must not terminate the campaign as an infrastructure failure.
- **Resource and failure protocol:** run eight standalone cells sequentially with per-device evaluation batches of four, retain only the small layer-0 validation cache and verified endpoint bundles until composition, and prune every large NPY/MsgPack payload after the final JSON and compact summary exist. Each standalone layer is a restart boundary. The campaign targets one 5--8-hour TPU v5e-8 allocation and sends Telegram start, completion, or caught-failure status.
- **Artifact protocol:** write the complete audit artifact plus `extent-progressive-composition-campaign-summary.md` and `.json`. The compact table contains replacement count, exact layers, additive and observed excess NLL, interaction, inflation with 95% interval, seed passes, baseline parity, and gates.
- **Interrupted attempt (ENGINEERING FAILURE):** the first run reached `1,250.3` seconds, completed layer-39 recovery, wrote endpoint checkpoint SHA-256 `1c7eff48181d588dab40d133f1682fb462489b4ee00e13132e4f06c1a51f3048`, and computed raw LM metrics. Result assembly then raised `KeyError: exp051-progressive-composition` because the new protocol was absent from a duplicated human-readable notes registry. The same omission also bypassed external-validation-cache routing, so those raw metrics used the four internal train-cache evaluation windows rather than the frozen 256 validation windows and are explicitly excluded from scientific analysis. The trained endpoint remains reusable when the same output directory survives.
- **Remediation:** `fix: complete EXP-051 protocol registry` centralizes objective membership, external-evaluation routing, method names, notes, and completion labels, and adds a registry-completeness regression test. The frozen scientific design is not changed.
- **Two-layer result (MEASURED):** additive expected excess NLL is `+0.81251962`, observed composed excess is `+0.80875989`, and interaction is `-0.00375973`. Mean inflation is `0.9952`, paired 95% interval `[0.9894, 1.0013]`, with `3/3` seed passes. The interval includes one, so the two-layer composition is statistically consistent with exact additivity.
- **Four-layer result (MEASURED):** additive expected excess is `+0.80997416`, observed excess is `+0.80401495`, and interaction is `-0.00595920`. Inflation is `0.9923`, interval `[0.9810, 1.0040]`, with `3/3` passes. This stage also remains consistent with additivity and shows no amplification.
- **Eight-layer primary result (MEASURED):** additive expected excess is `+0.92434200`, but observed excess is only `+0.80701774`, yielding interaction `-0.11732426`. Mean inflation is `0.8750`, paired interval `[0.8198, 0.9386]`, with `3/3` seed passes. The full interval lies below one: eight jointly inserted transplants incur about `12.5%` less excess NLL than predicted by summing their isolated effects. The frozen primary gate passes.
- **Composition-scaling decision:** observed excess NLL is nearly flat across 2/4/8 replacements (`0.80876 / 0.80401 / 0.80702`) rather than growing with replacement count. This supports the claim that independently recovered Mamba-3 layers remain composable through eight simultaneous substitutions and do not create a nonlinear shock cascade. The eight-layer sub-additivity is consistent with a layer-0-dominated boundary shock and compensating residual interactions, but the present nested design does not by itself identify the mechanism.
- **Limitation:** all composed branches remain approximately `+0.807` NLL worse than original Qwen. EXP-051 therefore establishes favorable composition scaling, not recovered model quality and not the feasibility of all 34 planned Mamba replacements. The next falsifiable scale test should extend the same nested protocol to 16 replacements before committing compute to the final 15/85 model.
- **Execution provenance:** original-Qwen reproduction passes with maximum absolute NLL discrepancy `0.00173622`; `6,488,064` optimizer-visible tokens; source compact-summary SHA-256 `3bb7446540bbf28054df5372ee487b87a0b48aafcb598c534b3adf68bd807565`.

### EXP-052 — Sixteen-layer boundary scaling and layer-0 ablation

- **Implementation commit title:** `feat: add 16-layer boundary scaling campaign`
- **Status:** completed numerically on Kaggle TPU v5e-8 in `6.292` hours; the frozen 16-layer composition gate and the separate boundary-mechanism gate both fail. The failure identifies a reproducible transition from sub-additive composition at eight replacements to super-additive shock at sixteen.
- **Questions:** (1) does the subcritical composition behavior from EXP-051 persist when simultaneous Mamba replacements double from eight to sixteen; (2) is the approximately constant absolute shock mainly a boundary effect from replacing layer 0 rather than a sum of internal-layer errors; and (3) do the eight newly added internal replacements amplify the existing eight-layer shock?
- **Frozen nested full sets:** `8={0,6,12,18,23,29,34,39}` is copied unchanged from EXP-051. `16={0,1,3,6,9,12,15,18,21,23,26,29,32,34,36,39}` adds the pre-selected layers `{1,3,9,15,21,26,32,36}` across early, middle, and late depth. The 16-layer probe is not the final 34-Mamba allocation.
- **Frozen recovery schedule:** layer 0 receives 8,192 one-pass length-32 windows; each of the other 15 layers receives 2,048. Seeds remain `123/456/789`; BF16 Lion remains `3e-5`; all three mixer-only, JOINT, and contribution arms are trained as standalone controls, while only JOINT endpoints are composed. Total optimizer-visible exposure is `11,206,656` tokens.
- **Frozen evaluation branches:** original Qwen plus, for each seed, `LAYER0-ONLY={0}`, `INTERNAL-7={6,12,18,23,29,34,39}`, `COMPOSED-8`, `INTERNAL-15={1,3,6,9,12,15,18,21,23,26,29,32,34,36,39}`, and `COMPOSED-16`. All 16 branches are streamed through the identical frozen Qwen decoder, final norm, and LM head on the same 256 official WikiText-2 validation windows.
- **Primary composition estimands:** retain additive expected excess, observed full excess, interaction, and inflation for 8 and 16 layers. In addition, estimate the incremental expected shock of the eight added layers and compare it with `NLL(COMPOSED-16)-NLL(COMPOSED-8)` using paired windows.
- **Composition-scaling gate:** the 16-layer full-composition gate retains mean inflation `<=1.25`, at least `2/3` seed ratios `<=1.50`, paired-bootstrap 95% upper endpoint `<=1.50`, positive standalone expected excess, finite values, and original-Qwen reproduction within `0.01` NLL. The incremental 8-to-16 ratio must independently meet the same inflation thresholds. Both must pass for the primary composition-scaling gate.
- **Boundary-mechanism estimands:** for 8 and 16 replacements compute layer-0-only excess, internal-only excess, full excess, and boundary interaction `full - layer0 - internal`. The primary 16-layer mechanism claim requires internal-only excess below layer-0-only excess in at least `2/3` seeds with paired-bootstrap upper endpoint below zero, and negative boundary interaction in at least `2/3` seeds with its interval wholly below zero. This mechanism gate is reported separately and cannot rescue a failed composition gate.
- **Failure semantics:** a false scientific or mechanism gate is a completed result. Only non-finite metrics, provenance mismatch, corrupt endpoint/checkpoint, or execution exception is an engineering failure. Raw LM metrics are written before aggregation so a late summary error does not erase the expensive measurement.
- **Resource protocol:** train 16 standalone cells from deepest to shallowest, making every cell an atomic restart boundary. After a cell's three-arm metrics are durably written, atomically compact its endpoint checkpoint to the JOINT arm needed by composition; this reduces the estimated retained endpoint payload from about 18 GiB to about 6 GiB without discarding any reported control metrics. Retain the compact bundles and layer-0 validation cache until the single 16-branch streamed pass, then delete NPY and MsgPack payloads after the final JSON and summaries exist. Transient Qwen weights prefer `/dev/shm`; disk headroom is checked before each cell. Target measured runtime is approximately 5--7 hours, with Telegram start/completion/failure notifications.
- **Artifact protocol:** the normal handoff is `extent-boundary-scaling-campaign-summary.md`, backed by the full machine artifact `extent-boundary-scaling-campaign.json`. The compact summary must contain full 8/16 scaling, incremental 8-to-16 scaling, boundary decomposition, paired intervals, seed counts, both gates, runtime, token exposure, and baseline parity.
- **Eight-layer replication (MEASURED):** additive expected excess NLL is `+0.92434200`, observed excess is `+0.80701774`, interaction is `-0.11732426`, and inflation is `0.8750` with paired 95% interval `[0.8169, 0.9436]` and `3/3` seed passes. These point estimates exactly reproduce EXP-051, while the independently resampled interval reaches the same conclusion: the eight-layer composition is significantly sub-additive and passes.
- **Sixteen-layer primary result (MEASURED):** additive expected excess is `+1.10758959`, whereas observed excess rises to `+1.50409885`. Interaction is `+0.39650926`; mean inflation is `1.3596`, paired interval `[1.2376, 1.5004]`, with `3/3` seed ratios below the per-seed `1.50` limit. The gate nevertheless fails for two pre-registered reasons: mean inflation exceeds `1.25`, and the bootstrap upper endpoint is marginally above `1.50`. The positive interaction and interval lower bound above one establish super-additivity rather than mere threshold sensitivity.
- **Incremental 8-to-16 result (MEASURED):** the standalone effects of added layers `{1,3,9,15,21,26,32,36}` predict only `+0.18324759` additional NLL, but inserting them together on top of the eight-layer model adds `+0.69708112`. Incremental interaction is `+0.51383353`; inflation is `3.9078` with interval `[2.9874, 5.4267]`. This is the clearest falsification in EXP-052: independently recovered layers substantially underestimate the damage introduced after the replacement density crosses the tested eight-layer regime.
- **Boundary decomposition (MEASURED):** layer-0-only excess is `+0.81876166`. With eight replacements, the seven-internal-only branch has excess `+0.16369723`, full excess is `+0.80701774`, and boundary interaction is significantly compensatory at `-0.17544116` with interval `[-0.2484, -0.1086]`; the mechanism gate passes. With sixteen replacements, the 15-internal-only branch rises to `+0.57117207`, still significantly below layer 0 (internal-minus-layer-0 interval `[-0.3797, -0.1029]`), but full excess becomes `+1.50409885` and boundary interaction changes sign to `+0.11416512` with interval `[-0.0282, +0.2501]`. Layer 0 remains the largest isolated component, but the hypothesis that it almost completely explains dense-composition shock is rejected and the 16-layer mechanism gate fails.
- **Scientific decision:** retain the positive claim only through eight simultaneous replacements. Do not extrapolate independent-layer recovery directly to the planned 34 Mamba layers. EXP-052 supports a density-dependent interaction or activation-distribution-shift hypothesis: the newly added layers are mild in isolation but strongly harmful after upstream Mamba substitutions accumulate. The next experiment must localize the onset within the eight added layers and distinguish replacement count from early-layer adjacency before any full-model recovery run. Appropriate diagnostics are nested `8/10/12/14/16` compositions plus matched-count sparse versus early-dense layouts, using the already frozen asymmetric per-layer budgets and held-out NLL protocol.
- **Execution provenance:** original-Qwen reproduction passes with maximum absolute NLL discrepancy `0.00173622`; `11,206,656` optimizer-visible tokens; source compact-summary SHA-256 `22e5a0c3e217c6c96c3194f67d4261c32ce4c0fd1a2498e0143345b721e11952`.

### EXP-053 — Composition-onset and conditional-layer localization

- **Implementation commit title:** `feat: add composition onset localization campaign`
- **Status:** implementation complete and pre-registered before execution. EXP-052 established a transition somewhere between eight and sixteen replacements; EXP-053 localizes that transition and tests whether it can be attributed to individual added layers rather than only higher-order interactions.
- **Frozen base and additions:** retain the successful EXP-051/052 base `B8={0,6,12,18,23,29,34,39}` and the eight EXP-052 additions `A={1,3,9,15,21,26,32,36}`. No new layer is introduced, so differences from EXP-052 estimate composition context rather than a changed endpoint atlas.
- **Frozen nested path:** `N8=B8`; `N10=B8+{1,36}`; `N12=B8+{1,3,32,36}`; `N14=B8+{1,3,9,26,32,36}`; and `N16=B8+A`. The outside-in paired order adds one early and one late layer at each step and was fixed before observing any EXP-053 branch.
- **Conditional single-addition branches:** for each seed, evaluate `B8+{l}` separately for every `l` in `A`. The primary per-layer estimand is conditional interaction: `[NLL(B8+l)-NLL(B8)] - [NLL(l)-NLL(Qwen)]`. Unlike an inflation ratio, this remains defined when an isolated replacement improves NLL; ratio is therefore diagnostic only when its standalone denominator is positive.
- **Matched-count layout branches:** at 12 replacements compare `EARLY=B8+{1,3,9,15}`, `BALANCED=N12=B8+{1,3,32,36}`, and `LATE=B8+{21,26,32,36}`. Mean excess NLL and all three paired differences are reported. These layout comparisons diagnose early-versus-late placement at fixed count but are secondary and cannot rescue the primary gates.
- **Evaluation surface:** original Qwen plus 15 registered branches per seed: five nested counts, eight single additions, and two extra matched layouts. The resulting 46 branches share the same 256 official WikiText-2 validation windows and are streamed together through the frozen decoder, final norm, and LM head.
- **Onset family and gate:** for each adjacent transition `8→10`, `10→12`, `12→14`, and `14→16`, compare the observed NLL increment with the sum of the newly added layers' standalone increments. A super-additive onset is detected when incremental interaction is positive for at least `2/3` seeds and its paired-bootstrap lower endpoint is above zero. Four tests use Bonferroni familywise `alpha=0.05`, giving `98.75%` adjusted two-sided intervals. The earliest detected upper count is the pre-specified onset location.
- **Single-layer attribution family and gate:** a layer is context-sensitive when its conditional interaction is positive for at least `2/3` seeds and the adjusted paired-bootstrap lower endpoint exceeds zero. Eight tests use Bonferroni familywise `alpha=0.05`, giving `99.375%` adjusted intervals. The attribution gate passes if at least one added layer meets this criterion. This interaction-based rule intentionally does not require positive standalone shock.
- **Overall scientific gate:** numerical validity, at least one corrected nested onset, and at least one corrected context-sensitive single layer are all required. If onset is found but no single layer passes, the result supports pairwise or higher-order interaction instead of first-order attribution. The original 16-layer composition gate is retained separately for replication and is not redefined as the localization gate.
- **Recovery and resources:** reproduce the frozen asymmetric schedule exactly: 8,192 steps for layer 0; 2,048 for every internal target; seeds `123/456/789`; three objective arms trained per standalone cell; JOINT endpoints composed. Total optimizer-visible exposure remains `11,206,656` tokens. After each cell result is durable, compact its endpoint atomically to JOINT only. The final 46-branch pass increases evaluation work without increasing endpoint storage; target runtime is approximately 7--8 hours on TPU v5e-8.
- **Failure and artifact protocol:** every standalone layer is a restart boundary; raw 46-branch LM metrics are written before aggregate analysis; Telegram reports start/completion/caught failure; and large cache/checkpoint payloads are deleted only after final artifacts exist. Normal handoff is `extent-composition-onset-campaign-summary.md`, containing nested stages, adjusted intervals, conditional single-layer attribution, matched layouts, gates, runtime, baseline parity, and token exposure.

### EXP-054 — Apple bridge and MOHAWK orientation initialization screen

- **Implementation commit title:** `feat: add mamba3 bridge initialization ablation campaign`
- **Status:** completed numerically on Kaggle TPU v5e-8 in `0.300` hours. All five arms, both target layers, and all three paired seeds completed; `passed=true`, `complete=true`, and the frozen screening gate fails.
- **Question:** does a learned linear-attention intermediary, a mixer-matrix orientation stage, or their composition provide a better Mamba-3 MIMO recovery initialization than canonical random parameters and the already-rejected direct QKVO port?
- **Literature-derived arms:** `APPLE-BRIDGE` trains independent Hedgehog Q/K feature maps against frozen Qwen attention (feature softmax first, then RoPE), then composes their learned pre-softmax maps with Qwen Q/K and partitions the resulting state features across MIMO rank. `MOHAWK-ORIENTATION` optimizes a head-averaged Mamba-3 exponential-trapezoidal mixer matrix before block-output recovery. `BRIDGE-PLUS-ORIENTATION` applies both stages in order. The source methods are Apple *Attention to Mamba* (arXiv:2604.14191) and MOHAWK (arXiv:2408.10189).
- **Necessary adaptation boundary:** the Apple arm is not an exact HedgeMamba reproduction. Canonical Extent Mamba-3 retains BC RMSNorm, its value-dependent MIMO gate, complex rotation, and exponential-trapezoidal dynamics, while the learned map uses feature softmax and an explicit normalized linear-attention denominator. The implementation therefore labels the transfer as a learned initialization and records `exact_functional_equivalence=false`.
- **MOHAWK adaptation boundary:** Qwen has 40 attention heads while the parameter-matched Mamba-3 mixer has 60 recurrent heads. Stage-1 orientation therefore compares causal row-normalized matrices after averaging heads, and excludes the value gate, D skip, and output projection. This is a MOHAWK-inspired Mamba-3 objective, not the original equal-head Mamba-2 Frobenius objective.
- **Frozen targets:** layers 18 and 0, in that order. Layer 18 is prioritized as the representative interior layer if the wall deadline interrupts the campaign. Seeds are `123/456/789`; sequence length is 32; every arm receives the identical 1,024-step JOINT mixer+decoder recovery schedule and paired data order.
- **Frozen controls and order:** `CONTROL-RANDOM`, `APPLE-BRIDGE`, `BRIDGE-PLUS-ORIENTATION`, `MOHAWK-ORIENTATION`, and `CONTROL-QKVO`. The first three run first so a deadline-limited artifact still contains the primary paired comparison. Direct QKVO remains a negative control rather than a proposed initializer.
- **Initializer construction:** 128 feature-map updates are performed once per layer/seed for both Apple-derived arms. Each oriented arm receives 128 orientation updates. These tokens and optimizer steps are reported separately from the common recovery budget; they are not silently counted as free initialization.
- **Evaluation:** checkpoints `0/256/512/1024` are evaluated on 64 official WikiText-2 validation windows from an external evaluation-only cache. Metrics include mixer, frozen decoder output, and counterfactual contribution relative L2, finite-gradient diagnostics, bridge output/matrix error, and orientation loss.
- **Exploratory screen gate:** at both completed target layers, `BRIDGE-PLUS-ORIENTATION` must beat `CONTROL-RANDOM` in at least `2/3` paired seeds and in mean final decoder relative L2. This gate ranks methods for a later locked confirmation; it is not full-model or publication-level evidence.
- **Four-hour resource protocol:** the campaign has a default 3.5-hour internal wall deadline, checks it during every initializer and recovery loop, writes a partial JSON after each completed arm, and exits normally with `status=deadline_partial` rather than losing completed measurements. Qwen shards use `/dev/shm` when available. Large activation arrays are deleted only after the corresponding layer JSON is durable. Telegram reports start, completion/deadline-partial, or caught failure.
- **Artifact protocol:** normal handoff is `extent-bridge-ablation-campaign-summary.md`; the full audit artifact is `extent-bridge-ablation-campaign.json`. No model checkpoint is required for this screen, and NPY activation caches must not be downloaded.
- **Random baselines (MEASURED):** mean final decoder relative L2 is `0.54782944` at layer 0 and `0.00550527` at layer 18 after the common 1,024-step decoder-aware recovery budget.
- **Apple bridge result (MEASURED):** `APPLE-BRIDGE` reaches `0.56515338` at layer 0 (`+0.01732394`, or `+3.16%`, relative to random) and `0.00633959` at layer 18 (`+0.00083432`, or `+15.15%`). It loses to random in `0/3` paired seeds at both depths. Folding the learned Hedgehog bridge into canonical Mamba-3 therefore creates negative transfer under this protocol.
- **Composed bridge plus orientation result (MEASURED):** `BRIDGE-PLUS-ORIENTATION` reaches `0.56612708` at layer 0 (`+0.01829764`, `+3.34%`) and `0.00631139` at layer 18 (`+0.00080611`, `+14.64%`), again with `0/3` wins at each layer. MOHAWK-style orientation does not rescue the bridge initializer.
- **MOHAWK-only result (MEASURED):** `MOHAWK-ORIENTATION` reaches `0.54821586` at layer 0 (`+0.00038643`, `+0.071%`) and `0.00551168` at layer 18 (`+0.00000641`, `+0.116%`). It loses all paired seeds, but its mean difference from random is tiny. The supported conclusion is that this 128-step head-averaged orientation proxy is neutral-to-slightly-harmful, not that matrix orientation is universally ineffective.
- **Direct-QKVO negative-control replication (MEASURED):** `CONTROL-QKVO` reaches `0.59610430` at layer 0 (`+0.04827486`, `+8.81%`) and `0.00553378` at layer 18 (`+0.00002851`, `+0.518%`), with `0/3` wins at both depths. This independently preserves the earlier conclusion that full-strength direct QKVO transplantation should not be selected.
- **Full bridge diagnostics (MEASURED):** the compact ranking hid severe Stage-1 instability. At layer 18, post-training bridge output relative L2 is `1.049/51.699/43.158` across seeds, with matrix relative MSE `0.265/496.479/162.386` and maximum raw gradient norms from `2.37e4` to `2.28e10`. At layer 0, output L2 is `1.905/457.086/2.175`, matrix MSE is `12.682/740189.235/21.072`, and maximum raw gradient norms reach `3.94e12`. All values remain finite and clipping protects downstream recovery, but two-layer/three-seed behavior is not numerically stable.
- **Orientation diagnostics (MEASURED):** mean MOHAWK-only orientation loss changes `0.706824 -> 0.708083` at layer 18 and `0.304230 -> 0.333543` at layer 0. Bridge-plus-orientation changes `0.707022 -> 0.700140` at layer 18 but worsens `0.300697 -> 0.329428` at layer 0. Therefore the current 128-step proxy does not consistently optimize even its own Stage-1 objective; its near-random recovery endpoint is not evidence of useful transferred orientation.
- **Mechanistic diagnosis:** the EXP-054 bridge applies RoPE to all 64 Hedgehog feature coordinates. Because RoPE destroys feature positivity, the explicit linear-attention denominator can approach zero, explaining extreme but finite outputs and gradients. Apple HedgeMamba also applies the feature map before RoPE, but inherits partial RoPE from its Pythia teacher and therefore retains a positive non-rotary suffix. Canonical Extent Mamba-3 rotates only `rope_fraction=0.5` of B/C state dimensions. EXP-054 consequently tested a mismatched full-RoPE adaptation rather than a numerically faithful bridge into the production Mamba-3 state geometry.
- **Scientific decision:** retain canonical random Mamba-3 initialization as the leading tested initializer and permanently preserve EXP-054 as a failed full-RoPE bridge result. The failure motivates one separately numbered correction, not reinterpretation: EXP-055 tests a partial-RoPE, cosine-only stabilized bridge. Only that locked correction may determine whether the Apple-derived direction advances.
- **Execution provenance:** supplied compact-summary SHA-256 `16b4401c4aba124cd7adf463cf8d12908152278db9f29950c81ae25701459f73`; supplied full-campaign SHA-256 `ab54ec75be561d0df645bbe08bdf216ab48b7b290f0c4f5c17f43b1b1d5d20cc`; raw compact artifact `results/EXP-054-bridge-ablation-campaign-summary.md`.

### EXP-055 — Stabilized partial-RoPE bridge confirmation

- **Implementation commit title:** `feat: add stabilized partial-rope bridge campaign`
- **Status:** completed numerically on Kaggle TPU v5e-8 in `0.706` hours. Both layers, all five arms, three seeds, and 4,096 recovery steps completed with finite values. The stabilized Apple bridge primary gate fails.
- **Hypothesis:** retaining a positive non-rotary Hedgehog suffix prevents near-zero normalization denominators, allowing the learned bridge to provide a useful Mamba-3 initialization under longer matched recovery.
- **Locked correction:** set bridge `rope_fraction=0.5` to match canonical Extent Mamba-3 B/C rotation, use cosine-only Stage-1 output matching (`matrix_loss_weight=0`) as described by Attention to Mamba, reduce bridge Lion LR from `1e-3` to `3e-4` in response to EXP-054 raw gradient norms up to `3.94e12`, and train bridge and orientation stages for 256 updates each. These three remediation choices are jointly tested; EXP-055 is not a factorial attribution of which individual change fixes stability.
- **Controlled recovery:** layers `18` then `0`, seeds `123/456/789`, sequence length 32, and the same five registered arms. `APPLE-BRIDGE` now denotes the locked stabilized recipe; random, bridge-plus-orientation, MOHAWK-only, and direct-QKVO remain controls. Every arm receives the identical 4,096-step decoder-aware recovery with checkpoints `0/256/1024/2048/4096` and paired data order.
- **Primary gate:** stabilized `APPLE-BRIDGE` must beat `CONTROL-RANDOM` in at least `2/3` paired seeds and in mean final decoder relative L2 at each completed target layer. Bridge-plus-orientation is secondary and cannot rescue a failed primary bridge gate. All bridge outputs, gradients, and recovery metrics must remain finite.
- **Interpretation boundary:** a pass supports the corrected initialization under short-context layerwise recovery, not the exact Apple HedgeMamba architecture, end-to-end 14B recovery, or long-context quality. A failure closes this learned bridge family for the current Extent Mamba-3 design and returns development to random initialization plus better recovery objectives.
- **Resource and durability protocol:** target at most 3.4 wall-clock hours on one v5e-8 allocation. The runner writes after every arm, returns `deadline_partial` at its internal deadline, uses RAM-backed Qwen shards when available, deletes only regenerable NPY arrays after each durable layer result, and sends Telegram start/completion/deadline/failure notifications. Normal handoff is `extent-stabilized-bridge-campaign-summary.md`; the full audit artifact is `extent-stabilized-bridge-campaign.json`.
- **Numerical-stability result (MEASURED):** partial RoPE and the reduced bridge LR eliminate the catastrophic EXP-054 scale explosion but do not make Stage 1 uniformly well behaved. Maximum raw bridge gradient norm falls from `3.94e12` to `1.49e4`. Layer-18 post-fit bridge output L2 remains poor at `3.872/7.324/3.535`; layer-0 outputs are `0.458/41.957/10.424`. The correction therefore improves numerical scale by many orders of magnitude without producing a reliable attention surrogate.
- **Primary Apple result (MEASURED):** at 4,096 steps, random / stabilized Apple mean decoder relative L2 is `0.52084569 / 0.52526938` at layer 0 and `0.00510891 / 0.00616920` at layer 18. Apple loses `0/3` seeds at both depths and is worse by `0.85%` and `20.75%`, respectively. At layer 0 its mean gap contracts from `+0.139535` at step 0 to `+0.004424` at step 4,096, but compute-efficient recovery still fails; at layer 18 the gap grows after step 256. The primary gate remains false.
- **Bridge-plus-orientation and MOHAWK results (MEASURED):** bridge-plus-orientation is worse than random by `1.05%` at layer 0 and `18.67%` at layer 18, with `0/3` wins. MOHAWK-only remains nearly neutral but consistently loses: `+0.214%` at layer 0 and `+0.095%` at layer 18, also `0/3`. Neither method advances.
- **Delayed QKVO crossover (MEASURED):** the direct-QKVO negative control remains substantially worse at layer 0 (`0.56601205` versus random `0.52084569`, `+8.67%`, `0/3` wins). At layer 18, however, it crosses random only after the longer budget: mean QKVO-minus-random is `+9.47e-5` at step 1,024, `+2.27e-6` at step 2,048, and `-5.53e-5` at step 4,096; the final QKVO mean is `0.00505362` versus `0.00510891`, a `1.08%` improvement with `3/3` wins. This is the first paired evidence that transplanted Qwen directions can become beneficial at long horizon in an internal layer while remaining harmful at the input boundary.
- **Scientific decision:** close the learned Hedgehog/bridge family for the present Extent Mamba-3 architecture. Preserve random initialization for layer 0. Reopen direct QKVO only as an internal-layer, long-horizon signal—not as a universal initializer. The next high-value experiment should test whether an algebraically exact SISO-to-MIMO lift of the same QKVO donor moves the layer-18 crossover earlier, using single-active-channel and balanced-rank embeddings against random and the current flat QKVO port. Multiple internal depths and end-to-end NLL are required before changing the full-model initialization policy.
- **Execution provenance:** supplied compact-summary SHA-256 `feece097075781898024a7059a40160a8184b983c2c9dc6da2fcce8641256c81`; supplied full-campaign SHA-256 `9d269e08b4f8122c5f9d3398ee7a6a73b4613de4caf14c09648deba96dc98dc3`; compact artifact `results/EXP-055-stabilized-bridge-campaign-summary.md`.

### EXP-056 — Operator-preserving SISO-to-MIMO lift (pre-registered)

- **Implementation commit title:** `feat: add operator-preserving MIMO lift campaign`
- **Motivation:** EXP-055 found a delayed direct-QKVO advantage at internal layer 18 only after 4,096 recovery steps. EXP-056 asks whether the donor was useful but embedded poorly into Mamba-3's MIMO recurrence.
- **Intervention:** `SINGLE-CHANNEL-LIFT` activates only rank channel zero. `BALANCED-RANK-LIFT` copies B/C across all rank channels, sets MIMO input/output scales to `1/R`, retains the gate, and multiplies the D skip by `R`. Because B/C are RMS-normalized, this compensation preserves the complete pre-training operator.
- **Frozen controls:** canonical random and the EXP-055 flat direct-QKVO port. All four arms receive identical ridge readout calibration, Lion optimizer, paired data order, and decoder-aware recovery.
- **Depth and compute:** layers `18, 6, 29, 0`; seeds `123/456/789`; sequence length 32; 4,096 recovery steps per arm; checkpoints `0/256/1024/2048/4096`; external WikiText-2 validation windows. Layer 0 is a separately interpreted boundary diagnostic.
- **Primary measurements:** paired decoder-output relative-L2 difference from random, wins across seeds, and earliest checkpoint where the mean difference crosses below zero. This is a layer-local decoder-aware screen, not an end-to-end NLL or full 14B claim.
- **Exploratory gate:** at each internal layer 6/18/29, balanced lift must beat random at step 4,096 in mean and at least `2/3` seeds, and cross no later than flat QKVO. Layer 0 cannot veto the internal-layer gate.
- **Mechanistic interpretation:** single and balanced lifts compute the same function at step zero. Any later separation therefore isolates optimization geometry from initial function quality.
- **Durability:** the v5e-8 campaign has a 7.5-hour internal deadline, saves after every arm, mirrors partial/final JSONs, prunes only regenerable arrays after durable results, and sends Telegram start/completion/deadline/failure notifications.
- **Artifacts:** normal handoff is `extent-mimo-lift-campaign-summary.md` plus `extent-mimo-lift-campaign.json`; failure handoff is `exp056-campaign-failure.json` plus `exp056-campaign-stage-manifest.json`.
- **Status (MEASURED):** completed on Kaggle TPU v5e-8 in `1.713` hours with all four layers, four arms, three seeds, and 4,096-step trajectories present. `complete=true`, `passed=true`, and the pre-registered internal-layer screening gate passes.
- **Exact-lift result (MEASURED):** balanced lift beats random at all four depths: layer 0 `0.50388815` versus `0.52084569` (`-3.26%`, `3/3`); layer 6 `0.00914741` versus `0.01058044` (`-13.54%`, `3/3`); layer 18 `0.00481987` versus `0.00510891` (`-5.66%`, `3/3`); and layer 29 `0.01269282` versus `0.01317885` (`-3.69%`, `3/3`). Its mean crossover occurs at the first post-training checkpoint, step 256, at every depth.
- **Single-channel replication (MEASURED):** single-channel lift also crosses by step 256 and finishes better than random at every layer: `-3.26%`, `-12.78%`, `-5.69%`, and `-3.60%` at layers 0/6/18/29. It wins `3/3`, `2/3`, `3/3`, and `3/3` paired seeds, respectively.
- **Parameterization attribution (MEASURED):** single and balanced outputs are exactly equal at step zero as designed. Their final mean difference is only `0.0063%` at layer 0, `0.889%` at layer 6, `0.0289%` at layer 18, and `0.0965%` at layer 29. Neither parameterization dominates consistently. The main supported mechanism is exact operator preservation; balanced MIMO optimization geometry is at most a secondary effect under this budget.
- **Flat-QKVO control (MEASURED):** flat QKVO remains harmful at layers 0 (`+8.67%`), 6 (`+8.16%`), and 29 (`+1.98%`). It repeats the delayed layer-18 crossover only at step 4,096 and finishes `1.08%` better than random. Thus the earlier isolated success was donor signal obscured by a poor embedding, not evidence that the flat port generalizes.
- **Shock/recovery observation:** exact lifts initially have worse calibrated decoder error than random at every depth, yet overtake random by step 256. The method does not eliminate immediate architectural shock; it places the model in a substantially more recoverable basin. This distinction must be explicit in the paper.
- **Numerical stability (MEASURED):** all runs are finite. Maximum recorded gradient norms for balanced lift are `12.07`, `26.50`, `36.13`, and `100.78` at layers 0/6/18/29. Gradient clipping remains active; the result supports stable recovery, not unbounded-gradient equivalence.
- **Scientific decision:** exact operator-preserving lift replaces canonical random initialization as the leading Mamba-3 transplant candidate. Do not claim full-model superiority yet. Next require a locked external-text confirmation and progressive multi-layer/end-to-end NLL comparison using exact lift versus random, while treating single and balanced as near-equivalent unless later scale tests separate them.
- **Execution provenance:** compact-summary SHA-256 `cee9326520d3fa4dfc7a2efc83ce4fd3f6bce730919234a35cc40e76e15079ab`; full-campaign SHA-256 `dbc6e191e0cc6c9f1db613be63905b4aa1ce39af1eed240d4621443e06aa7652`; compact artifact `results/EXP-056-mimo-lift-campaign-summary.md`.

### EXP-057 — Two-hour locked fresh-text exact-lift confirmation (pre-registered)

- **Implementation commit title:** `feat: add two-hour exact-lift confirmation campaign`.
- **Purpose:** independently confirm EXP-056 under a frozen distribution shift before spending compute on endpoint composition. This campaign does not reuse EXP-056 activations or select hyperparameters from its validation outcomes.
- **Locked changes:** sequence length increases from 32 to 64; training uses WikiText-2 train token offset `131072`; evaluation uses the disjoint validation split at offset `8192`. Layers remain `18, 6, 29, 0`, seeds remain `123/456/789`, and recovery remains 4,096 updates.
- **Arms:** only `CONTROL-RANDOM` and `BALANCED-RANK-LIFT`. Removing already-resolved single/flat controls directs the two-hour budget to a paired confirmation with twice the context length.
- **Primary gate:** balanced exact lift must beat random in mean final decoder relative L2 and at least `2/3` paired seeds at every layer. Unlike EXP-056, layer 0 is part of the locked confirmation gate.
- **Boundary:** a pass confirms layer-local recovery generalization. It still does not establish multi-layer composition or end-to-end language-model NLL; those remain the next experiment.
- **Resource protocol:** one fresh v5e-8 process, 1.85-hour internal deadline, layer order 18→6→29→0, partial JSON after every arm, regenerable-cache pruning, and Telegram start/final/failure notifications.
- **Status (MEASURED):** `deadline_partial`, `complete=false`, `passed=true`. Layers 18 and 6 completed all paired arms/seeds; layers 29 and 0 were not measured. The declared all-layer confirmation gate is therefore false by construction and must not be reported as a scientific failure.
- **Protocol deviation (MEASURED):** wall time was `3.145` hours despite the declared 1.85-hour internal deadline. Deadline checks cannot interrupt an already-dispatched JAX compilation/device operation, so the implementation did not provide the promised hard bound. Preserve this deviation in the audit trail; future short campaigns must budget at arm boundaries and reserve compilation margin rather than describe a monotonic check as a hard deadline.
- **Layer-18 confirmation (MEASURED):** balanced lift reaches mean decoder relative L2 `0.00731246` versus random `0.00772127`, an improvement of `5.295%` with `3/3` paired wins. It starts worse at step zero but is already better in every seed by step 256 and remains better through step 4,096.
- **Layer-6 confirmation (MEASURED):** balanced lift reaches `0.01016222` versus random `0.01370219`, an improvement of `25.835%` with `3/3` paired wins. It again begins substantially worse, crosses by step 256 in all seeds, and expands its advantage with recovery.
- **Stability (MEASURED):** all twelve completed trajectories are finite. Maximum balanced-lift gradient norms are `16.45` at layer 18 and `17.91` at layer 6. No numerical instability explains the gains.
- **Interpretation:** EXP-057 independently confirms the EXP-056 mechanism on new token offsets and twice the context length at two internal depths. Combined evidence is now `6/6` balanced-lift wins per confirmed depth across the exploratory and locked runs. Generalization to layers 29/0 at length 64 remains unconfirmed, and multi-layer composition/end-to-end NLL remains entirely untested for this initializer.
- **Decision:** retain balanced exact lift as the leading internal-layer initializer. Do not rerun the missing two cells merely to turn the Boolean gate green unless compute becomes available; the higher-value next experiment is a small composed end-to-end comparison, with a genuinely conservative wall-budget design.
- **Execution provenance:** compact-summary SHA-256 `16ca45ad8bd44fe4700708776f29ebfbe2d24c59071a990362cb96ca55889d58`; full-campaign SHA-256 `cada5cb274596a54d54d9a9f737253d269f645e0db296bc8ea4fa444af270623`; compact artifact `results/EXP-057-exact-lift-confirmation-summary.md`.

### EXP-058 — One-hour exact-lift composition pilot (pre-registered)

- **Implementation commit title:** `feat: add one-hour exact-lift composition pilot`.
- **Question:** after equal short recovery, does exact operator lift reduce the end-to-end NLL damage caused by composing two Mamba replacements, relative to random Mamba initialization?
- **Frozen design:** Qwen layers 0 and 18; paired model seed 123; random versus balanced exact lift; 1,024 decoder-aware updates per layer/arm; sequence length 32; new train offset `196608`; new validation offset `16384`; 256 validation windows.
- **Composition evaluation:** store the four trained endpoints, stream frozen Qwen3-14B once for the random pair and once for the exact-lift pair, and measure final LM-head NLL on identical token windows. The original Qwen branch is retained as a calibration reference.
- **Pilot gate:** exact-lift two-layer composition mean NLL must be finite and lower than random two-layer composition mean NLL. Excess NLL versus original Qwen is reported but does not gate this initializer ranking.
- **Interpretation boundary:** one seed and 1,024 updates cannot establish publication-level composition robustness. A pass only licenses a later multiseed composition confirmation; a failure rejects immediate scaling of exact lift to the 85% replacement plan under the present recovery recipe.
- **Resource strategy:** prioritize completing both endpoint pairs and one shared streaming pass rather than attempting more seeds. Durable endpoint checkpoints, stage manifest, Telegram start/final/failure, and compact output are retained. No hard wall-time claim is made because JAX dispatch cannot be interrupted safely.
- **Status (MEASURED):** completed numerically on TPU v5e-8 in `0.195` hours. Both endpoint pairs and the shared 256-window full-model stream are finite; `passed=true` and the one-seed pilot gate passes.
- **End-to-end NLL (MEASURED):** original Qwen NLL is `4.28904451`; the random two-layer composition reaches `5.59118033`; balanced exact lift reaches `5.27788715`. Exact lift improves NLL by `0.31329318` (`5.60%`) relative to random and removes `24.06%` of random's excess NLL above Qwen.
- **Residual composition shock (MEASURED):** exact lift still has `+0.98884264` NLL above original Qwen. The result supports a better initialization, not successful recovery of the two-layer hybrid. Scaling this endpoint directly to 85% replacement would be unjustified.
- **Layer-local endpoints (MEASURED):** after 1,024 updates, layer-0 decoder relative L2 is `0.52601750` for exact lift versus `0.54560047` for random; layer 18 is `0.00528307` versus `0.00552079`. Both exact endpoints begin worse at step zero and recover to better endpoints, matching EXP-056/057's basin interpretation.
- **Propagation diagnostic (MEASURED):** exact composition has lower hidden relative L2 than random at every recorded depth. Relative divergence reductions are `3.59%` after layer 0, `6.59%` at layer 9, `6.60%` at layer 18, `6.74%` at layer 19, `6.79%` at layer 29, and `9.04%` at layer 39. The transferred advantage survives and grows through frozen downstream layers rather than disappearing locally.
- **Stability (MEASURED):** all endpoint and streaming values are finite. Maximum gradient norms are `8.24`/`10.69` for exact lift at layers 0/18 and `9.99`/`8.61` for random.
- **Scientific interpretation:** EXP-058 is the first evidence that exact operator lift improves composed, end-to-end language-model behavior—not only layerwise representation matching. Because the comparison uses one paired initialization seed and only two replaced layers, it is pilot evidence. The next confirmatory design must use at least three paired seeds and multiple replacement counts, report uncertainty over validation windows, and retain random plus original-Qwen controls.
- **Decision:** advance exact lift to a multiseed progressive-composition confirmation when TPU budget returns. Do not spend remaining sub-hour fragments repeating this cell; the immediate engineering priority can shift to checkpointable full-model recovery infrastructure and MLA integration while preserving EXP-058 as the initializer-selection basis.
- **Execution provenance:** compact-summary SHA-256 `15ea05d48703f150e1b02f4e119c1613808b21a0754aeb54d45e012672d77272`; full-campaign SHA-256 `e7d9b4b13a52fcf01031b6d31906862a9b3cfbc36ff9208b84920d2d19a6f86c`; compact artifact `results/EXP-058-exact-lift-composition-pilot-summary.md`.

### EXP-059 — Multiseed exact-lift composition confirmation (pre-registered)

- **Implementation commit title:** `feat: add multiseed exact-lift composition confirmation`.
- **Locked extension of EXP-058:** repeat the same layers 0+18, 1,024-step recovery, train/validation offsets, sequence length, and full-model streaming evaluation for paired seeds `123/456/789`. No hyperparameter is changed in response to the pilot result.
- **Branches:** original Qwen plus six composed branches: random and balanced exact lift for each paired model seed. Endpoint identifiers with `seed+1000` denote exact lift and are bookkeeping labels, not independent seeds.
- **Primary statistic:** per-seed exact-minus-random final NLL on identical 256 validation windows. A hierarchical paired bootstrap resamples model seeds and validation windows with 2,000 draws.
- **Confirmation gate:** mean exact-minus-random NLL below zero, exact lift wins at least `2/3` model seeds, all values finite, and the paired-bootstrap 95% upper confidence bound is below zero.
- **Scope:** a pass confirms a two-layer composition advantage under short recovery. It does not establish scaling to 4/8/34 replaced layers or recovery to original-Qwen NLL.
- **Status (MEASURED):** completed on TPU v5e-8 in `0.332` hours. All 12 endpoint trajectories, seven full-model branches, and 256 evaluation windows completed with finite values. The pre-registered confirmation gate passes.
- **Primary result (MEASURED):** exact lift beats random composition in all `3/3` paired seeds. Exact-minus-random NLL is `-0.31329318`, `-0.14561638`, and `-0.16830390` for seeds 123/456/789. The mean paired difference is `-0.20907115`; the hierarchical paired-bootstrap 95% interval is `[-0.31977238, -0.11063733]`, wholly below zero.
- **Effect size (MEASURED):** mean random-composition NLL is `5.53835315`; mean exact-lift NLL is `5.32928199`; original Qwen is `4.28904451`. Exact lift reduces total NLL by `3.77%` relative to random and removes `16.73%` of random composition's excess NLL above Qwen. Per-seed random-relative reductions are `5.60%`, `2.69%`, and `3.00%`.
- **Endpoint agreement (MEASURED):** exact lift finishes below random decoder relative L2 for every layer/seed endpoint. Layer-0 exact/random endpoints are `0.526018/0.545600`, `0.525941/0.541853`, and `0.529189/0.545401`; layer-18 endpoints are `0.005283/0.005521`, `0.005272/0.005496`, and `0.005306/0.005521`.
- **Propagation (MEASURED):** averaged hidden relative L2 is lower for exact lift at every diagnostic depth: reductions of `3.17%`, `2.57%`, `2.57%`, `1.64%`, `1.75%`, and `5.19%` at layers 0/9/18/19/29/39. One exact seed is locally worse around layers 9–29, but its final NLL still beats its paired random branch; mean propagation and the primary NLL comparison remain consistent.
- **Stability (MEASURED):** all endpoint gradients and streamed outputs are finite. The largest recorded exact-lift gradient norm is `23.09` (layer 18, seed 789); no failure is hidden by aggregation.
- **Scientific conclusion:** the exact operator-preserving transplant now has replicated layer-local evidence (EXP-056), independent fresh-text evidence at two depths (EXP-057), and a pre-registered multiseed end-to-end two-layer composition advantage (EXP-059). This supports the claim that preserving the SISO operator during MIMO embedding yields a more recoverable cross-architecture initialization than random or flat QKVO transplantation.
- **Remaining boundary:** exact composition remains roughly `+1.04024` NLL above original Qwen after 1,024 updates. The result establishes relative initializer quality, not recovered model quality, long-context inference, 15/85 replacement scaling, or an MLA interaction. The next publication-critical experiment is progressive 2/4/8-layer scaling with matched exact/random controls and then a checkpointable full-model recovery run.
- **Execution provenance:** compact-summary SHA-256 `7d6f4a83a29331810d5ae99ba62b33e8b55e5420f9e1c1ab9c83b0460c613dea`; full-campaign SHA-256 `3aa7298a8e5265ce8eed95502f11f151752edabec901882d4896925118adb1b9`; compact artifact `results/EXP-059-exact-lift-composition-confirmation-summary.md`.

### EXP-060 — Exact-lift progressive 2/4/8-layer scaling (pre-registered)

- **Implementation commit title:** `feat: add exact-lift progressive scaling campaign`.
- **Question:** does the confirmed two-layer exact-lift advantage persist as independently trained replacements are composed into nested 4- and 8-layer hybrids, or does composition error grow faster than the initialization benefit?
- **Nested sets:** `2={0,18}`, `4={0,12,18,29}`, and `8={0,6,12,18,23,29,34,39}`. These cover input boundary, early/interior/late depths, and preserve exact nesting for paired scaling comparisons.
- **Paired design:** random versus balanced exact lift for model seeds `123/456/789`. Layer 0 receives 8,192 decoder-aware updates; every other endpoint receives 4,096. Sequence length 32, train offset `262144`, validation offset `24576`, and 256 validation windows are frozen before execution.
- **Evaluation:** one streamed Qwen3-14B pass evaluates original Qwen plus random/exact branches at each 2/4/8 replacement count. Primary outcomes are paired end-to-end NLL differences; hidden-state divergence is recorded at diagnostic depths.
- **Statistics and gate:** for each count independently, exact lift must win at least `2/3` seeds, have negative mean exact-minus-random NLL, and a hierarchical paired-bootstrap 95% upper bound below zero. The global gate requires all three stages to pass and all outputs to remain finite.
- **Durability:** each fully trained layer writes a BF16 endpoint checkpoint and becomes a restart boundary. Interior layers run before the longer layer-0 cell. Activation arrays are deleted only after their endpoint is durable; layer-0 validation is retained for the final stream. Telegram reports start/completion/failure and the failure artifact records the precise stage.
- **Resource expectation:** designed for a fresh 5–8-hour v5e-8 allocation. Runtime is an estimate from EXP-058/059 and is not described as a hard deadline. Endpoint payloads may occupy several GiB in Kaggle output and must not be downloaded after normal completion.
- **Interpretation boundary:** a pass supports exact-lift scaling through eight replacements, not the final 34 Mamba layers, MLA conversion, long-context performance, or full recovery to Qwen NLL. A failed 8-layer stage with passing 2/4 stages would identify a composition boundary rather than invalidate the transplant mechanism.
- **Status (MEASURED):** completed on Kaggle TPU v5e-8 in `4.658` hours. All eight layer endpoint bundles, 48 endpoint trajectories, 19 streamed full-model branches, and 256 evaluation windows completed with finite outputs. The global scientific gate passes.
- **Primary scaling result (MEASURED):** exact lift wins `3/3` seeds at every replacement count. Mean exact-minus-random NLL is `-0.16736624` for 2 layers, `-0.15952437` for 4, and `-0.15670423` for 8. Hierarchical paired-bootstrap 95% intervals are `[-0.20993397,-0.12216314]`, `[-0.20455281,-0.11114863]`, and `[-0.22736608,-0.08829038]`; every upper bound remains below zero.
- **Effect-size stability (MEASURED):** mean random/exact NLL is `5.194667/5.027301`, `5.187954/5.028430`, and `5.141958/4.985253` at 2/4/8 replacements. Exact lift reduces NLL by `3.22%`, `3.08%`, and `3.05%` relative to random and removes `20.93%`, `20.11%`, and `20.97%` of random's excess above the original-Qwen NLL `4.39481045`. The benefit is approximately constant rather than collapsing with replacement count.
- **Composition behavior (MEASURED):** under independently recovered endpoints, neither random nor exact NLL worsens monotonically from 2 to 8 replacements; the 8-layer branches are slightly better than their corresponding 2/4-layer branches. Therefore EXP-060 finds no early superlinear composition catastrophe through eight selected layers. It does not imply that arbitrary additional replacements improve quality.
- **Representation-surrogate finding (MEASURED):** averaged exact-lift hidden relative L2 is `3.95%` lower immediately after layer 0, but is roughly `3–7%` higher than random through several middle diagnostics. At final layer 39 it becomes `6.85%`, `6.63%`, and `15.28%` lower for 2/4/8 replacements. Exact lift nevertheless wins NLL in every paired seed. Intermediate teacher hidden-L2 is consequently not a reliable monotonic surrogate for final language-model quality; the exact-lift trajectory can deviate more mid-network and recover better downstream.
- **Gradient caveat (MEASURED):** all runs remain finite and clipping is active, but raw maximum gradient norms are extreme at layers 34 and 39. Layer 34 reaches `7.15e5` (random) and up to `5.42e5` (exact); layer 39 reaches `2.41e4` (random) and `1.56e5` (exact). Final-step gradients are much smaller and endpoints outperform random locally, but this is evidence of rare optimization spikes. Full-model recovery must retain clipping, log pre/post-clip norms, and investigate robust loss scaling rather than claim uniform numerical stability.
- **Scientific conclusion:** EXP-060 confirms that the operator-preserving transplant advantage scales from two through eight strategically distributed Mamba-3 replacements without measurable erosion of end-to-end NLL benefit. Together with EXP-056/057/059, this establishes exact lift as the selected initialization method for the Mamba portion of Extent, subject to full-model validation.
- **Next decision:** stop further small layerwise initializer screens. The publication-critical next phase is checkpointable 15/85 hybrid recovery with staged replacement or curriculum, exact lift versus a matched random-control run at smaller scale, and simultaneous MLA interaction ablations. Long context, throughput, HBM, and generation quality must be measured separately.
- **Execution provenance:** compact-summary SHA-256 `4c476a0a6217f9e248ac4d8d7f00bbee80ca7678f0a7d8b25ff8aa43352db6a5`; full-campaign SHA-256 `422e0d0ec03d12aac7ee7dc027c04fe87351c045cdffe61b3c20d62d58e9760b`; compact artifact `results/EXP-060-exact-lift-scaling-campaign-summary.md`.

### EXP-061 — Production-aligned 4/8/12-layer exact-lift scaling (pre-registered)

- **Implementation commit title:** `feat: add production-aligned exact-lift scaling campaign`.
- **Motivation:** EXP-060 established a stable exact-lift advantage through eight replacements, but two tested positions (12 and 39) are retained-attention positions in the final 15/85 schedule. EXP-061 removes this mismatch and spends the next full TPU allocation on stronger initializer evidence rather than an isolated memory probe.
- **Nested production-aligned sets:** `4={0,9,18,30}`, `8={0,3,9,13,18,27,30,38}`, and `12={0,3,6,9,13,16,18,22,27,30,34,38}`. Every position belongs to the frozen 34-layer Mamba schedule; none overlaps retained attention indices `(5,12,19,25,32,39)`.
- **Paired controls:** canonical random Mamba initialization versus `INIT-K-balanced-qkvo-lift`, paired model seeds `123/456/789`, identical token order, optimizer, recovery budget, and validation windows. No flat-QKVO arm is repeated because its negative-control role is already established by EXP-055/056.
- **Recovery budget:** layer 0 receives 8,192 decoder-aware steps; each of the other 11 layers receives 4,096 per arm/seed. Training uses sequence length 32, train offset 327,680, validation offset 28,672, and 256 external validation windows. This is 53,248 updates per arm/seed, or 319,488 matched optimizer updates across two arms and three seeds, and is expected from EXP-060 throughput to occupy roughly 6–7 v5e-8 hours; runtime is an estimate, not a scientific outcome.
- **Primary endpoint:** frozen end-to-end LM-head NLL for nested 4/8/12-layer random and exact compositions. At every count, exact lift must have negative mean paired NLL difference, win at least `2/3` seeds, and have a hierarchical paired-bootstrap 95% upper confidence bound below zero. The global gate requires all three counts and finite outputs.
- **Scientific value:** a pass would extend the exact-lift result to 12 replacements (35.3% of the final 34 Mamba positions) without relying on layers that the deployed hybrid retains as attention. A failure only at 12 would establish an initializer-composition boundary and directly inform staged full-model recovery.
- **Durability:** each completed layer stores a hash-checked endpoint checkpoint and stage manifest. Interior layers run before the longer layer-0 cell. Training/evaluation activation arrays are pruned only after their dependents are durable. Large endpoint payloads are deleted only after the final streamed evaluation and result JSON succeed. Start, completion, and caught-failure Telegram notifications are enabled by default.
- **Artifacts:** download only `extent-production-aligned-scaling-campaign-summary.md`, `extent-production-aligned-scaling-campaign.json`, and `exp061-stage-manifest.json`. Endpoint/activation payloads are regenerable infrastructure, not paper evidence.
- **Interpretation boundary:** this is a strong scaling test of the Mamba initialization method, not a trained 34-Mamba/6-MLA checkpoint. MLA interaction, long context, incremental-decode throughput, and full-model recovery remain separate experiments.
- **Status (MEASURED):** completed on Kaggle TPU v5e-8 in `7.045` hours. All 12 layer endpoint bundles, 72 matched endpoint trajectories, 19 streamed full-model branches, and 256 validation windows completed with finite outputs. The pre-registered global scientific gate passes.
- **Primary result (MEASURED):** exact lift wins `3/3` paired seeds at every production-aligned scale. Mean exact-minus-random NLL is `-0.20600782`, `-0.12278098`, and `-0.16493674` for 4/8/12 replacements. Hierarchical paired-bootstrap 95% intervals are `[-0.26025011,-0.13995209]`, `[-0.22310965,-0.02059079]`, and `[-0.25011441,-0.03233432]`; all upper bounds remain below zero.
- **Effect size (MEASURED):** random/exact mean NLL is `5.094335/4.888327`, `5.236408/5.113627`, and `5.284666/5.119729` for 4/8/12 replacements. Exact lift reduces NLL by `4.04%`, `2.34%`, and `3.12%` relative to random, and removes `21.76%`, `11.28%`, and `14.51%` of random's excess above original-Qwen NLL `4.14765491`.
- **Scaling conclusion:** the advantage does not collapse when moving from eight to twelve replacements and no retained-attention position is used. EXP-061 therefore confirms the initializer on 12 of the final 34 Mamba positions (35.3%) under matched recovery. Together with EXP-056/057/059/060, this is strong replicated evidence for the initialization method; it is still not evidence that the unrecovered 34/6 model matches Qwen.
- **Optimization caveat (MEASURED):** all gradients remain finite and clipping is active, but late-layer raw spikes are severe. Layer 38 reaches maximum pre-clip norms of `2.95e7` for exact lift and `4.11e6` for random; layer 34 reaches `7.83e5` and `1.27e6`. Some final layer-38 norms also remain large. Full-model recovery requires explicit pre/post-clip logging, non-finite step skipping, and a robust-loss or loss-scaling ablation rather than assuming uniformly stable optimization.
- **Representation caveat (MEASURED):** final hidden relative L2 is `7.33%` lower for exact lift at four replacements, but approximately `306%` and `111%` higher at 8 and 12 replacements even though exact lift wins NLL in every paired seed. This strengthens EXP-060's finding that intermediate/final teacher hidden-L2 is not a reliable monotonic surrogate for language-model likelihood under cross-architecture recovery. NLL must remain the primary model-level endpoint.
- **Artifact correction:** the supplied full JSON contains a stale human-readable gate string saying `2/4/8`; the actual registered layer sets, analyzed stages, summary, and Boolean gate are correctly `4/8/12`. The analyzer is corrected to render stage counts dynamically; no numerical value or pass/fail decision changes.
- **Decision:** stop progressive layer-count screens. Exact lift is now the selected Mamba initialization for full materialization. The next engineering milestone is a resumable, sharded 34-Mamba/6-RoRoPE-BKV checkpoint builder followed by a small full-model recovery bring-up. The next scientific comparison should measure tokens-to-NLL for exact lift versus matched random at a smaller tractable whole-model scale, not add another isolated-layer sweep.
- **Execution provenance:** compact-summary SHA-256 `9dbb00e2b7db11469002c3f0d190d9613a198484fe5a4aeb646c27f9f2969900`; full-campaign SHA-256 `8eb8591ff054fc60b3da9b7de7db8f680e4453de46ff72fed0856dea6f955eeb`; compact artifact `results/EXP-061-production-aligned-scaling-campaign-summary.md`.

### EXP-062 — Full 34/6 sharded materialization bring-up (pre-registered)

- **Implementation commit title:** `feat: add sharded full hybrid materialization campaign`.
- **Purpose:** perform the first real, non-shape-only construction of the selected Extent-14B architecture on v5e-8 before checkpoint export or recovery training.
- **Source/calibration:** pinned Qwen3-14B revision `40c069...e18`; WikiText-2 train offset 393,216; 16 independent windows of length 32 (512 tokens) calibrate each retained attention layer. Six target-layer caches are generated with data-parallel teacher inference and BF16 compute. Calibration rank 448 is valid because 512 samples exceed the selected latent rank.
- **Materialization order:** initialize the full parameter tree directly in `data/fsdp/tensor=1/4/2` shards; stream 203 preserved embedding/norm/MLP/head tensors; replace 34 random Mamba subtrees with `INIT-K-balanced-qkvo-lift`; replace six attention subtrees with EXP-019 rank-448 RoRoPE+BKV fits; allocate BF16 Lion momentum using the same parameter layouts.
- **Runtime validation:** compile and execute full-model causal-LM forward losses at sequence lengths 8, 32, and 128. Every loss must be finite. The result records real parameter and optimizer bytes per device, every mixer conversion report, PCA calibration reconstruction error, cache reduction, backend, and duration.
- **Failure containment:** a partial JSON is atomically updated after every materialized layer. Caught failures write stage, exception type, traceback, and completed mixer reports; Telegram reports start, completion, or caught failure. Source weights default to `/dev/shm` so Kaggle disk is not exhausted.
- **Gate:** `34/34` Mamba and `6/6` MLA reports, `203` direct tensors, `443` source tensors, successful Lion allocation, and finite forward probes at all three contexts. This is an engineering GO/NO-GO gate, not a quality comparison.
- **Checkpoint boundary:** EXP-062 deliberately does not serialize/upload the 27.5 GiB resulting tree. Sharded safetensors export and private Hugging Face upload are enabled only after this materialization contract passes, preventing publication of a large invalid artifact.
- **Interpretation boundary:** a pass proves that selected layerwise methods compose into an executable sharded 14B model. It does not prove finite backward gradients, recovered NLL, long-context behavior, generation quality, or inference speed.
- **Artifact:** `extent-full-hybrid-materialization.json` on success; `exp062-materialization-failure.json` on failure. Do not download calibration arrays or Qwen shards.
- **Status (MEASURED):** completed on Kaggle TPU v5e-8 in `0.1534` hours (`9.20` minutes). The campaign reports `passed=true`, all 443 pinned source tensors, 203 direct mappings, exactly 34 exact-lift Mamba mixers, and exactly six RoRoPE/BKV mixers.
- **Real HBM allocation (MEASURED):** every TPU holds `3.4512973 GiB` of BF16 parameters and `3.4512973 GiB` of BF16 Lion state, for `6.9025945 GiB` persistent allocation/device before gradients, activations, executables, and temporaries. This confirms the analytical sharding estimate and leaves about 9.10 GiB of nominal 16 GiB HBM before backward allocations.
- **MLA calibration (MEASURED):** all six layers use 512 calibration tokens and reproduce the frozen 71.875% retained-attention cache reduction. Joint reconstruction relative L2 rises with depth from `0.02505` at layer 5 through approximately `0.04–0.05` at layers 12/19/25/32/39. All fits and mapped tensors remain finite.
- **Full forward execution (MEASURED):** complete 40-layer forward compilation/execution succeeds with finite losses at contexts 8, 32, and 128 in `15.69`, `16.46`, and `19.10` seconds. Probe losses `15.9241/19.6923/18.6049` use synthetic sequential token IDs and are only numerical-health checks; they are not recovery-quality estimates and must not be compared with WikiText NLL.
- **Scientific/engineering conclusion:** the independently selected Mamba and MLA methods compose into one executable, correctly sharded 14.766B model. The previous architecture-compatibility blocker is closed. Full backward remains untested and is the final 14B bring-up gate before shifting controlled recovery work to the Qwen3-1.7B proxy.
- **Execution provenance:** full result SHA-256 `8f7c42b6e5552cd124793f0c84a1f851c670275ed7f2eca2e27c37e8a146a2db`; compact artifact `results/EXP-062-full-hybrid-materialization-summary.md`.
- **EXP-062B backward gate (PRE-REGISTERED):** with the materialization contract fixed, rerun the same builder with only context 8 forward and one full `value_and_grad` probe at context 8 while the sharded BF16 Lion state remains allocated. Cast gradients to the production BF16 gradient contract and reduce only loss, global gradient norm, maximum absolute gradient, finiteness, and the count of non-finite leaves; do not return or serialize the 14B gradient tree and do not update parameters.
- **EXP-062B gate:** PASS requires successful compilation/execution, finite loss/norm/max-gradient, and exactly zero non-finite gradient leaves. OOM is a valid engineering NO-GO. A pass closes numerical/sharding bring-up only; it is not a recovery-training or quality result.
- **EXP-062B result (MEASURED):** PASS on TPU v5e-8. The complete 14.766B graph at context 8 reports loss `23.203165`, BF16 global gradient norm `828,493,760`, maximum absolute gradient `46,466,468`, `grads_finite=true`, and exactly zero non-finite parameter leaves. Backward compilation plus execution takes `96.46` seconds while `3.4513 GiB/device` parameters and `3.4513 GiB/device` Lion state remain allocated.
- **EXP-062B interpretation:** full 14B forward/backward sharding and memory feasibility are closed. The enormous raw shock gradient makes FP32 norm computation and clipping to 1.0 mandatory, but is not itself a failure because every leaf remains finite. Further controlled initialization ablations move to Qwen3-1.7B-Base; the expensive 14B model is no longer the method-development testbed.

### EXP-063 — Mamba-3 in the Qwen (M3Q) homotopy screen (pre-registered)

- **Implementation commit title:** `feat: add M3Q homotopy distillation campaign`.
- **Purpose:** test a new cross-architecture recovery recipe, not merely repeat random-versus-exact initialization on a smaller model. M3Q starts from the operator-preserving exact MIMO lift and makes replacement shock an explicit curriculum: the frozen decoder initially receives teacher-attention output, then continuously transitions to the trainable Mamba-3 output.
- **Pinned source:** `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; 28 dense layers, hidden size 2,048, intermediate size 6,144, 16 query heads, eight KV heads, 128 head dimension, tied embeddings, and 32,768-token advertised context. The single safetensors file contains 310 mapped tensors and 3,441,149,952 tensor bytes.
- **Registered controls and methods:** `CONTROL-RANDOM`, `CONTROL-FLAT-QKVO`, standard `BALANCED-RANK-LIFT`, plus exact-lift M3Q with linear, cosine, and delayed-cosine attention-to-Mamba schedules. The M3Q arms share the identical exact initializer and optimizer; only the bridge schedule changes. Delayed cosine holds alpha at zero through 15% of steps, transitions from 15–70%, and trains fully deployable alpha=1 Mamba for the final 30%.
- **Paired design:** layers `0,6,13,20,27` provide uniform boundary/interior depth coverage; seeds `123/456/789`; sequence length 64; 8,192 recovery steps per arm; checkpoints `0/256/1024/2048/4096/8192`; BF16 compute and gradients; identical calibration, data order, Lion configuration, readout calibration, frozen decoder tail, and validation data within each comparison.
- **No bridge leakage:** teacher attention can participate only in the training curriculum. Every registered held-out checkpoint measures the standalone Mamba-3 replacement with alpha fixed to one, including step zero. Thus an M3Q arm cannot pass by retaining teacher computation at inference.
- **Data boundary:** training uses a pinned WikiText-2 train slice starting at token 393,216. Evaluation uses the disjoint validation split from offset zero. Each target layer receives its true residual stream and frozen attention target from the complete upstream Qwen3-1.7B teacher.
- **Endpoints:** primary endpoint is held-out frozen-decoder output relative L2 at step 8,192. Sample efficiency is normalized trapezoidal AUC across all six held-out deployable checkpoints. Lower AUC means less recovery compute, preventing one favorable endpoint from defining the result.
- **Schedule selection and exploratory gate:** select one global M3Q schedule by lowest mean AUC across all five layers, never a different best schedule per layer. The selected schedule must beat standard exact lift at at least four of five layer gates, in across-layer mean final error, and in across-layer mean AUC. A layer gate additionally requires at least two of three paired-seed final wins. A pass advances the recipe to a fresh-data locked confirmation; it is not itself confirmatory evidence because schedule selection and screening use the same campaign.
- **Resilience:** per-layer results and a stage manifest are durable; completed layers resume; regenerable activation arrays are pruned only after their result is saved; a 7.25-hour deadline returns a compact partial artifact; Telegram reports start and final/failed state.
- **Next scientific stage:** if M3Q passes fresh-data confirmation, apply the fixed schedule to the full 24-Mamba/4-GQA Qwen3-1.7B hybrid with end-to-end teacher KL. Only after that isolate MLA in a factorial comparison (GQA hybrid versus MLA hybrid), then confirm the selected recipe once at 14B. This separates the claimed transplant contribution from MLA and scale effects.
- **Completed result (MEASURED):** the campaign completed all five layers and all numerical checks in `2.777` hours on TPU v5e-8. Cosine had the lowest across-layer mean M3Q AUC and was therefore selected globally as registered. It passed only `1/5` layer gates, so the exploratory advancement gate is false. Relative to standard exact lift, its final/AUC gains were layer 0 `-2.03%/-2.50%`, layer 6 approximately `-0.00%/-0.01%`, layer 13 `+0.01%/-0.01%`, layer 20 `+0.46%/-2.22%`, and layer 27 `+12.39%/+17.50%`.
- **Scientific decision:** reject one globally fixed attention-to-Mamba homotopy as an M3Q component. The sole material win is layer 27, which the registered 1.7B hybrid retains as attention; homotopy worsens layer 0, which is replaced by Mamba. This negative result does not reject exact MIMO lift or end-to-end recovery. It redirects the next experiment to full-model staged hidden-state then prediction distillation, paired against ordinary end-to-end KL from the same exact initialization.
- **Artifact integrity:** full artifact SHA-256 `2e7003eb4804efb095ee6e4dcf2ae4cbe98162eb6af3aaad116497642aa7bcca`; summary SHA-256 `d8e420a1384923e85bb2a901750b8c5b8da7e0a8653bb28135c2fe11c944e5e8`.

### EXP-064 — Qwen3-1.7B full-model M3Q distillation (pre-registered)

- **Implementation commit title:** `feat: add Qwen3-1.7B full-model M3Q distillation`.
- **Purpose:** test whether staged hidden-state alignment adds recovery value beyond ordinary end-to-end prediction distillation after exact MIMO initialization. This is the first whole-model M3Q method test; unlike the layer-local screens, gradients flow through the complete 28-layer student.
- **Isolation contract:** the student has 24 exact-lift Mamba-3 layers and four source-faithful retained Qwen GQA layers at `6,13,20,27`. MLA is deliberately absent so an attention-compression effect cannot be misattributed to the Mamba transplant/recovery method. Shape-only preflight reports `1,675,304,064` parameters and `1.174 GiB/device` for BF16 weights, gradients, and Lion state before teacher/activation memory.
- **Online teacher:** the full pinned Qwen3-1.7B teacher is independently initialized into the same `data=1, fsdp=4, tensor=2` mesh and receives all 310 checkpoint tensors through a complete one-to-one streaming import. Teacher logits and hidden states remain on accelerator and are never serialized into an infeasible offline cache.
- **Registered arms and order:** `EXACT-KL`, `M3Q-HIDDEN-BRIDGE-KL`, then contextual `RANDOM-KL`. The first two begin from byte-equivalent exact MIMO/GQA/direct-weight initialization and use identical tokens, Lion schedule, and data order. The M3Q arm jointly minimizes prediction KL, true-token cross entropy, and scale-free hidden-state error over all 24 replaced layers for the first 25% of steps, then uses the same prediction objective as `EXACT-KL`. `RANDOM-KL` leaves only Mamba mixers random while preserving Qwen direct weights and GQA.
- **Training protocol:** sequence length 256; 3,072 steps per arm; BF16 weights, gradients, and Lion momentum; FP32 loss reductions and gradient-health diagnostics; temperature 2.0; cross-entropy weight 0.1; learning rate `3e-5`, 128 warmup steps, zero weight decay, and global gradient clipping at 1.0. Training uses one deterministic pass over 3,072 WikiText-2 train windows from token offset 393,216.
- **Evaluation:** checkpoints `0,128,512,1024,2048,3072` use eight separately executed contiguous validation windows from offset zero. Metrics are teacher NLL, student NLL, excess NLL, teacher-to-student prediction KL, top-1 agreement, and numerical finiteness. Eight single-window executions bound peak logit memory while preserving a 2,040-token held-out aggregate.
- **Primary gate:** the complete M3Q arm must beat complete `EXACT-KL` in final held-out excess NLL, final prediction KL, and normalized excess-NLL AUC. Exact excess NLL must remain positive. `RANDOM-KL` is contextual and may be omitted only if the hard 7.25-hour wall deadline occurs after the primary pair; a partial primary pair can never pass.
- **Resilience and scope:** compact metrics are mirrored after step zero and every checkpoint; exceptions create `exp064-failure.json`; Telegram reports start, completion, deadline-partial, or failure. No multi-gigabyte model checkpoint is written. A pass selects staged M3Q for a fresh-data replication and only then an MLA factorial; it is not yet a 14B quality claim.
- **Completed result (MEASURED):** all three arms completed 3,072 steps on TPU v5e-8 in `0.979` hours. The scientific gate failed. `EXACT-KL` ended at excess NLL `15.09372`, KL `19.90180`, agreement `0.00049`, and normalized excess-NLL AUC `14.63696`. `M3Q-HIDDEN-BRIDGE-KL` ended at excess NLL `15.30315`, KL `20.25145`, zero measured top-1 agreement, and AUC `16.47394`. Thus hidden staging is worse than ordinary exact-init KL by `1.388%` at the final excess-NLL endpoint and by `12.550%` in AUC.
- **Contextual random result (MEASURED):** `RANDOM-KL` ends at excess NLL `10.87794`, KL `15.94475`, agreement `0.00980`, and AUC `11.04340`; it beats `EXACT-KL` by `27.932%` in final excess NLL and `24.550%` in AUC. This is a single deterministic full-model run rather than a paired multiseed claim, but the margin is too large to continue treating the current QKVO exact lift as the production recovery method.
- **Scientific decision:** reject the implemented hidden-state staging and demote `INIT-K-balanced-qkvo-lift` to a required negative/control baseline. Layer-local wins from EXP-056–061 do not compose into a useful whole-model initializer under end-to-end KL. Further TPU time must target Mamba-3-specific dynamics rather than additional training of the same QKVO lift.
- **Artifact integrity:** full artifact SHA-256 `a94cb2e1a872043be326d7504a60edf337e051a4e3d2fbd03548c5612b3e3402`; compact repository artifact `results/EXP-064-qwen17-full-model-m3q-summary.md`.

### EXP-065 — Mamba-3 exact-dual complex bridge (pre-registered)

- **Implementation commit title:** `feat: add Mamba3-aware complex bridge campaign`.
- **Hypothesis:** a Transformer should not be folded from a generic learned linear-attention feature map into Mamba-3 after the fact. Instead, the intermediate block should already use the canonical Mamba-3 MIMO parameterization—rank-specific value/gate/readout factors, complex B/C rotation, heavy-tail decay, input-dependent timestep, and trapezoidal injection—while executing its SSD dual quadratically during teacher-facing preparation.
- **Zero-shock contract:** the training-only dual block and deployable recurrent block consume exactly the same parameter tree. The transition changes only execution mode; it performs no matrix fitting, channel resize, parameter copy, or approximation. An automated FP32 shape/parity test requires dual and recurrent outputs to agree, and every BF16 experimental arm records bridge-to-recurrent relative L2 before recovery.
- **Relation to prior work:** Apple HedgeMamba is retained as the generic linear-feature-map baseline family, while MOHAWK is retained as the matrix-orientation baseline family. EXP-065 tests a distinct Mamba-3-specific intervention: exact canonical MIMO/complex/trapezoidal dual preconditioning. The experiment is a novelty screen, not by itself a novelty claim; a positive result still requires a dedicated related-work audit and direct locked comparisons.
- **Pinned proxy and design:** `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; layers `0,6,13,20,27`; seeds `123/456/789`; sequence length 96; disjoint train offset `786,432` and validation offset `65,536`; 12,288 decoder-aware recovery steps with checkpoints `0/256/1024/3072/6144/12288`.
- **Registered arms:** canonical random; balanced QKVO exact lift; exact-dual preconditioning from random; exact-dual preconditioning from exact lift with complex phase disabled; and the complete exact-dual preconditioning from exact lift. Each dual arm receives 3,072 separately reported preparation steps before identical readout calibration and recovery. Construction compute is not hidden inside recovery-token accounting.
- **Mechanism ablations:** `M3Q-DUAL-RANDOM` tests whether QKVO seeding still contributes after Mamba-aware preparation. `M3Q-DUAL-EXACT-NO-COMPLEX` freezes only the angle projection during preparation, then restores canonical trainability for recovery; its comparison with the complete method tests whether complex preparation itself matters rather than merely adding optimization steps.
- **Primary endpoints:** held-out frozen-decoder relative L2 at step 12,288 and normalized recovery AUC. The complete dual method must beat both canonical random and balanced exact lift in final error, AUC, and at least two of three paired seeds at four of five depths. Mechanism support additionally requires complex phase or exact-lift seeding to win at three of five depths. All thresholds are fixed before TPU execution.
- **Resource and resilience contract:** one invocation has a hard `7.75`-hour deadline, strictly below the 9-hour session maximum. Every completed layer is an atomic resume boundary; partial/failure JSON and the stage manifest survive ordinary Python exceptions; regenerated activation arrays are pruned only after durable layer results; Telegram reports start and terminal state. The output is compact JSON plus a human-readable summary, not model checkpoints.
- **Decision ladder for the remaining TPU budget:** session 1 runs EXP-065 only. If its primary gate passes, session 2 moves the selected dual recipe to paired full-model Qwen3-1.7B against random and exact lift. If it fails, session 2 is redirected to diagnosing the measured failure (phase, decay/trapezoid, or QKVO seeding) rather than blindly scaling it. MLA remains excluded until the Mamba transplant method survives full-model confirmation.
- **Completed result (MEASURED, deadline-partial):** four of five target layers (`0,6,13,20`) completed all five arms and all three seeds in `8.545` hours; layer 27 did not complete, so the registered five-layer scientific gate is false by construction. All numerical and exact-dual/recurrent parity checks pass. BF16 bridge-to-recurrence relative L2 remains between zero and `8.31e-4`, confirming that the execution switch itself introduces negligible shock.
- **Primary-method decision:** reject exact-QKVO-seeded dual preparation. `M3Q-DUAL-EXACT` beats random at only two of the nominal five gates and balanced exact lift at only one; more directly, exact seeding loses to random seeding after dual preparation at every completed depth (`0/4` wins). This independently agrees with EXP-064's whole-model evidence that current QKVO lift transfers a harmful optimization prior.
- **Positive mechanism evidence:** complete complex preparation beats the no-complex ablation at all four completed depths. At layer 20 seed 789, complex preparation prevents a severe outlier (`0.51484 -> 0.11204` final decoder L2). This is exploratory mechanism evidence because the primary method failed and layer 27 is missing, but it justifies retaining complex Mamba-3 dynamics in the next method.
- **Random-dual evidence:** `M3Q-DUAL-RANDOM` is the best layer-0 arm (`0.48706` versus random `0.69554`, a `29.97%` final improvement) and reduces layer-20 recovery AUC from `0.17728` to `0.11038` (`37.74%`) while ending close to balanced lift. It is worse at layers 6 and 13. The effect is therefore real but depth-dependent, not a universal initializer yet.
- **Failure diagnosis:** bridge starting losses grow from approximately `2` at layer 0 to millions at layer 20 even though the inputs are RMS-normalized. Readout fitting currently occurs only after dual preparation. The next registered intervention moves readout calibration before the bridge and replaces global output normalization with frozen teacher per-token RMS whitening; both target the measured depth-scale pathology without changing Mamba capacity or adding recovery tokens.
- **Runtime correction:** the nominal `7.75`-hour check was exceeded because activation-cache construction is an indivisible operation and can begin just before the deadline. Future campaigns reserve cache-build headroom and check the deadline again after validation-cache construction; a Python-loop deadline alone is not described as a hard runtime guarantee.
- **Artifact integrity:** full artifact SHA-256 `c05cf63c39ba2911f514368d13d83313de56ca98a7c796d25bd872f48bf0782a`; summary SHA-256 `f7371c893c8067f7a1582219397ea851201521d5fe5f6b9985db9888d9466fbc`; compact repository artifact `results/EXP-065-m3q-complex-bridge-summary.md`.

### EXP-066 — scale-stabilized random exact-dual bridge (pre-registered)

- **Implementation commit title:** `feat: add scale-stabilized exact-dual bridge campaign`.
- **Hypothesis:** EXP-065's depth dependence is caused partly by training canonical Mamba-3 dynamics through an uncalibrated output projection and a globally normalized target. A dual bridge seeded from random Mamba parameters, fitted once at its readout before dynamics training, and optimized with frozen teacher per-token RMS whitening should retain the layer-0/20 benefits without harming interior layers.
- **Registered arms:** canonical random; balanced QKVO exact lift; the original uncalibrated dual-random recipe; stabilized dual-random with complex phase disabled; and complete stabilized complex dual-random. The primary comparison therefore isolates stabilization from extra bridge compute, while the no-complex pair retains the positive EXP-065 mechanism test.
- **Design:** Qwen3-1.7B-Base; layer order `27,20,13,6,0` so the missing depth and the strongest non-boundary signal are measured first; seeds `123/456/789`; sequence length 96; 8,192 recovery steps; 2,048 separately accounted dual-preparation steps; checkpoints `0/256/1024/2048/4096/8192`; disjoint validation offset `131,072`.
- **Primary gate:** complete stabilized complex dual-random must beat canonical random, balanced lift, and legacy dual-random in final decoder L2, normalized AUC, and at least two of three paired seeds at four of five layers. Complex phase must beat its stabilized no-complex pair at at least three layers. The strict gate prevents a layer-0-only win from defining the method.
- **Runtime/resilience:** one invocation uses a `7.25`-hour campaign budget and reserves 35 minutes before beginning a new layer's indivisible cache build. It checks time again after each cache. Completed layer JSON is the resume boundary; partial results and Telegram terminal status are retained. No large checkpoint is produced by this layerwise screen.
- **Checkpoint policy:** private Hugging Face checkpoint upload is deferred to the first full-model method that passes its layerwise gate. At that point the durable payload will contain student parameters, BF16 Lion state, step/data cursor, configuration, hashes, and compact metrics. Regenerable activations and source Qwen shards will remain excluded.
- **Completed result (MEASURED):** all five layers and all registered arms complete in `3.984` hours with finite BF16 training and parity checks. The scientific gate fails decisively: stabilized complex beats random at `0/5`, exact lift at `1/5`, legacy dual-random at `1/5`, and its matched no-complex arm at `0/5` registered layer gates.
- **Stabilization decision:** reject pre-bridge readout calibration plus token-whitened output loss. The intervention erases the useful layer-0 and layer-27 basins rather than making them depth-invariant. At layer 27 it ends at `1.19166` versus `0.07574` for legacy dual-random; at layer 0 it ends at `0.73134` versus `0.52671`. Lower construction loss is not a sufficient compatibility metric and must not be used to select a method.
- **Confirmed depth dependence:** legacy random exact-dual preparation remains strongly positive at layer 0 (`25.15%` final improvement over random) and extraordinarily positive at layer 27 (`93.00%` over random and `95.23%` over exact lift). It is close to exact lift at layer 20 and loses at layers 6/13. The effect replicates EXP-065 on fresh validation offset and shorter recovery, so it is not explained by one seed or one endpoint.
- **Architectural implication:** a uniform-depth 15% attention schedule is no longer scientifically justified. Under a fixed 15% budget, attention should be retained in layers with the worst measured Transformer-to-Mamba compatibility, while highly compatible layers such as 0 and 27 should be converted. This converts family/depth-dependent transfer failure into a measurable allocation rule rather than hiding it with more global distillation.
- **Next experiment:** construct a full 28-layer compatibility atlas using the legacy random exact-dual recipe and paired random control, then freeze the four retained Qwen attention layers (14.3%, the nearest discrete 1.7B proxy for 15%) by a pre-registered ranking rule. Confirm top/bottom-ranked layers with additional seeds before any full-model run. The atlas must test whether a cheap early-recovery score predicts final recovery gain; otherwise allocation would require prohibitively training every candidate.
- **Artifact integrity:** full SHA-256 `d028fb20d1428ae54b19e4505aa60610d1f4ec821680f18e303f05b4ee0f3e86`; user summary SHA-256 `5d1608b330de088357c6b3d0df7bfe649719bbee84401a17525c6d771226818e`; compact artifact `results/EXP-066-m3q-stabilized-bridge-summary.md`.

### EXP-067 — full-depth Mamba compatibility atlas (pre-registered)

- **Implementation commit title:** `feat: add resumable full-depth compatibility atlas`.
- **Motivation:** EXP-065/066 show that random exact-dual Mamba-3 preparation is strongly beneficial at layers 0 and 27, useful near layer 20, and harmful at layers 6/13. A uniform retained-attention schedule would ignore this replicated depth dependence. EXP-067 measures every Qwen3-1.7B layer before freezing the 15/85 proxy allocation.
- **Frozen scope:** all 28 Qwen3-1.7B-Base layers, paired seeds `123/456`, canonical random versus legacy `M3Q-DUAL-RANDOM`, sequence length 64, 2,048 recovery steps, 1,024 separately accounted dual-preparation steps, checkpoints `0/256/1024/2048`, and 32 held-out validation windows. The broad layer order starts `27,0,20,6,13,...` so a deadline-partial session still extends the existing evidence rather than repeating adjacent depths.
- **Allocation rule:** rank layers by mean held-out step-2,048 dual decoder-output relative L2 and retain attention in the four hardest layers. Four of 28 layers is 14.29%, the nearest discrete proxy for the fixed 15% production budget. The rule is frozen before observing atlas results; no manual aesthetically spaced schedule may replace it after the fact.
- **Efficiency question:** compute Spearman correlation between the step-256 and step-1,024 difficulty rankings and the final step-2,048 ranking. The registered predictor gate requires all 28 layers and step-1,024/final Spearman at least 0.80. This tests whether future architecture allocation can use a cheap recovery probe instead of fully training every candidate layer.
- **Durability contract:** every completed layer is written as compact JSON and immediately uploaded to the private Hugging Face dataset under `experiments/exp067-compatibility-atlas/layers/`. A fresh Kaggle session restores those artifacts and skips completed layers when the identical campaign cell is rerun. HF errors are non-fatal to active TPU computation, but are printed and retained in the local result.
- **Runtime contract:** one invocation uses at most 6.75 hours and reserves 20 minutes before starting a new activation-cache build. This leaves roughly 30 minutes inside the currently available 7.3-hour TPU session for final serialization, HF synchronization, and Kaggle overhead. The campaign is intentionally resumable across session boundaries; partial completion is a valid checkpoint, not a scientific result.
- **Scientific boundary:** the atlas selects which layers remain attention under a fixed budget. It does not yet establish recovered full-model NLL, prove that the selected four-layer hybrid beats an evenly spaced schedule, convert GQA to MLA, or validate long-context inference. Those require a locked full-model comparison after the atlas.
- **Artifacts:** `extent-m3q-compatibility-atlas.json` and `extent-m3q-compatibility-atlas-summary.md`, mirrored to the HF dataset as `latest.json` and `latest-summary.md`. Regenerable Qwen shards and activation caches are not uploaded.
- **Completed result (MEASURED):** all `28/28` layers and both paired arms complete in `1.749` TPU hours with finite outputs. The registered early-predictor gate passes: step-1,024 versus step-2,048 layer-difficulty Spearman is `0.9978106`.
- **Frozen allocation result (MEASURED):** by the pre-registered absolute final dual-error rule, the four retained-attention layers are `[0,1,20,26]`. This is not an evenly spaced schedule. Layers 0 and 1 remain the hardest replacements despite dual preparation improving them over random; layers 20 and 26 narrowly exceed nearby layers 27/25/24 in final absolute error.
- **Depth mechanism (MEASURED):** dual preparation improves final error at layers 0–2 and 19–27 (apart from only a small gain at 19), while generally losing to ordinary random recovery at layers 3–18. Its relative advantage becomes extreme at layers 24–27 (`94.29–97.87%` except layer 27 at `94.85%`). This confirms a structured depth regime rather than noisy isolated failures.
- **Interpretation:** early 1,024-step screening appears sufficient to rank layer compatibility cheaply, but the final four-layer allocation is not frozen for full-model use until a fresh-seed, longer-horizon confirmation tests whether the close 20/26/27 boundary changes.
- **Artifact integrity:** full SHA-256 `754b815bbd29d855acd3eda08ce6b269ae8d25c68f61edb932feae9a9edda503`; summary SHA-256 `acc7cda6c1ac063817cf8364e6fb2021688dff8a2a15f91100cba93df418f8f5`; compact artifact `results/EXP-067-m3q-compatibility-atlas-summary.md`.

### EXP-068 — independent long-horizon compatibility confirmation (pre-registered)

- **Implementation commit title:** `feat: add long-horizon compatibility confirmation`.
- **Question:** does the complete EXP-067 depth ranking survive a new model seed, fresh disjoint text, and four times more recurrent recovery, or was the selected `[0,1,20,26]` schedule an early-training artifact?
- **Frozen design:** all 28 Qwen3-1.7B layers; seed `789`; canonical random versus `M3Q-DUAL-RANDOM`; 8,192 recovery steps; 2,048 separately counted dual-preparation steps; checkpoints `0/256/1024/4096/8192`; sequence length 64; train offset `1,441,792`; validation offset `229,376`.
- **Primary confirmation gate:** all 28 layers complete, EXP-067 versus EXP-068 final difficulty Spearman at least `0.80`, and at least three of four retained-attention layers overlap. The four-layer selection is recomputed using the unchanged absolute final dual-error rule.
- **Secondary endpoint:** step-1,024 versus step-8,192 rank correlation measures whether the cheap allocation probe remains valid over a four-times-longer horizon.
- **Runtime/durability:** 4.5-hour maximum inside the remaining five-hour session. Every completed layer is uploaded to `experiments/exp068-long-horizon-compatibility-confirmation/layers/`; rerunning the same cell restores and skips completed layers. Final/failure snapshots and Telegram terminal status are enabled.
- **Interpretation boundary:** a pass freezes the small-model attention allocation for a subsequent selected-versus-even full-model comparison. It does not itself show an NLL or throughput advantage and does not yet decide the separate 14B layer indices.
- **First execution (MEASURED, deadline-partial):** completed `12/28` layers in `4.232` hours and uploaded every completed layer to HF. All trajectories are finite. The remaining layer order begins at layer 5 and is recoverable by rerunning the same entry point in a fresh session.
- **Partial ranking evidence (MEASURED):** across the 12 shared completed layers, EXP-067 step-2,048 versus EXP-068 step-8,192 final difficulty has Spearman `0.979021`; within EXP-068, step-1,024 versus step-8,192 Spearman is exactly `1.0`. This strongly supports ranking stability but cannot pass the pre-registered all-28-layer gate.
- **Partial depth replication (MEASURED):** dual preparation beats random at layers `0,1,17,20,22,24,26,27` and loses at `3,6,10,13`, reproducing the boundary/depth structure on a fresh seed and four-times-longer recovery.
- **Correction:** the first partial summary printed retained-layer overlap `0/4` because incomplete atlases intentionally emit no four-layer selection. This is not measured disagreement. Reporting now marks overlap as pending until all 28 layers complete; no numerical result changes.
- **Artifact integrity:** full partial SHA-256 `8c6d4df2b247dacb59ca8bfa5c06ce16fe4cc5bdffb1f121b5c001272309f5b4`; summary SHA-256 `d64ef6271ec9b208215a728ca8ec9f11203de6fbdeb69cad683c4dbb067686ef`; compact artifact `results/EXP-068-long-horizon-compatibility-partial-summary.md`.
- **Second execution (MEASURED, deadline-partial):** HF resume restored all first-run layers and completed 11 more in `4.172` hours, bringing cumulative coverage to `23/28`. EXP-067/068 rank Spearman is now `0.987154`; EXP-068 step-1,024/final Spearman is `0.999012`. The remaining layers are `18,9,16,11,14`; the retained-attention set and confirmation gate correctly remain pending.
- **Second-execution integrity:** full SHA-256 `596ca6e20764a874393e0b49cf359a8c20f8cb85ef052fe562ab1ce01e5936b6`; summary SHA-256 `a85f84c64d5c5a349fd2d3e0d03eaf11cb81d2b8713bb4dc67b9ef94432a2ff3`.

- **Final execution (MEASURED):** all 28 layers complete. The last invocation took `1.302316` hours, restored 23 completed layers and reports uploading the final five to HF. Total reported runtime across three invocations is approximately `9.706` hours. Numerical, early-predictor and independent confirmation gates pass.
- **Final ranking (MEASURED):** EXP-067/068 Spearman is `0.9901477833`; within EXP-068, step-1,024 versus step-8,192 Spearman is `0.9994526546`. The selected set is `[0,1,26,27]`, overlapping EXP-067's `[0,1,20,26]` at three of four positions and satisfying the registered threshold. Layer 27 (`0.08519217`) narrowly exceeds layer 20 (`0.08446497`) in final dual decoder relative L2; the experiment does not establish statistical separation of those two positions.
- **Final method scope:** dual preparation wins final decoder L2 at 14/28 layers (`0–2,17–27`) and loses at 14/28 (`3–16`). A high difficulty-rank correlation supports a reproducible screening score under these conditions; it does not prove that this score measures whole-model importance or improves NLL. Residual-stream scaling may contribute to depth ranking. Preparation compute differs between dual and random and must be included in efficiency comparisons.
- **Decision:** close EXP-068; no further repetitions are needed for this registered confirmation. Preserve the cheap-probe set `[0,1,20,26]` and the long-horizon candidate `[0,1,26,27]` for a locked whole-model comparison against evenly spaced attention, using fresh evaluation data. This keeps the independently confirmed cheap-probe rule distinct from the set reselected on confirmation data.
- **Final integrity:** full SHA-256 `9a3593290a453f1c3e7aab132ddd85497bdf1af001b191c0a500c34500b85e20`; summary SHA-256 `e612689c5cfef4c3c0f5450f38415815435b78478e1f1e2a2fe5fe5818943514`; final interpretation in `results/EXP-068-long-horizon-compatibility-final-summary.md`.

### EXP-069 — paired full-model attention allocation (pre-registered)

- **Implementation commit title:** `feat: add resumable full-model allocation comparison`.
- **Question:** compare EXP-068's `[0,1,26,27]` with EXP-064's evenly spaced `[6,13,20,27]` under the same new random-dual preparation recipe. EXP-064 compared initializers with one fixed placement; it cannot isolate the effect of placement under the new recipe.
- **Controls:** pinned Qwen3-1.7B-Base; two paired seeds `123/456`; four retained GQA and 24 Mamba mixers per arm. All common Mamba positions consume the same checkpoint bytes, documented by hashes. Execution order is reversed for the second seed. Direct Qwen weights, optimizer, training order, context length, and evaluation are matched.
- **Preparation:** one bank of the 27 Mamba positions needed by either arm, separately per seed; 1,024 dual steps, readout calibration, then 2,048 recurrent decoder-aware steps per layer at context 64. Each model consumes 24 such endpoints. No random-control or QKVO-lift sweep is repeated.
- **Full recovery:** 3,072 updates per arm at context 256, batch one; all student parameters train with BF16 Lion, LR `3e-5`, warmup 128, no decay, FP32 clipping norm 1.0, KL temperature 2 plus CE weight 0.1. Teacher runs frozen on the same TPU host as supported by EXP-064. GQA remains unchanged to isolate placement.
- **Evaluation/gate:** common WikiText-2 test prefix of 64 windows, separate from the atlas validation slices; metrics at `0/256/1024/2048/3072`, including window NLL. ATLAS must beat UNIFORM in final whole-model NLL and NLL AUC for both seeds. This closes a placement comparison, not universality, MLA, or long-context quality.
- **Durability:** HF stores prepared parameters and full parameters/Lion/step/data contract/metrics. Restore validates hashes and configuration and resumes the same deterministic data order. Full states live in RAM-backed storage and sync at steps 0, every 1,024 steps, final step and controlled deadline. A numerical failure leaves the previous durable checkpoint intact. The seven-hour soft wall budget reserves 20 minutes for finalization; external hard termination cannot execute callbacks.
- **First TPU result (reviewed 2026-09-06):** failed after 1.234301 hours, seed-123 UNIFORM attempted update 106 (105 completed). Only step-0 evaluation: student NLL 19.655661 versus teacher 2.805075; excess 16.850586. No completed allocation pair, so no placement conclusion. Latest log events confirm 51/54 prepared endpoints uploaded plus full step-0 weights/Lion; seed-456 layers 21/22/23 lack confirmed upload. Failed scalar metrics were not recorded by the old guard, so the numerical cause remains unresolved. Added JSON-safe failure diagnostics without changing the registered training recipe. Details: `results/EXP-069-allocation-failure-summary.md`; protocol: `results/EXP-069-allocation-protocol.md`.

EXP-069 diagnostic update (2026-09-07): rerun at `e66b697` restored 51 preparations and full step 0, rebuilt/uploaded the other three, then reproduced failure on update 106 in 0.213792 hours. Only `grad_norm` was infinite; all gradient elements and loss were finite (max gradient 1.6360293e18, loss 26.594046). Identified FP32 sum-of-squares overflow; corrected the shared norm/clipping with a maximum-scaled fallback while preserving the ordinary finite reduction. Numerical implementation version is recorded. This does not resolve the underlying large-gradient concern or establish a placement result. Details and source checksum: `results/EXP-069-allocation-failure-summary.md`.

EXP-069 completed result (reviewed 2026-09-07, revision `a385256`): all four arms completed 3,072 updates in a resumed 1.604789-hour session, with final HF uploads confirmed in the log. Final NLL UNIFORM/ATLAS: seed 123 = 13.268912/25.793214; seed 456 = 21.141170/16.397492, versus teacher approximately 2.805. ATLAS-minus-UNIFORM final deltas +12.524302/-4.743678, AUC deltas +9.292229/+7.041444: registered gate fails. Local compatibility ranking has not shown a reliable whole-model allocation benefit. Norm-overflow fix enabled completion, but all arms worsen initially and full recovery remains poor; finite execution is not recovery success. Do not repeat the allocation sweep unchanged. Next priority is diagnosing composition/gradient-scale and recovery-schedule failure with one fixed placement. Details, learning curves, durability and source checksum: `results/EXP-069-allocation-completed-summary.md`.

### EXP-070 — Frozen composition and matched-input diagnosis (pre-registered)

- **Motivation:** EXP-069 showed poor whole-model recovery despite finite training; user hypothesis H3.4 motivates testing whether prepared blocks fail on hybrid-generated inputs. No stabilization method is claimed yet.
- **Scope:** restore EXP-069 seed-123/456 prepared endpoints, retain the final UNIFORM GQA plan, and compose nested 0/1/4/12/24 replacement prefixes. Measure two paired validation-text windows at lengths 64/256 (40 probes). No training or optimizer updates; final trained models are not the target of this initial diagnosis.
- **Measurements:** held-out NLL/KL; per-parameter and packed-Mamba-component gradient norms; gradient concentration by layer/submodule; input/activation RMS; and decoder-contribution errors with teacher and student evaluated on the same teacher inputs versus the same hybrid inputs. Save absolute error and target scale alongside relative L2. Local-versus-full execution controls expose BF16/compiler discrepancies.
- **Decision boundary:** increased error on hybrid inputs supports distribution sensitivity, not a demonstrated improvement from sequential calibration. Count and position are confounded in the nested diagnostic path. A later matched recovery comparison is required for any method claim.
- **Operations:** validated HF preparation restore, per-probe JSON/Markdown output sync, completed-probe resume, Telegram start/final notifications, seven-hour soft deadline with reserve. Runtime is unmeasured; no promise of filling a 5–8-hour session. Protocol and final-cell entry point: `results/EXP-070-input-shift-protocol.md`.

EXP-070 first TPU attempt: v1 at `ec26585` stopped after ~2.27 minutes with only one all-GQA baseline saved; no replaced-layer measurements. Recorded BF16 teacher/student NLL difference -0.00159144 and KL 0.00310707 were nonzero despite identical weights; the next failed probe's metrics were not persisted. The source of the discrepancy remains unconfirmed. Version 2 adds an explicit FP32 forward math control, retains BF16 numerical-background metrics, and saves rejected baselines before raising. Separate `exp070-v2` outputs/HF namespace avoid mixing acceptance rules. Details: `results/EXP-070-baseline-failure-summary.md`. No input-shift hypothesis has been tested yet.

EXP-070 v2 TPU attempt: stopped after ~2.33 minutes with 0/40 accepted probes and no Mamba measurement. The all-GQA FP32 control had NLL difference +0.00109506, KL 0.000131511, top-1 agreement 1.0 and logit relative L2 0.00211660, failing the pre-set overly strict control. V3 uses highest matmul precision, adds per-layer hidden comparisons, and converts the control into an observed-background corruption guard rather than an equality claim. New thresholds are post-v2 and cannot serve as independent equivalence evidence. V3 artifacts have a separate namespace. Details/checksum: `results/EXP-070-v2-fp32-control-summary.md`.

EXP-070 v3 completed result: 40/40 frozen probes in 0.519455 hours at revision `ef700b1`. Highest-precision all-GQA controls pass (maximum hidden relative L2 `4.28e-7`, logits relative L2 <=`7.66e-6`); real BF16 numerical-background KL remains about `0.0031`. Hybrid/teacher-input decoder error-RMS median ratio grows from `1.00` at one replacement to `1.05/2.16/17.55` at 4/12/24; mean ratio at 24 is `85.73`, and `95%` of pooled layer/cell ratios exceed one. Hybrid input RMS reaches `27,343.58` at layer 26 versus teacher-stream scale around tens; raw full-objective gradient norms reach `1.21e15`. Value+gate carry approximately `94%` of packed Mamba in-projection squared gradient at 24 replacements; dt/decay/angle are not the direct gradient majority. Input-distribution sensitivity is therefore supported, but deep endpoints are also poor on fresh teacher inputs, so sequential calibration is motivated rather than proven sufficient. Next matched experiment compares equal-budget teacher-input, hybrid-input and mixed-input sequential recovery from identical endpoints. Full analysis and artifact checksum: `results/EXP-070-completed-summary.md`.

### EXP-071 — Matched sequential input-distribution recovery (pre-registered)

- **Implementation commit title:** `feat: add matched sequential recovery campaign`.
- **Causal question:** EXP-070 measured rapidly growing error under hybrid-generated residual inputs. EXP-071 changes only that training distribution while matching starting endpoint, text and optimizer updates, testing whether the observed mechanism is actionable rather than merely correlated with failure.
- **Arms:** `TEACHER` receives natural Qwen residuals; `ONPOLICY` receives its own sequential hybrid-prefix residuals; `MIXED` receives equal pools of both. The frozen Qwen decoder block supplies conditional targets to all arms through the same standalone execution path and batch shape.
- **Scope:** Qwen3-1.7B-Base, seeds `123/456`, first 12 sequential non-attention positions under retained GQA `[6,13,20,27]`, 1,024 current-Mamba-only Lion updates per layer, BF16 weights/compute/gradients. The FP32 relative-MSE objective subtracts the shared residual identity and therefore scores the learned decoder contribution rather than allowing a large input to hide mixer error. Every current layer begins from the byte-identified matching EXP-069 endpoint; earlier endpoints and all Qwen modules are frozen.
- **Data/evaluation:** 128 length-64 training windows on a new pinned train offset; eight length-256 WikiText-2 test windows at a disjoint offset. Whole-model NLL/KL/agreement is measured after 1/4/8/12 replacements; original EXP-069 endpoints provide no-extra-update references.
- **Primary gate:** at both seeds, `MIXED` must beat equal-update `TEACHER` in final held-out NLL and milestone NLL-delta AUC. `ONPOLICY` diagnoses whether pure deployment adaptation helps but catastrophically forgets the natural stream.
- **Durability:** every completed layer is an exact-contract private-HF checkpoint. Failed upload halts before dependent stages; rerunning the same entry point restores progress. The eight-hour soft budget reserves 20 minutes inside a nine-hour Kaggle session. Compact JSON/Markdown and Telegram terminal status are automatic.
- **Boundary:** a pass supports sequential mixed-distribution recovery over 12 replacements. It does not yet prove full 85% recovery, scale transfer to 14B, MLA conversion, long-context quality or inference speed. Protocol and launch cell: `results/EXP-071-sequential-recovery-protocol.md`.

EXP-071 first run (reviewed 2026-09-08): stopped after `0.137749` hours at depth 7 because the seed-456 `MIXED` layer-7 checkpoint received an `HfHubHTTPError`; the durability guard correctly stopped before dependent work. Forty-one new endpoints were uploaded successfully, so only the failed short layer must repeat. All reported numerics are finite. At the last common evaluation, depth 4, `ONPOLICY` beats matched `TEACHER` by `0.537193/0.102823` NLL for seeds 123/456 and beats the no-extra-update baseline by `0.592683/0.557858`. `MIXED` beats `TEACHER` by `0.427050` for seed 123 but loses by `0.005310` for seed 456; the depth-12 primary gate remains unevaluable. The engineering-only correction removes per-arm summary uploads while preserving per-layer endpoint uploads and extends transient-HF retry backoff. The experimental contract is unchanged. Details and checksums: `results/EXP-071-first-partial-summary.md`.

EXP-071 completed result (reviewed 2026-09-08): the resumed run completes both seeds through depth 12; the two sessions total approximately `0.270184` TPU hours. The registered `MIXED` gate passes: final NLL improves over matched `TEACHER` by `2.055885/1.929761`, with negative milestone-delta AUC in both seeds. Pure `ONPOLICY`, originally secondary, is stronger: final NLL `6.446534/6.569794` versus `8.749188/8.672009` for `TEACHER`, reducing excess NLL by `42.38%/39.25%`. From depth 8 to 12, `TEACHER` collapses by `+1.776001/+2.062225` NLL while `ONPOLICY` stays essentially flat (`-0.019162/+0.077146`). This supports sequential hybrid-conditioned recovery as an actionable method, not merely an input-shift diagnosis. Original Qwen remains much better (`NLL 3.315843`), only 12 positions were tested, and `ONPOLICY` requires locked confirmation because it was not the original primary arm. Full analysis/checksums: `results/EXP-071-completed-summary.md`.

### EXP-072 — Full-depth sequential recovery confirmation (pre-registered)

- **Implementation commit title:** `feat: add full-depth sequential recovery confirmation`.
- **Question:** confirm the post-EXP-071 selected `ONPOLICY` method on fresh text, four times more recovery, twice the training context, 32 evaluation windows, and all 24 Mamba positions of Qwen3-1.7B.
- **Matched design:** seeds `123/456`; all arms restart from the exact original EXP-069 endpoints rather than EXP-071 endpoints. `TEACHER`, `ONPOLICY`, and `MIXED` receive 4,096 current-Mamba-only updates per layer at context 128. Earlier recovered blocks and every Qwen parameter remain frozen.
- **Evaluation/gate:** fresh test offset; NLL/KL/agreement after 1/4/8/12/16/20/24 replacements. At both seeds, `ONPOLICY` must beat equal-update `TEACHER` in final NLL and milestone NLL-delta AUC. `MIXED` is secondary. Reusing prepared model seeds makes this a fresh-data/longer-budget confirmation, not an independent-initialization replication.
- **Scale:** 589,824 optimizer updates plus full-prefix cache generation and 32-window evaluations. Expected runtime is 5–8 hours, with an eight-hour soft deadline and resumable partial completion; this estimate is operational, not scientific evidence.
- **Durability:** each layer endpoint uploads immediately; summaries only at registered milestones/terminal states; transient HF failures use extended backoff and persistent failure halts before dependent work. Protocol and launch cell: `results/EXP-072-full-depth-sequential-protocol.md`.

EXP-072 v1 startup incident (reviewed 2026-09-08): the campaign stopped after approximately `0.020` hours before any training, baseline, checkpoint or model metric. The requested WikiText training prefix ended at token `2,686,976`, but the pinned tokenizer/corpus supplies `2,540,999`. This is an operational configuration error, not a failed scientific gate; the displayed `False` gate is merely the empty-result default. Before observing outcomes, v2 moves the fresh slice to `[2,424,832, 2,490,368)`, adds a deterministic capacity guard, and uses isolated local/state/HF namespaces to prevent the failed v1 record from entering resume state. All model, arm, seed, budget and decision settings remain frozen. Supplied artifact SHA-256: full `510cc2365c72af4f98590828e1f3818a9335bfa61dc72ff47eea45bb6a1f7b46`, summary `2bec7c0cdb96adf24c14290b0967694b5a0b47a5934efd0a87ccb3e2cce6d710`.

EXP-072 v2 first partial result (reviewed 2026-09-08): stopped safely after `1.033192` hours when HF rejected the seed-456 `ONPOLICY` layer-22 upload after all retries. Seed 123 has matched depth-20 results; seed 456 has matched depth-16 results plus a depth-20 `MIXED` result. At the last common milestone, `ONPOLICY` NLL is `6.378428/6.504203` versus `16.957157/16.240802` for equal-update `TEACHER`, a total-NLL reduction of `62.39%/59.95%` and excess-NLL reduction of `76.16%/73.91%`. This independently reproduces the sequential-input mechanism on fresh data and a four-times-longer budget, but depth 24 and the registered gate remain unevaluable. Both sequential arms worsen modestly at seed-123 depth 20, making completion essential. Resume the identical contract; no hyperparameter change is authorized. The engineering-only uploader retry window is extended beyond seven minutes. Details/checksums: `results/EXP-072-first-partial-summary.md`.

EXP-072 v2 completed result (reviewed 2026-09-08): the registered full-depth gate passes. At 24 replacements, `ONPOLICY` NLL is `10.545254/10.266002` versus `22.346362/22.037049` for equal-update `TEACHER`, with final deltas `-11.801108/-11.771046` and AUC deltas `-6.854519/-5.862882`. This cuts total NLL by `52.81%/53.42%` and excess NLL above Qwen by `61.21%/62.05%`. The advantage grows strongly with depth, confirming accumulated deployment-input mismatch as the mechanism. However, `ONPOLICY` worsens from about `6.9–7.0` at depth 20 to `10.3–10.5` at depth 24 and remains far from Qwen (`~3.0675`), so local sequential recovery is necessary but insufficient. Details/checksums: `results/EXP-072-completed-summary.md`.

### EXP-073 — Sequential warm-start joint recovery (pre-registered)

- **Implementation commit title:** `feat: chain sequential and joint recovery campaigns`.
- **Question:** after EXP-072 completes all 24 replacements, does its `ONPOLICY` advantage survive ordinary joint training of the entire student, or do the two layerwise starting points converge immediately?
- **Matched design:** seeds `123/456`; completed `TEACHER` versus `ONPOLICY` EXP-072 endpoints; both arms have identical layerwise update counts and receive 8,192 identical all-parameter Lion updates at context 256. Only the earlier recovery input distribution differs.
- **Evaluation/gate:** fresh 32-window test slice at steps `0/1024/2048/4096/6144/8192`. At both seeds, `ONPOLICY` must beat `TEACHER` in final held-out NLL and NLL-delta AUC. Endpoint hashes and token hashes are resume invariants.
- **Runtime:** expected joint phase 4–6 TPU hours. The chained entry point finishes EXP-072 and immediately starts EXP-073, preventing an otherwise mostly idle allocation. Full states and optimizer cursors are resumable through the private HF dataset.
- **Boundary:** this establishes persistence of the sequential warm start under full-model recovery, not compute-normalized superiority to random initialization. A pass triggers that final matched random comparison; MLA/14B/long-context remain separate. Protocol: `results/EXP-073-sequential-joint-recovery-protocol.md`.

EXP-073 first run (reviewed 2026-09-08): stopped after `0.064536` hours before optimizer step 1. Only seed-123 `TEACHER` step-0 NLL `21.987584` was measured. The composed student reused unchanged Qwen device buffers also supplied as frozen teacher input; donating the student tree therefore triggered JAX's same-call alias guard. This is an engineering failure and leaves the gate unconsumed. A sharding-preserving `may_alias=False` copy now gives the student independent storage before optimizer creation; values and the registered contract are unchanged, and the uploaded step-0 state resumes. Details/checksums: `results/EXP-073-first-failure-summary.md`.

EXP-073 completed result (reviewed 2026-09-09): the pre-registered relative warm-start gate passes at both seeds after 8,192 all-parameter updates. Final `ONPOLICY-minus-TEACHER` NLL is `-22.725672/-14.060614`, with NLL-delta AUC `-16.711477/-10.608272`; mean final difference is `-18.393143`. ONPOLICY therefore preserves its advantage through joint training. Absolute recovery is unstable: seed 123 ONPOLICY improves from `10.916678` to a best/final `8.804904/9.278139`, but seed 456 reaches `8.949895` at step 2,048 before collapsing and ending at `23.089646` versus `9.692907` initially. Both TEACHER arms finish much worse than step zero. Thus this is a positive relative-method result but a negative verdict on the current constant-`3e-5`, all-parameter Lion schedule. Stabilize the shared joint optimizer before a compute-accounted random comparison. Details/checksums: `results/EXP-073-completed-summary.md`.

### EXP-074 — Stabilized joint recovery (pre-registered)

- **Implementation commit title:** `feat: add stabilized joint recovery campaign`.
- **Question:** can a pre-declared lower-step Lion recipe turn the recoverable but unstable EXP-073 ONPOLICY start into consistent full-model improvement on both seeds?
- **Arms:** `LR3E-6` is primary; `LR1E-6` is an exploratory conservative control. Both restart from the same completed EXP-072 ONPOLICY endpoints and receive 8,192 all-parameter steps on the exact EXP-073 tokens/objective. Common warmup is 512, clipping 0.3, cosine end LR 0.1× peak, and weight decay zero.
- **Gate:** at both seeds, primary final NLL must beat its own step-zero NLL and no registered checkpoint may exceed `1.25×` step-zero NLL. Fixed checkpoints are `0/1024/2048/4096/6144/8192`; best-checkpoint selection cannot pass the gate. If only the exploratory arm passes, it requires locked confirmation.
- **Efficiency:** the exact EXP-073 trajectories serve as historical controls and are not rerun. Four new branches should take roughly four TPU hours and are fully resumable. This selects a stable joint optimizer before the later compute-accounted random comparison. Protocol: `results/EXP-074-stable-joint-recovery-protocol.md`.

EXP-074 completed result (reviewed 2026-09-09): the registered `LR3E-6` stability gate fails at both seeds, and exploratory `LR1E-6` also fails. `LR3E-6` changes NLL from `10.916678→15.672053` and `9.692907→15.944340`, with maximum/start ratios `1.5974×/1.6783×`. `LR1E-6` is less destructive but still changes NLL to `12.653587/12.504670`, raises it by `15.91%/29.01%`, and exceeds the fixed `1.25×` excursion bound. No trained checkpoint beats step zero in any arm/seed. Global LR reduction alone therefore does not stabilize all-parameter Lion; protect copied Qwen weights before comparing against random. Details/checksums: `results/EXP-074-completed-summary.md`.

### EXP-075 — Protected joint recovery (pre-registered)

- **Implementation commit title:** `feat: add protected joint recovery campaign`.
- **Question:** is full-model degradation caused primarily by updating copied Qwen parameters rather than by joint Mamba optimization itself?
- **Arms:** registered `MAMBA-ONLY` freezes every non-Mamba leaf; exploratory `MAMBA-NORMS` additionally trains normalization scales. Both use the same ONPOLICY starts, two seeds, EXP-074 data/objective, 8,192 updates, peak LR `3e-6`, warmup 512 and clipping 0.3.
- **Gate:** both primary seeds must finish below their own step-zero NLL with no registered checkpoint above `1.25×` start. The unchanged EXP-074 `LR3E-6` curve is the exact all-parameter causal control and is not rerun.
- **Boundary:** a pass selects which parameter families may move during joint recovery. It does not yet prove compute-normalized superiority over random, 14B transfer or MLA compatibility. Protocol: `results/EXP-075-protected-joint-recovery-protocol.md`.

EXP-075 completed result (reviewed 2026-09-09): the registered gate fails. `MAMBA-ONLY` changes NLL from `10.916678→11.694606` and `9.692907→11.304210`; `MAMBA-NORMS` changes it to `11.472961/11.904981`. No trained checkpoint beats step zero. Freezing Qwen nevertheless reduces final damage dramatically versus EXP-074's matched all-parameter arm (`+0.78/+1.61` instead of `+4.76/+6.25` NLL for MAMBA-ONLY), so copied-backbone movement is a major instability amplifier but not the whole cause. The protected curves spike early and recover as LR decays; norms give no consistent benefit. Keep Qwen frozen and isolate the loss geometry next. Details/checksums: `results/EXP-075-completed-summary.md`.

### EXP-076 — Protected objective bridge (pre-registered)

- **Implementation commit title:** `feat: add protected objective bridge campaign`.
- **Question:** can local decoder-block alignment provide a stable full-model training signal where EXP-075's single output KL/CE objective fails?
- **Matched arms:** `DELTA-BRIDGE` (primary) matches `h_l-h_(l-1)` at all 24 replaced positions; `STATE-BRIDGE` matches accumulated `h_l`. Both restart from exact EXP-072 ONPOLICY endpoints, train only Mamba leaves, and use identical seeds, tokens, 8,192 updates and EXP-075 Lion schedule. Output KL and causal CE have zero training weight.
- **Historical causal control:** EXP-075 MAMBA-ONLY has the same starts, mask, data, optimizer and update count, differing in objective only. It is not rerun.
- **Gate:** at both seeds, DELTA-BRIDGE final NLL must beat its own step-zero NLL with no checkpoint above `1.25×` start. A STATE-BRIDGE-only pass requires locked confirmation.
- **Purpose:** this tests whether the successful sequential contribution target from EXP-072 survives when all replacements train jointly. A failure points next to optimizer dynamics rather than another initialization sweep. Protocol: `results/EXP-076-objective-bridge-protocol.md`.

EXP-076 completed result (reviewed 2026-09-10): the registered gate fails; neither arm ever beats its initial NLL. DELTA-BRIDGE ends at `14.093563/14.952869` versus STATE-BRIDGE `16.507872/18.012545`, from starts `10.916678/9.692907`. Delta matching therefore reduces final degradation relative to state matching by `43.18%/36.78%`, supporting contribution subtraction as the better hidden target, but it is worse than EXP-075's output-KL/CE Mamba-only trajectories. Sampled hidden losses vary from order one to above 200,000 and finite pre-clipping gradient norms reach `~2.16e19`. Reject hidden-only joint recovery; retain delta matching only as a possible staged auxiliary and isolate optimizer dynamics next. Details/checksums: `results/EXP-076-completed-summary.md`.

### EXP-077 — Protected AdamW recovery (pre-registered)

- **Implementation commit title:** `feat: add protected AdamW recovery campaign`.
- **Question:** is Lion's sign-style update the remaining source of protected full-depth instability?
- **Primary arm:** `ADAMW-3E-6` changes only optimizer family relative to EXP-075 MAMBA-ONLY while preserving source endpoints, frozen Qwen mask, tokens, KL+CE objective, update count, schedule shape, clipping and numerical peak LR.
- **Exploratory arm:** `ADAMW-1E-5` tests a larger AdamW-specific scale; it is not treated as a one-factor causal comparison.
- **Numerics:** BF16 gradients and BF16 first/second moments preserve the memory-aware project contract. Adam moments are explicitly sharded with their parameter leaves. Both arms use 512 warmup steps, cosine decay, zero weight decay and clip 0.3 for 8,192 updates at context 256.
- **Gate:** at both seeds, primary final NLL must beat its own start with no fixed checkpoint above `1.25×` start. A secondary-only pass requires locked confirmation. Both failing moves the method to bounded/trust-region recovery rather than another ordinary LR/objective sweep. Protocol: `results/EXP-077-adamw-recovery-protocol.md`.

EXP-077 completed result (reviewed 2026-09-10): the registered gate fails narrowly. At matched peak LR `3e-6`, AdamW changes NLL by only `+0.289210/+0.288247`, versus Lion's `+0.777928/+1.611303` in EXP-075; this reduces final degradation by `62.82%/82.11%` and maximum excursions from `1.5436×/1.4835×` to `1.2039×/1.0499×`. Optimizer family therefore matters, but step zero remains best at both seeds. AdamW `1e-5` is clearly worse (`+4.447118/+0.973362`). Move from global optimizer/LR sweeps to a per-tensor relative-step bound. Details/checksums: `results/EXP-077-completed-summary.md`.

### EXP-078 — Protected trust-ratio recovery (pre-registered)

- **Implementation commit title:** `feat: add trust-ratio recovery campaign`.
- **Question:** can an update bound relative to each Mamba tensor's own norm convert EXP-077's near-stability into actual recovery?
- **Method:** BF16 LAMB rescales each leaf's Adam direction by `||parameter||/||direction||` before the scalar schedule. This prevents large projections, small recurrent scalars and biases from receiving the same absolute normalized step. A `1e-6` norm floor handles zero/tiny leaves.
- **Arms:** registered `TRUST-1E-4` and conservative `TRUST-3E-5`. Both preserve the EXP-077 source, frozen Qwen mask, KL+CE target, data, two seeds, context 256, 8,192 steps, warmup, cosine schedule, clipping and no-decay contract.
- **Gate:** primary must improve final NLL from step zero at both seeds with no checkpoint above `1.25×` start. A secondary-only pass requires confirmation. Failure moves to depth/staged update boundaries, not another global LR sweep. Protocol: `results/EXP-078-trust-ratio-recovery-protocol.md`.

EXP-078 completed result (reviewed 2026-09-10): the registered `TRUST-1E-4` gate fails badly (`+2.185281/+5.178166` final NLL). Exploratory `TRUST-3E-5` is the first joint recipe to cross below its own start at both seeds at any registered checkpoint: seed 123 reaches `10.732380` versus `10.916678` at step 4096 before regressing, while seed 456 ends at `9.341055` versus `9.692907`. This is not a pass or valid per-seed early stopping. Paired-window evidence is weak (mean/SE `-0.184/0.624` and `-0.352/0.266`; only `18/32` and `11/32` windows improve), but it warrants retaining the conservative trust ratio while isolating depth coupling. Details/checksums: `results/EXP-078-completed-summary.md`.

### EXP-079 — Attention-bounded depth segment recovery (pre-registered)

- **Implementation commit title:** `feat: add depth-segment recovery campaign`.
- **Question:** which natural six-Mamba run can improve under global KL+CE when the other 18 replacements are frozen?
- **Segments:** layers `(0-5)`, `(7-12)`, `(14-19)`, and `(21-26)` are bounded by retained GQA at `6/13/20/27`. Each arm trains exactly one segment from identical EXP-072 ONPOLICY starts; all other Mamba and Qwen leaves remain frozen.
- **Registered primary:** deepest `SEGMENT-4`, because its updates do not perturb the input distribution of any downstream Mamba block. The other three segments form a locked depth atlas, with reversed execution order at seed 456.
- **Training:** selected exploratory EXP-078 trust ratio `3e-5`, BF16 LAMB, 6,144 steps, context 256, two seeds and fixed KL+CE/data/evaluation. This totals 49,152 full-model optimizer steps and targets 5.5–7 TPU hours.
- **Gate:** primary final NLL must improve from step zero at both seeds with no checkpoint above `1.25×`. Any secondary selection requires confirmation. The depth ranking determines a later staged coordinate-recovery order. Protocol: `results/EXP-079-depth-segment-recovery-protocol.md`.

EXP-079 completed result (reviewed 2026-09-11): the registered gate fails. Training only one six-Mamba segment keeps every trajectory within the `1.25×` bound, so depth restriction removes the violent full-stack excursions, but no segment improves final NLL at both seeds. Deepest SEGMENT-4 is best for seed 123 (`10.916678→10.374044`, `-0.542634`) yet worsens seed 456 (`9.692907→10.176323`, `+0.483416`). Segment 3 briefly improves seed 123 at step 2048; every seed-456 segment otherwise remains above start. The depth hypothesis is seed-dependent and cannot select a segment. Combined with EXP-072, this favors local conditional targets over protected global KL even after optimizer and depth controls. Details/checksums: `results/EXP-079-completed-summary.md`.

### EXP-080 — Second on-policy coordinate sweep (pre-registered)

- **Implementation commit title:** `feat: add second-sweep recovery campaign`.
- **Question:** does revisiting all 24 already recovered Mamba layers on the current full-hybrid input distribution reduce residual composition error, and does sweep direction matter?
- **Arms:** registered `FORWARD-SWEEP` updates layers in causal order; matched `REVERSE-SWEEP` updates the same layers in reverse. Both begin from identical complete EXP-072 ONPOLICY endpoints. Each current layer receives the same local Qwen-conditioned contribution target and 2,048 BF16 Lion updates; every other parameter is frozen.
- **Evaluation:** full 24-Mamba NLL before the sweep and after 6/12/18/24 coordinate updates on the fixed 32-window test slice. Training uses a distinct 512-window length-128 slice. Total is 196,608 local updates with per-layer HF resume boundaries.
- **Gate:** at both seeds, forward final NLL must beat its own start, remain within `1.25×`, and beat reverse final NLL. This tests iterative coordinate recovery as a method; it is not another global optimizer sweep. Protocol: `results/EXP-080-second-sweep-protocol.md`.

EXP-080 partial result (reviewed 2026-09-11): the execution stopped after `0.684051` hours because the seed-456 forward position-16 endpoint could not be committed to Hugging Face after the full retry window (`HfHubHTTPError`). The locally saved endpoint and all numerical trajectories were finite; this is an operational rather than numerical failure. The partial result is nevertheless scientifically conclusive for the registered primary gate. Seed-123 forward completes at `12.245170` from `10.916678` (`+1.328492`, `+12.17%`), while reverse completes at `12.317666` (`+1.400988`, `+12.83%`). Seed-456 reverse deteriorates from `9.692907` to `14.008691` (`+44.53%`) and violates the `1.25×` bound; its partial forward curve is also above start after 6 and 12 updates. Because the gate requires forward improvement at both seeds, completing the remaining seed-456 coordinates cannot change its failure. Do not spend TPU quota solely to finish EXP-080. The result shows that a second locally optimized coordinate pass still damages the assembled model; update order is not the missing mechanism. Full/summary SHA-256: `9ea93988d960e566731ba104fbb0a58bd0f3546af04b478ad0fc67a704fb2e92` / `3e6acd940d352b8f6d2fb3a1998858c140eef280ef64b411006f509889acb8bc`. Details: `results/EXP-080-partial-conclusive-summary.md`.

### EXP-081 — Assembled-model trust-region coordinate recovery (pre-registered)

- **Implementation commit title:** `feat: add assembled-model trust-region sweep`.
- **Question:** can a held-out full-model acceptance rule prevent the composition failure measured in EXP-072/080, where locally improved Mamba blocks worsen the assembled hybrid?
- **Matched arms:** `HARD-ACCEPT` chooses between the unchanged block and the complete local proposal. Registered `TRUST-LINE` additionally searches interpolation coefficients `0.125/0.25/0.5/0.75` before the complete proposal. Both use the same forward layer order, starting endpoints, local objective, tokens and 2,048-step proposal budget.
- **Acceptance boundary:** selection minimizes assembled-model teacher prediction KL on a separate 32-window train calibration slice. Alpha zero is mandatory, and a nonzero update needs at least `0.1%` relative KL improvement. Locked test NLL is evaluated only at positions `0/6/12/18/24` and cannot select a block, alpha or checkpoint.
- **Gate:** at both seeds, TRUST-LINE must improve final locked NLL, remain within `1.25×` start, accept a nonzero update and beat HARD-ACCEPT. Stable rejection without NLL recovery is explicitly a negative result, not a pass.
- **Compute:** two seeds, two arms and 24 coordinates produce 196,608 local BF16 Lion updates plus assembled-model calibration evaluations. Fresh proposal/calibration/test offsets are recorded in the immutable protocol.
- **Durability correction:** complete branch state is uploaded only after every six coordinates, at most 16 checkpoint commits rather than EXP-080's 96 per-layer commits. A failed HF upload is recorded and active TPU work continues; a restart may repeat at most five coordinates. Protocol: `results/EXP-081-trust-region-sweep-protocol.md`.

EXP-081 completed result (reviewed 2026-09-11): the registered gate passes in `0.694239` TPU hours with finite trajectories and no durability warnings. TRUST-LINE changes locked-test NLL by `-0.213726` (`-2.113%`) at seed 123 and `-0.028699` (`-0.233%`) at seed 456, while HARD-ACCEPT worsens it by `+0.598795/+0.460585`. Fractional line search beats the matched binary control by `0.812521/0.489284` final NLL and keeps maximum milestone ratios at `1.02296/1.0`. It accepts only `6/24` and `5/24` coordinates; only layer 26 is shared across the two selected sets. This is the first tested assembled 24-Mamba recovery rule to improve full-model NLL at both seeds, supporting the mechanism that candidate updates must be judged after composition. The small seed-456 effect, seed-dependent layer identity, and absence of per-window NLL in the original artifact prevent a strong final claim. Freeze the recipe and run fresh-data replication before changing alphas, threshold, or moving to 14B/MLA. Full/summary SHA-256: `d0d4bc607813f1ebbd962088d1e6030b649d6b9b6b2cb943660e7cfd7d246d59` / `62d9a45e47d6d8b3494ec575febe7aa04f128ea2b5431979382179f1d412ebd`. Details: `results/EXP-081-completed-summary.md`.

### EXP-082 — Multi-split trust-region replication (pre-registered)

- **Implementation commit title:** `feat: add multi-split trust-region confirmation`.
- **Question:** does the frozen EXP-081 recipe reproduce when proposal, calibration, and locked evaluation text all change?
- **Design:** eight complete data replications, each containing source seeds 123/456 and matched HARD-ACCEPT/TRUST-LINE arms. Alpha grid, 0.1% acceptance threshold, forward order, optimizer and 2,048 local steps are unchanged. This is 1,572,864 local proposal updates plus assembled-model alpha evaluation.
- **Data independence:** all eight proposal and calibration slices are mutually disjoint fresh WikiText-2 train ranges. Locked evaluation is cross-corpus on eight disjoint ranges from the pinned `deepmind/pg19@4d28bd7` validation manifest and deterministic Google-hosted PG-19 assets. Runtime token hashes freeze exact content. Every final evaluation retains its 32 per-window NLL values.
- **Statistical gate:** each source seed is evaluated separately across eight data replications. Mean TRUST-LINE final-minus-start NLL and its upper normal 95% bound must both be negative; at least six of eight replications must improve; mean final NLL must beat HARD-ACCEPT; every trajectory must stay within `1.25×`; and every replication must accept a nonzero update. Both seeds must pass.
- **Inference boundary:** individual windows are correlated within a trained replication, so the primary standard error uses eight replication means. Window-level uncertainty is descriptive only. These are fresh-data replications sharing two initialization seeds, not 16 independent model initializations.
- **Runtime/resume:** expected v5e-8 runtime is 5–7 hours from EXP-081's measured throughput. Every inner replication retains four milestone checkpoints and its own HF namespace; the outer result uploads after each completed replication and resumes without repeating completed work. Protocol: `results/EXP-082-trust-region-replication-protocol.md`.

EXP-082 completed result (reviewed 2026-09-12): all eight data replications complete in `5.239880` TPU hours without durability warnings; the registered aggregate gate fails. Seed 123 is strongly positive against its own start: TRUST-LINE improves PG-19 NLL in `8/8` replications, mean delta `-0.723808`, replication SE `0.113996`, normal 95% upper bound `-0.500376`, and no milestone exceeds start. It nevertheless loses to HARD-ACCEPT by `+0.340009` mean final NLL, failing the matched-control clause. Seed 456 does not replicate: `5/8` improve, mean delta `+0.129601`, SE `0.416917`, upper bound `+0.946758`, with individual deltas `-1.373439…+1.920572`; it is tied with HARD-ACCEPT at `+0.024053`. WikiText calibration KL decreases in every run, but its magnitude does not predict cross-corpus PG-19 direction, identifying single-domain selection as the concrete failure. Layers 0 and 26 are accepted in `13/16` and `15/16` seed-replications respectively, while other coordinates remain sparse. Retain trust interpolation as a mechanism but reject the current single-domain recipe as a confirmed method. Full/summary SHA-256: `7df5f7aa8da802c5103e8d845caad62beabe9b7955ad8524a6c571eb51c43edb` / `8480e83ca6d39e8eb3267167686ba1f605aa41b042254ecc1cf2beb4c77e0fe3`. Details: `results/EXP-082-completed-summary.md`.

### EXP-083 — Dual-domain assembled trust recovery (pre-registered)

- **Implementation commit title:** `feat: add dual-domain trust recovery campaign`.
- **Intervention:** compare matched full alpha-grid arms that compute both domains. WIKI-ONLY selects on WikiText-103 assembled KL; registered DUAL-CONSENSUS accepts an alpha only when both WikiText-103 and PG-19 validation KL improve by at least 0.1%, then minimizes the worst normalized domain score.
- **Freshness:** proposal/primary-calibration text moves to disjoint pinned WikiText-103 train slices. PG-19 validation becomes secondary calibration after EXP-082 exposed its outcomes. Final evaluation moves to previously untouched PG-19 test books and cannot select candidates.
- **Compute:** eight data replications × two source seeds × two arms × 24 coordinates × 2,048 local updates = 1,572,864 local proposal steps, plus matched two-domain alpha forwards. Expected runtime is 6–8 v5e-8 hours.
- **Gate:** for each source seed, DUAL-CONSENSUS needs a negative replicate-mean final NLL delta with negative 95% upper bound, at least `6/8` improving replications, lower mean final NLL than WIKI-ONLY, no `>1.25×` excursion, and nonzero acceptance in at least six replications. Both seeds must pass.
- **Boundary:** this changes the acceptance signal, not the Mamba initializer or local proposal. A failure distinguishes proposal inadequacy from single-domain overfitting and prevents a post-hoc layer-0/26 subset from being presented as confirmation. Protocol: `results/EXP-083-dual-domain-trust-protocol.md`.

### Full 15/85 hybrid transplant preflight — production integration milestone

- **Implementation commit title:** `feat: add full hybrid transplant preflight`.
- **Purpose:** convert the separately selected Mamba and MLA research results into one auditable 40-layer production initialization contract before any full checkpoint import or recovery run.
- **Layer contract:** exactly six retained attention layers `(5,12,19,25,32,39)` and 34 Mamba-3 MIMO layers, i.e. exactly 15% attention / 85% Mamba. Every layer owns exactly the six Qwen mixer tensors Q/K/V/O plus Q/K norms.
- **Mamba policy:** all 34 replaced layers select `INIT-K-balanced-qkvo-lift`, the operator-preserving transplant confirmed by EXP-056/057/059/060. This is a selected policy, not yet a completed 34-layer checkpoint import.
- **Attention policy:** all six retained attention layers now instantiate the EXP-019 `Qwen3RoRoPEBKVAttention` contract rather than the stale generic MaxText-style MLA scaffold. Frozen conversion settings are activation-PCA latent rank 448, one complete 128-element RoRoPE key, eight source KV heads, and no latent RMSNorm.
- **Source ownership audit:** 203 embeddings/norm/MLP/head tensors plus 240 mixer tensors cover exactly `443/443` tensors in pinned `Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18`, with no overlap between direct and mixer ownership.
- **Cache accounting:** each retained Qwen GQA layer stores 2,048 BF16 elements/token; EXP-019 stores 576. Across six attention layers this is `12,288 -> 3,456` elements/token, a 71.875% reduction for the retained-attention KV cache. This excludes recurrent Mamba state and does not yet constitute a measured generation-throughput result.
- **Shape-only production audit:** the integrated architecture traces `14,765,658,870` parameters and 631 tensors. Under the v5e-8 `data/fsdp/tensor=1/4/2` layout, ideal BF16 weights are 3.451 GiB/device and weights + BF16 gradients + BF16 Lion momentum are 10.354 GiB/device, leaving 5.646 GiB of a nominal 16 GiB device for activations, XLA temporaries, executable buffers, and uneven replication.
- **Engineering gate:** `scripts.full_hybrid_transplant_preflight.py --require-ready` reports the layer schedule, 443/443 source ownership, selected method IDs, cache accounting, and architecture compatibility without allocating model/checkpoint arrays. Its first run correctly rejected the stale rank-512/64-wide generic MLA contract; after production integration it reports `verdict=GO`.
- **Numerical boundary:** this milestone contains CPU shape/smoke tests and analytical HBM accounting. It does not calibrate the six RoRoPE+BKV bases, materialize the 34 exact-lift mixers, execute a full 14B forward/backward, measure incremental decode, or establish recovered quality. Those remain required before full training.

### Cloud TPU campaign protocol

- **Campaign contents:** run EXP-045, EXP-046, and EXP-047 sequentially in one TPU v5e-8 allocation. Completed experiment and cell JSONs are restart boundaries; a rerun skips every numerically valid result.
- **Failure containment:** every cache and endpoint is written atomically with hash/provenance checks. Qwen shards are pruned after last use; completed depth-cache arrays are removed before context evaluation; each completed context cache is removed after its result; and large endpoint payloads are removed only after every dependent context cell has completed. Final scientific JSONs and checkpoint metadata remain.
- **Resource policy:** transient Qwen weights use `/dev/shm` only if the mounted RAM filesystem exposes at least 36 GiB free; otherwise bounded disk streaming is used. The campaign never assumes that total host RAM is automatically mounted as a filesystem.
- **Notification policy:** when enabled, Telegram receives exactly a campaign-start notification and a final notification. Normal completion reports numerical and scientific status. A caught exception writes `extent-tpu-campaign-failure.json`, marks the exact failed stage, and sends the error type/stage. External hard termination of the VM cannot execute a final callback and is outside this guarantee.
- **Artifact policy:** the primary handoff is `extent-tpu-campaign.json`; individual EXP/cell JSONs and stage manifests remain for audit. Regenerable activation arrays and completed endpoint payloads are intentionally excluded from final cloud output after all dependent measurements finish.
- **Completed execution summary (MEASURED):** the campaign ran from `2026-08-24T09:25:54Z` to `13:11:22Z` (`3.758` hours), used eight TPU devices and RAM-backed Qwen shard storage, completed every numerical stage, pruned 32 cache arrays and four endpoint payloads, and produced `passed=true`, `scientific_gate_passed=false`. The user-supplied campaign artifact has SHA-256 `381353260e2a171aea8265bfae270ae790a97b8e731bd8e6ee0587169cd69fb6`.

## 8. Reasoning SFT boundary

“Claude-like reasoning” is not part of the architecture-recovery claim. It should be a later experiment with explicit data provenance, permissions, filtering, and a frozen pre-SFT checkpoint. Otherwise architecture recovery and behavior imitation become confounded. Prefer reproducible/open reasoning datasets or lawfully generated teacher traces, and evaluate reasoning improvements separately from retained base capabilities.

## 9. Related work anchors

- Mamba-3: Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, arXiv:2603.15569. https://arxiv.org/abs/2603.15569
- Wang et al., *The Mamba in the Llama: Distilling and Accelerating Hybrid Models*, arXiv:2408.15237. https://arxiv.org/abs/2408.15237
- Bick et al., *Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models* (MOHAWK), arXiv:2408.10189. https://arxiv.org/abs/2408.10189
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

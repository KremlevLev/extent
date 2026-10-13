# Mamba-3 in the Qwen: Composition Shift, Numerical Stability, and Recovery Limits in Cross-Architecture Transfer

**Working draft, updated 13 October 2026. Not submitted.** Results are limited to the recorded configurations. EXP-111 and EXP-113 are completed; EXP-112 is partial. This draft does not claim a recovered Qwen-equivalent model, a superior full-model initializer, or faster inference.

## Abstract

Transferring pretrained Transformers to state-space language models requires more than a locally accurate mapping of sequence mixers. We study Qwen-to-Mamba-3 MIMO transfer through partial replacement, full-depth composition, sequential recovery, and numerically robust adaptation. An operator-preserving SISO-to-MIMO lift improves held-out language-model negative log-likelihood relative to random initialization in partial compositions, yet this advantage does not extend to the tested full-model recovery protocols. Matched-input diagnostics reveal severe residual-stream distribution shift when independently recovered blocks are composed. Training replacement blocks on inputs produced by the progressively modified model substantially improves full-depth quality: at 24 replacements, sequential on-policy recovery obtains NLL 10.55 and 10.27, compared with 22.35 and 22.04 for equal-update recovery on clean teacher inputs. We also identify FP32 sum-of-squares norm overflow despite finite gradient elements, using paired numerical proposals on identical gradients and optimizer states. Stable clipping and protected rank-32 corrections subsequently reduce full-model test NLL from 12.97/12.28 to 7.27/7.31 after 2.10 million additional input tokens per trajectory. A substantial teacher gap and a recovery plateau remain. These findings distinguish local transfer fidelity, composition robustness, numerical validity, and learning efficiency, and motivate controlled end-to-end cost comparisons rather than inference from early recovery curves.

## 1. Introduction

Pretrained Transformer checkpoints contain an investment that could be reused in architectures with different sequence-mixing mechanisms. Mamba-3 MIMO offers a target for such reuse, but a useful transfer procedure must preserve model behavior after many blocks have been replaced and must recover quality at a defensible compute cost.

Our study investigates three questions. Does a mathematically faithful embedding of a donor recurrence provide a useful whole-model initialization? How does changing the input distribution of downstream blocks affect composition? Can robust, restricted adaptation recover useful quality without destabilizing recurrent dynamics?

The present evidence supports a study of transfer mechanisms and failure modes. It does not yet establish compute-efficient recovery relative to a tuned ordinary-training baseline. Experiments EXP-111–113 were prepared specifically to measure that comparison. At this revision, EXP-111 and EXP-113 are completed; EXP-112, the same-start full-decoder comparator, remains partial.

## 2. Models, evaluation, and scope

The current full-depth testbed uses Qwen/Qwen3-1.7B-Base at revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Its 28 decoder blocks become a hybrid with 24 Mamba-3 MIMO mixers and four retained GQA mixers at indices 6, 13, 20, and 27. MLPs, normalization, vocabulary, and retained attention originate from Qwen; trainable subsets depend on the experiment. The Mamba configuration uses state dimension 64 and MIMO rank 4. **MIMO rank 4 and correction rank 32 are different quantities.**

Partial-composition experiments also use the Qwen3-14B source family. A 34-Mamba/6-attention full-size hybrid has been materialized and exercised for engineering feasibility, but its language-model recovery has not been demonstrated. The 1.7B experiments retain GQA to isolate Mamba transfer; they do not establish joint Mamba/MLA recovery.

We report token-level negative log-likelihood (NLL), with lower values indicating better prediction. Perplexity is exp(NLL); it is not a literal count of candidate answers. Excess NLL subtracts the original teacher's NLL on the same evaluation slice. Absolute NLL values from different slices or evaluation revisions are not directly comparable.

WikiText-103 is the primary training domain in the reported recovery experiments; later experiments include PG-19 evaluation. Source seeds 123 and 456 recur across campaigns. Shared starts, controls, and evaluation sets are not independent replications. Some evaluations were repeatedly inspected and must be presented as exploratory. Equal input tokens do not imply equal teacher passes, parameter counts, or FLOPs.

## 3. Transfer and recovery procedures

### 3.1 Operator-preserving MIMO lift

The balanced lift embeds a pooled QKVO-derived SISO donor into the canonical MIMO parameterization. It copies B/C channel behavior, distributes input/output channel coefficients, and compensates the skip path. In the implementation, MIMO input/output coefficients are 1/R and the skip coefficient D is multiplied by R. This preserves the donor recurrence under the implemented contract; it does **not** make softmax attention equivalent to Mamba-3. The algebra and its assumptions require a dedicated derivation before submission.

### 3.2 Sequential recovery on hybrid-prefix inputs

Independent recovery fits a replacement block on clean Qwen inputs. Sequential on-policy recovery instead fits each newly inserted block using inputs produced by the already modified prefix. The teacher target is evaluated under the corresponding input condition. The comparison aligns layer order, training data, and per-layer update budgets. This intervention addresses the distribution on which the replacement must operate after composition.

### 3.3 Stable clipping and protected corrections

The correction procedure freezes the base model and trains low-rank changes to Mamba input/output kernels with true-token CE. Protected input corrections leave the projection columns controlling timestep, decay, trapezoid, and angle unchanged, while allowing value, gate, B, and C corrections. Protection concerns these projection columns, not every possible recurrent parameter.

To compute clipping without squaring very large gradients, let m=max|g| and s=sqrt(sum((g/m)^2)). The clipped direction can be formed from (g/m)/s, with explicit zero handling and checks for nonfinite elements. Logarithmic norm diagnostics avoid requiring the product m*s to remain representable. We do not mask nonfinite gradients or replace them with zero. This is an established numerical principle implemented for this recovery path; its discovery here is not a claim to have invented stable norm computation or low-rank adaptation.

## 4. Results

### 4.1 Local transfer gains do not guarantee full-model gains

EXP-061 evaluates production-aligned partial compositions over three paired seeds. At 4, 8, and 12 replacements, mean exact-minus-random NLL is -0.2060, -0.1228, and -0.1649, respectively, with wins in all three seeds at each depth. At 12 replacements this corresponds to approximately 3.12% lower total NLL and 14.51% lower excess NLL. Reported bootstrap intervals are below zero, but three seeds provide limited uncertainty evidence.

The tested full-model recipe reverses this ordering. In EXP-064, random initialization has 27.93% lower final excess NLL than exact lift. A longer EXP-089 comparison completes seed 123 at 25.17 million training input tokens: exact lift has excess NLL 15.5410, versus 9.7633 for random. Seed 456 is incomplete; the registered conjunctive gate is nevertheless impossible after seed 123 fails. These results reject the tested full-model advantage, not all possible transfer initializers. The partial and full-model comparisons differ in scale and protocol, so they do not isolate replacement depth as the sole cause of reversal.

### 4.2 Composition exposes a severe input-distribution shift

EXP-070 evaluates identical Mamba and Qwen blocks on matched teacher-generated and hybrid-prefix-generated inputs without training. At 24 replacements, the median hybrid-input/teacher-input decoder-error RMS ratio is 17.55 and 95% of recorded ratios exceed one. These pooled measurements are descriptive and correlated across layers and nested prefixes; they are not independent samples for significance testing. Teacher-input error also worsens in later layers, so input shift is not the only observed limitation.

EXP-072 tests the corresponding recovery intervention. At full depth, ONPOLICY reaches NLL 10.5453/10.2660, versus TEACHER 22.3464/22.0370 for seeds 123/456. Both registered seed gates pass. However, ONPOLICY worsens from 7.0043/6.8942 at depth 20 to 10.5453/10.2660 at depth 24. The original Qwen is approximately 3.0675 on that slice. Sequential recovery strongly improves the matched baseline but does not restore teacher quality or establish superiority to random at equal total cost.

![Sequential recovery by replacement depth](figures/sequential_depth.png)

*Figure 1. EXP-072-v2 saved held-out NLL at registered replacement depths, shown separately for each source seed. Lines connect recorded evaluations. This is a depth plot, not a training-time curve.*

### 4.3 Norm overflow and learning failure are distinct

In EXP-097, all four NAIVE branches stop at attempted steps 4341, 3207, 5331, and 4752. Their forward values, losses, gradient elements, current states, and same-state SAFE proposals remain finite, while the raw FP32 sum-of-squares norm is nonfinite. Finite log10 norms lie between 19.43 and 20.12. The NAIVE proposals themselves are also recorded finite: the immediate stop is the explicit norm guard, not evidence of NaN parameters.

All eight SAFE branches complete 8192 steps. The separate SAFE trajectories diverge from NAIVE before the stopping event; this does not demonstrate that switching the failed trajectory to SAFE would recover its quality. EXP-097 passes its numerical diagnostic gate, which is narrower than a gate for useful model recovery.

### 4.4 Protected low-rank recovery improves NLL but leaves a plateau

The prespecified protected rank-32 arm in EXP-097 reaches held-out NLL 7.2668/7.3098 from 12.9680/12.2770. Each trajectory uses 8192 windows of length 256: 2,097,152 additional input tokens, or 2,088,960 next-token targets. This count excludes the trained warm-start preparation and differs from the campaign-wide token budget summed over arms. The original Qwen NLL is 2.9803 on this slice. Protection and rank are confounded against the rank-8 comparator in EXP-097.

![Validation recovery trajectories](figures/recovery_validation.png)

*Figure 2. EXP-097 validation NLL versus additional input tokens, separately by seed. These validation values use a different slice from the 12.97/12.28 test starts quoted in the text. Failed trajectories end at their last recorded validation; crosses identify that recorded point, not the exact failure time. No missing final test value is imputed. Warm-start cost is not shown.*

EXP-098 isolates protection at rank 32. Its Wiki advantage is 6.3155/0.0556 NLL; the second seed fails the registered 0.1 improvement threshold. PG-19 favors protection at both seeds, but the combined cross-domain gate fails. EXP-099 does not establish a benefit of increasing protected correction rank to 64. Late teacher anchoring in EXP-101 produces a domain tradeoff, and dynamics release in EXP-103 does not improve the paired result. EXP-102 has opposing LR preferences between Wiki and PG-19.

Short full-decoder and Mamba-only recovery in EXP-104/105 do not pass their paired gates. Longer full-decoder recovery in EXP-106 completes both 32768-step trajectories, improving Wiki NLL by only 0.0145/0.0650 and PG-19 by 0.0124/0.0874 over its own starts. EXP-108–110 remain partial: comparisons at the shared 24576-window point do not establish consistent benefit from batch accumulation or layer-group clipping. Their final 65536-window gates remain unresolved. These outcomes constrain tested recipes; they do not identify the plateau's cause or prove an architectural quality ceiling.

### 4.5 Preliminary efficiency-control evidence and local interventions

EXP-111 completes 32768 updates per seed and reaches fresh locked Wiki NLL 7.3451/7.3098 and PG-19 NLL 9.9345/9.9786. On the reused historical panel, the first two consecutive measurements below 7.5 occur at steps 5120/6144 and 6144/7168 for seeds 123 and 456. This repeats substantial early recovery, but does not reach NLL 7.0 and does not eliminate later fluctuations.

EXP-113 completes both 32768-update trajectories with native random Mamba, full-decoder CE, and validation-selected LR 1e-4. At this endpoint, fresh locked Wiki NLL is 7.7062/9.0913 and PG-19 NLL is 11.1829/11.7925, versus EXP-111's 7.3451/7.3098 and 9.9345/9.9786. Both endpoints are recorded durable, with no pending uploads. Random recovery is highly nonmonotonic: seed 123's historical NLL changes from 7.5855 at step 21504 to 17.0469, 18.6624, and 29.6866 at the next three recorded evaluations, before returning to 7.5832 at step 26624. Neither seed confirms two consecutive historical measurements below 7.5 by the registered end.

EXP-112 uses the same initial effective model as EXP-111: the recorded start evaluations match exactly for both seeds. Its validation-selected LR is 3e-4, and the run stops at the registered session deadline with durable MAIN states at steps 12288/8642. At the common 8192-update point (2.0972 million MAIN input tokens per seed), fresh Wiki NLL is 15.9941/13.5068 for EXP-112, 11.9960/30.6360 for EXP-113, and 7.5094/7.3806 for EXP-111. The short LR pilots do not guarantee a stable long-horizon optimizer; the partial full-decoder comparison cannot establish its final achievable quality.

EXP-111 uses 16.7772 million training input tokens across both MAIN trajectories; EXP-113 uses 18.3501 million including all LR pilots and their paid selected-path replay. Recorded training time is 5.503 h and 7.130 h respectively, while summed runner time is 6.514 h and 10.055 h (EXP-113 spans two sessions). MAIN-path curves exclude pilot tokens, and EXP-111's warm-start preparation cost remains unknown. These observations support the protected recovery recipe on the measured budgets, not isolated initializer superiority, a universal speedup factor, or an equal-total-cost pipeline comparison. No recipe reaches historical NLL 7.0 within its observed horizon.

Read-only rollback probes identify stage-dependent harmful changes. In EXP-113 seed 456 at step 8192, reverting only layer 8 to its own initial subtree reduces NLL on the eight-window diagnostic panel from 30.3624 to 13.2122, whereas reverting neighboring layer 9 increases NLL by 1.2276. In EXP-111, reverting layer 26 at later probes strongly worsens both seeds, despite its high contribution mismatch to the original Qwen. All recorded identity rollback controls have zero NLL error. These selected validation-panel interventions demonstrate local effects in a particular composition; they do not establish a universal culprit, a deployable rollback treatment, or the cause of the plateau.

![Preliminary recovery comparison](figures/quality_curves.png)

*Figure 3. Selected MAIN trajectories for EXP-111 and EXP-113 (completed) and EXP-112 (partial), with separate historical and fresh locked Wiki evaluations. No smoothing or missing endpoint imputation. The x-axis excludes pilot and warm-start costs; the recipes differ in trainable subspace, initialization, LR, and warmup. Lines connect observations and do not imply intermediate measurements. No isolated initializer or total-cost claim follows.*

## 5. Discussion and remaining efficiency comparison

The evidence distinguishes four issues: fidelity of a local parameter mapping; compatibility under composition; validity of numerical training operations; and the cost of achieving useful quality. Solving one does not establish the others. Low-rank protected recovery is currently a working candidate, while sequential input-conditioned recovery and norm-overflow diagnosis are better-supported findings.

EXP-111 compares protected rank-32 recovery from the original ONPOLICY warm start; EXP-112 uses ordinary full-decoder CE from the same effective model; EXP-113 uses ordinary CE with native random Mamba subtrees and the original Qwen's other weights. Full-decoder controls receive prespecified validation-selected LR pilots. All search and replay costs must be charged. We will report additional recovery cost separately from total preparation-plus-recovery cost, include attempts that do not reach a quality threshold, and avoid extrapolating beyond measured endpoints. **The complete three-way and total-pipeline comparisons are not available yet.**

A stronger method paper needs replicated efficiency gains, fresh held-out evaluation, comparison with relevant transfer recipes, and further scale/context validation. Inference benefits require optimized implementations and direct hardware measurement; the current reference scan does not support a speed claim.

## 6. Limitations

The main full-depth findings concern one source family and two recurring source seeds. Evaluations span distinct slices and numerical implementations. Partial models, full models, and later recovery stages differ in training procedure and cannot be collapsed into one learning curve. Reused controls are dependent observations. Some campaigns are partial or have lost final model states despite saved metrics. All confirmed quality conclusions are bounded by their registered data and horizons. Long-context recovery, 14B quality, joint MLA conversion, downstream capabilities, complete pipeline cost, and optimized inference efficiency remain unestablished.

## Related work to integrate

- [Mamba-3: Improved Sequence Modeling using State Space Principles](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html).
- [The Mamba in the Llama: Distilling and Accelerating Hybrid Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/723933067ad315269b620bc0d2c05cba-Abstract-Conference.html).
- [Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models (MOHAWK)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3848fef259495bfd04d60cdc5c1b4db7-Abstract-Conference.html).
- [Attention to Mamba: A Recipe for Cross-Architecture Distillation](https://arxiv.org/abs/2604.14191).

Before submission, supply verified bibliographic metadata, a detailed method comparison, the exact-lift derivation, complete data/optimizer tables, and a provenance appendix. No first-of-its-kind claim is made in this draft.

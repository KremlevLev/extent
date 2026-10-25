# EXP-160 — data diversity before scaling end-to-end recovery

Prepared 25 October 2026. No accelerator experiment launched by preparation.

## Where Extent is relative to MOHAWK

Source: Bick et al., Transformers to SSMs, https://arxiv.org/html/2408.10189v2
Sections 4.1–4.3, 5.1, 5.5. The February 2025 revision also adds hybrid results;
the 3B pure Phi-Mamba recipe is 80M/160M/2.76B across stages, C4, context2048.

| MOHAWK | Purpose | Extent evidence/status |
|---|---|---|
| 1 Matrix orientation | Match causal token-to-token mixing operators on teacher-prefix inputs, after aligning surrounding components | NOT established as a reproduced MOHAWK stage. We have a canonical Mamba3 quadratic dual execution path, but execution parity is not an attention-matrix fitting result. |
| 2 Hidden alignment | Fit individual complete mixer blocks to teacher block outputs on shared inputs | Earlier local fits and EXP072 sequential/on-policy fitting are related, but not an equal-budget reproduction. EXP072 explicitly showed composition/input shift and degradation at final replacements. |
| 3 End-to-end KD | Reconcile composition and match teacher predictions in the assembled student | Closest to current EXP159, but only ~1.9M added positions per arm, frozen donor backbone, rank256 corrections, repeated small Wiki pool. No broad capability restoration. |

EXP159's four normalized residual-stream losses applied jointly in the full model
are NOT MOHAWK stage2. Its lack of a clear advantage cannot reject their
independent block-alignment procedure. The current checkpoint should not be
reset just to relabel it stage1. Keep the successful lineage, improve stage3 data,
and investigate a separate stage1/2 initialization bank from the original donor.

### What matrix orientation means for Mamba3 MIMO

It matches how earlier token values contribute to later token outputs, not raw
weight matrices, and not the Mamba state matrix A against Qwen Q/K weights.
`extent/layers/mamba3.py` already implements `mamba3_dual_reference` and the
trapezoidal causal mask. It retains rotations, previous-token injections, MIMO
value/gate/readout factors and skip. Matching only that scalar mask would omit
essential parts of the operator. Mamba mixing need not be positive/stochastic.

MOHAWK changed Mamba2 head structure, gate/convolution and discretization to align
with its teacher. Our Qwen GQA and canonical M3 MIMO are different; copying those
changes would define a different architecture. The next stage1 prototype needs:

1. A stated common value/output basis or explicit head map (Qwen16 query heads
   versus current Mamba24 heads); a raw per-head Frobenius loss is undefined
   without that map.
2. Fixed input-dependent selectors and a materialized linear value operator,
   including rotation/trapezoid/MIMO/skip, before nonlinear output normalization.
   Validate its application against the existing scan on tiny examples, including
   causality, diagonal terms and finite gradients. Full network Jacobians are
   not silently interchangeable with MOHAWK's conditional mixer matrices.
3. First 1/12/24-replacement probes, held-out operator and block-output error,
   then composed NLL and broad downstream diagnostics. Fit ALL canonical mixer
   coordinates in this separate bank rather than assuming rank32 suffices.
4. Compare this initialization against the existing initialization with matched
   downstream stage3 tokens and separately report preparation/teacher FLOPs.

This is a design, NOT an implemented matrix-alignment experiment or a win.

## Immediate experiment: identical KD, old Wiki versus fresh C4

Two T4 workers, source immutable EXP155 CURRENT118304, revision
5e0d5026d03da0320c72162f5216215b137e43c6, source pin EXP157. Deliberately starts
from the same source as159 for an interpretable comparison, not its later KD
checkpoint. Fresh Adam512 warmup, protectedrank256+pregate, mixer3e-4/norm1e-4,
stableclip1. Both L=.5CE+.5forwardKL(Qwen||student), T1, hidden_beta0.

- GPU0 control: old fixed 4,194,304-position Wiki FIT and original permutation.
  Its repetition is deliberate and labelled; it is not the new training corpus.
- GPU1 intervention: first unused C4 window, then next, never modulo/restart0 on
  resume. NEW=checkpoint_global-118304 determines the exact next window.

This changes source domain, diversity and repeat rate together. It tests the
practical broad-data recipe, not repetition as a uniquely isolated cause.
Single preparation seed; compare matched NEW counts as well as wall time.
No claim that old data caused all earlier plateaus or that C4 must win.

C4 dataset allenai/c4, en, revision1588ec454efa1a09f29cd18ddd04fe05fc8653a2.
Default100,000,000 train positions, 512*256 validation positions, pinned Qwen
tokenizer; EOS concatenation, no padding, context256/batch1 to keep math fixed.
This is the first corpus tranche, not the full multi-billion-token plan and not
MOHAWK context2048 reproduction. Later context growth requires separate tests.
English C4 is chosen to match the reference dataset family; RU/code/chat are not
claimed included. Future multilingual/domain expansion is a separate version.

Validation is packed first from official C4 validation. SHA256 of NFKC and
whitespace-normalized whole documents is shared across splits and excludes exact
duplicates. Wikipedia URLs are excluded to reduce overlap with old diagnostics.
This does NOT establish near-duplicate or benchmark decontamination. Released
web-corpus provenance/licensing follows the C4 card (ODC-BY); keep attribution.

## Corpus operations and storage

`scripts.prepare_broad_corpus` runs on CPU BEFORE allocating the training GPU.
Downloads just one pinned compressed shard at a time into its owned temporary
directory, deletes it after consumption; never downloads the entire C4 dataset.
Retries use existing deadline/429/Retry-After policy. A tokenizer is the only
model-related download. No model weights or full hidden/logit caches are built.

Packed int32 train data is400MB, validation~0.52MB. SQLite document hashes and
tokenizer add overhead. Default8GB local preparation ceiling and2GiB free reserve;
source shard is temporary. Limit checks stop rather than delete unrelated files.
SQLite commits after fsync of token files. Resume truncates uncommitted tails,
restores source cursors and dedup journal; pinned source rows may be rescanned
without retokenizing committed documents. Final manifest records SHA, provenance,
budgets, unique accepted document counts, rejected documents and duplicates.
An incomplete corpus cannot train. Completed files are hash-checked on load.
Keep this one corpus directory; copy it to the training host/read-only dataset.
No raw corpus or binary checkpoint is committed to Git. No HF upload is implicit.

## Evaluation, ceilings and durability

Source160-window ABC replay<=.03 mandatory BEFORE updating. Add D: first128
non-training C4 windows, SAME128 at source and endpoints/final; separate NLL.
Original Qwen D128 NLL is measured with the existing block-streamed teacher and
tiled head; its teacher evaluation positions are recorded separately from training.
Keep reused ABC and EN/RU diagnostics. A selects best for historical consistency;
D/EN/RU and matched current endpoints are essential to avoid judging new data
only on Wiki. D is diagnostic validation, not an untouched final benchmark.
Any capability claim needs a locked question/generation benchmark after training.

12h allocation:11.8h invocation includes18min save reserve, setup/compile/eval.
262144 NEW steps ceiling (67.1M positions), not a short stopping target. If data
ends, report data_exhausted and SAVE; no silent epoch. Source/global/Adam cursor
and corpus identity belong to the resume contract; altered corpus is rejected.
Own exp160-corpus-recovery-v1 namespace; old159 code/state untouched.
Same adapter-only checkpoint transport, bounded local generations, HF every4h
and exit,8GB/worker upload cap, parent start/finish notifications only. An18min
reserve is not a guarantee of network success; pending states remain explicit.

## Evidence levels

Tiny CPU tests exercise cross-split dedup, crash recovery, corruption rejection,
non-cycling exhaustion, split download stopping, real worker gradients, resume
next-window selection and source replay rejection; parent GPU isolation/failure.
These are not a1.7B physical GPU throughput, memory or scientific result.
See EXP-160-engineering-audit.json for actually completed local checks.

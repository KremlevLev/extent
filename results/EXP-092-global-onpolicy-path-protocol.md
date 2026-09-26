# EXP-092 — whole-model ONPOLICY path diagnostic

Status: prepared, not run on TPU.

## Why this follows EXP-091

The completed EXP-091 seed-123 pair worsened sharply from its common ONPOLICY
start under both full-model Lion arms; the fixed 0.03 backbone-update anchor
also lost to ordinary updates. Spending another long session on the same
optimizer would not test a new rescue mechanism. EXP-092 asks a cheaper,
necessary question: does the *joint, full-model* parameter direction learned
by EXP-091 contain a useful fractional point, even though its raw endpoint is
bad? This is not the per-coordinate filter tested in EXP-081–084; every
interpolated point changes all 24 Mamba blocks and copied Qwen parameters
together, retaining interactions among replacements.

## Frozen design

- Source: the exact hash-verified EXP-072-v2 ONPOLICY assembled starting
  models, seeds 123/456, and completed EXP-091 endpoints. Seed 123 contributes
  PLAIN and ANCHOR paths; seed 456 contributes only the completed ANCHOR path.
  The incomplete seed-456 PLAIN trajectory is not used or compared.
- Global interpolation in FP32 before BF16 storage at alpha
  `[0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]`.
- Select the lowest mean NLL on 16 fresh length-256 WikiText-103 validation
  windows beginning at offset 8192. Keep alpha zero unless the gain reaches
  0.01 NLL. These windows were not EXP-091's reported validation prefix.
- Evaluate only alpha zero, the selected alpha, and alpha one on a separate
  locked WikiText-103 **test** split: 32 length-256 windows from offset zero.
  Test metrics never select an alpha. Retain per-window NLL and source hashes.
- Primary gate: for ANCHOR at **both seeds**, selected alpha is nonzero and
  locked-test mean NLL improves by more than 0.01 versus alpha zero.
  Seed-123 PLAIN is contextual. A pass motivates a separate, pre-registered
  staged-training campaign; it does not itself demonstrate recovered quality,
  14B transfer or equal-compute superiority over random initialization.

## Operations

One notebook call, no new training and no optimizer states created. All source
manifests are checked on HF before loading Qwen. EXP-091 model+Lion payloads
are read only; no historical result or checkpoint is modified. Branch-level
JSON/Markdown progress uploads to a dedicated EXP-092 HF namespace, making
partial runs resumable. Telegram reports start, completion, deadline partial
or caught runtime failure. The 8.25-hour argument is a safety ceiling, not an
expected runtime. Stop as soon as all three paths are evaluated; do not fill
remaining TPU time with unjustified training. TPU runtime and peak HBM are
unknown until the first run.

# EXP172 — frozen clean EXP171 quality on 2×T4

Prepared 5 November 2026. No physical T4 result yet.

## Frozen source

CURRENT5120, 83,886,080 accepted input tokens, from EXP171, not the historical
random-prepared bank. HF dataset levkremlev888/extent revision
`d341d2b8f87592f617c44c92c5d7532669c75755`, prefix
`experiments/exp171-clean-qkvo-tpu-v1/ALIGN`.
Original Qwen/Qwen3-1.7B-Base revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
Download and SHA-verify only the 12 stacked mixer parameter leaves (~3.2GB),
not Adam or the training corpus. Transpose projection kernels only; compare
every converted coordinate exactly before placement. Original Qwen is frozen.

Three masks: original teacher (28 attention); gqa4 (24 Mamba + 4 attention,
indices 6/13/20/27); gqa7 (21 Mamba + 7 attention, 3/7/11/15/19/23/27).
These are ~2.13B/~2.08B hybrids versus the ~1.72B teacher, not matched sizes.

## Measurements

- Zero-shot ARC-Easy, ARC-Challenge, PIQA, WinoGrande, HellaSwag, MMLU;
  default 1000 examples/task (or full split if smaller). Fixed EXP156 seed/order,
  MMLU balanced round-robin across subjects. Tasks interleaved, not serial.
- Score answer continuation log likelihood only. Report acc and character-
  normalized acc_norm, Wilson intervals, matched teacher/hybrid differences
  with paired-bootstrap intervals. Wino/MMLU acc is primary; others acc_norm.
  These reuse the separately versioned EXP156 templates, NOT official harness
  CLI scores or official five-shot MMLU. Retain exact prompt hashes and counts.
- Independent English TinyStories and Russian Wikipedia NLL, plus broader Edu
  validation: 32 windows each at context256 and1024, same tokens for all roles.
  1024 panels contain 32,736 targets versus the old four-window4092-target panel.
  Panels at different lengths overlap intentionally and are not independent
  replications. Raw pins/checksums and panel-token hashes saved in contract.
  TinyStories is simple English, not a broad English-web quality measure.
- 24 fixed English/Russian/code prompts, greedy48-token continuation, including
  EOS; save complete strings/IDs. Qualitative diagnostic, no pass@1 claim.
- Full-prefix replay for generation, no inference-cache/latency benchmark claim.

Pin datasets using results/EXP-156-data-pin.json. No training or checkpoint
selection based on benchmark outcomes. No claim of pretraining decontamination.

## Precision and gates

PyTorch2.11.0 CUDA12.8, Transformers4.57.1; **no JAX/Torch-XLA** in runtime.
T4 projection/MLP GEMMs FP16, mixer coordinates and recurrent/SSD state FP32,
log-softmax FP32. Do not cast the entire Mamba bank to FP16.
Before collecting task results on each GPU:

1. Actual learned mixers0/13/26,65 tokens crossing SSD boundary: blockwise
   output vs independent token recurrence, relative L2<=.003.
2. Original HF full-model path vs explicit layer loop, FP32 real128-token input,
   relative L2<=.001.
3. Same composed models, FP16 vs FP32 token NLL on128 real Edu tokens:
   mean absolute difference<=.03, max<=.25, finite throughout.
4. Replay same four Edu1024 windows as TPU; mean NLL within .08 of teacher
   2.476397/gqa4 2.598124/gqa7 2.583224. TPU values are last evaluation4864,
   frozen checkpoint5120; tolerance covers both additional256 updates and
   BF16/FP16 differences. A failed gate stops that worker and retains logs;
   never silently relax thresholds, substitute old weights or report accuracy.

The gates bound conversion/precision differences, not prove all inputs equivalent.
The original four-window panel is a replay gate, not independent quality evidence.

## Operation / last cell

Import **notebooks/extent-t4-quality-172.ipynb** into a fresh Kaggle notebook.
Select GPU T4×2, Internet on; attach the same four secrets:
EXTENT_DEPLOY_KEY_B64, HF_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
Dedicated environment installs GPU dependencies; no nvidia-smi precondition.
Kaggle Python3.13 rejected the first notebook's ensurepip bootstrap before any
weights or GPU work. Fixed: `venv --without-pip`, then host
`python -m pip --python <env-python> install ...`. This officially supported
pip route installs into the isolated target even when it contains no pip.
No global Torch replacement, get-pip download, or sudo needed.
Default session budget12h includes installation/prepare/evaluation. Final cell:

```python
TOTAL_HOURS = 12
MAX_PER_TASK = 1000
```

The notebook subtracts setup time and invokes engine_quality172. Do not run this
from the old TPU dependency notebook. Command for an already prepared GPU env:

```python
from engine_quality172 import main
result = main(['--max-wall-hours', '11.5', '--max-per-task', '1000'])
```

Each GPU loads the full model+bank and handles its half of jobs. Every question
is completed on all three models before its row becomes durable. No DDP and
no assumption that two16GB GPUs form a32GB allocation. Inference batch1 avoids
padding and optimizer memory; there is no training-batch planner in this eval.
GPU gates include a brief FP32 path (~10GB parameter allocation before temps).

HF downloads use existing paced/jittered429/network retries and session deadline.
Source revision/manifest immutable. Completed questions/windows/prompts saved
as JSONL, fsync after each. Same session/work-dir can resume completed items;
discard a torn final line only. Fresh cloud sessions do not retain temp journals
unless exported/restored; this is not an automatic cross-session HF resume.
Worker errors are visible, completed matched rows survive. STOP file in work-dir
ends between units/roles/generation tokens; an interrupted unit is redone later.

Only two Telegram messages: start and final status, no getUpdates listener.
Logs/progress every minute in notebook. Defaults use /kaggle/temp for weights,
data and env, avoiding hundreds of checkpoint chunks in notebook Output.
Notebook copies final aggregate and worker logs to /kaggle/working/extent-results.
**Send only extent-m3q-quality172.json.** On failure this includes worker errors;
if a traceback is needed, send the corresponding worker log too.

Deadline partial results use exactly matched item sets for all three roles,
with per-task n; do not compare with published full-split Qwen scores as though
settings were identical. Actual completion time/throughput/memory unmeasured.

## Local validation

21 CPU tests passed (7 new EXP172 +14 existing clean PyTorch tests),84.12s.
Includes actual tiny Qwen teacher-path parity, mapped-leaf equality, JAX171
vs PyTorch blockwise forward with nonzero dynamics, causality, chunk integrity,
answer-only scoring and interrupted-journal resume. Real remote manifest5120
validated,3,193,268,992 parameter bytes. Notebook code cells parse.
No physical T4 speed/memory/precision-gate result observed locally.

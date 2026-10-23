# EXP-156 вЂ” frozen capability assessment, R256 hybrid versus original Qwen

Prepared24 October2026. One12h Kaggle GPU T4x2 session. Evaluation only;
no training, optimizer reset, parameter updates, folding of153/155, or source
checkpoint overwrite. TPU continuation is independent of this frozen snapshot.

## Models and isolation

GPU0: original `Qwen/Qwen3-1.7B-Base`, revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
GPU1: EXP155 CURRENT80260, rank32+224=256, 24Mamba3/4GQA in28layers,
MIMO rank4. The correction rank is a different quantity from MIMO rank.
Source HF dataset `levkremlev888/extent` at immutable revision
`74cd4878817a330e1ae8399acbafe5e76b40a4c5`, exact manifest/reference in
[EXP-156-source-pin.json](EXP-156-source-pin.json). Last A/B/C
5.109496/5.150924/4.665559. Selection fixed BEFORE capability evaluation;
do not substitute inherited A-selectedBEST78870 or a newer training checkpoint.

Only CURRENT correction parameters downloaded, no Adam/best/training-state
allocation on GPU. Reconstruct the validated old136-folded base exactly once;
keep trained old32/extra224 and pregate. Original source code hashes checked.
Preserve source BF16 weights/activations, FP32 correction arithmetic/log-softmax;
no unverified FP16 conversion for T4 speed. Two fresh processes with separate
CUDA_VISIBLE_DEVICES; the parent prepares shared data/model downloads once.

## Required preflight before accepting benchmark items

- Same JAX/jaxlib0.11.0,flax0.12.8,optax0.2.8,math settings as155.
- Hybrid: actual all160 per-window CURRENT A/B/C replay <=.03 against155.
  Actual/reference/worst measurements persisted before rejection. Recomputed
  on every invocation, including own-result resumes.
- Original (repaired after first startup failure): five short English/Russian,
  choice/math/code prefixes of the actual Qwen checkpoint compared with CPU
  Transformers4.51.0/FP32 eager attention against a temporary FP32 CPU JAX
  engine. Per-token NLL discrepancy must be<=.003. Parameters transfer to CPU
  BEFORE dtype conversion, and a CPU default-device context covers both the
  forward and vocabulary head. Temporary parameters are freed before evaluation;
  benchmark teacher stays BF16 on GPU. This is a CPU formula/mapping gate,
  not FP32 GPU implementation equivalence.
  CPU Transformers BF16 versus GPU JAX BF16 is also measured and retained
  per token, but is not a tight formula-equivalence gate across backends.
  It is not claimed bitwise-equivalent to official Transformers BF16 scores.
  This checks short prefixes, not every long prompt or an HF accuracy score.
- Runtime checks reject nonfinite choice/generation/panel losses.
  A failed worker stops the peer, saves original traceback, and does not use
  the remaining session for an invalid paired experiment.

The inference engine evaluates growth-aware blocks without a gradient graph.
Right padding never contributes to answer likelihood. No left truncation.
Context+continuation BPE prefix consistency checked. Score only continuation
tokens, never the question/prompt. Fused vocabulary head bounded to16positions
at a time. Batch2 choices; memory guard can retry identical choices one by one,
recording fallback instead of changing context/precision. Teacher tied head is
explicitly required. No generated Python is executed.

## Six zero-shot multiple-choice tasks

| Task | Split | Questions | Prespecified primary metric |
|---|---|---:|---|
| ARC-Easy | test |2376|acc_norm|
| ARC-Challenge | test |1172|acc_norm|
| PIQA | validation |1838|acc_norm|
| WinoGrande XL | validation |1267|acc|
| HellaSwag | validation |10042|acc_norm|
| MMLU | test, all57subjects |14042|acc|
| **Total per model** | |**30737**|No single pooled score|

Immutable dataset commits/files/SHA/bytes in EXP-156-data-pin.json. Six real
Parquet files were locally downloaded, hashes checked, every row formatted.
All30737 questions tokenized with the actual pinned Qwen tokenizer: no prefix
boundary failures and none over2048tokens. Max lengths ARC167/178,PIQA244,
Wino37,Hella148,MMLU1011. See data-audit/tokenization-audit JSONs.

Templates adapted from EleutherAI lm-evaluation-harness (MIT), with explicit
source task definitions: [ARC](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/arc),
[PIQA](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/piqa),
[WinoGrande](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/winogrande),
[HellaSwag](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/hellaswag),
[MMLU](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu).
Evaluator code/ordering hashed by its own contract; output is separately
versioned custom JAX zero-shot evaluation, not an official lm_eval CLI run.

ARC/PIQA/Hella score entire answer continuations; acc_norm divides log likelihood
by answer CHARACTER count, as these task definitions do. Also save raw acc.
Wino scores the COMMON sentence suffix given each alternate filled-in prefix,
not the probability of the option name. MMLU lists A/B/C/D and scores the
answer letter, with subject description and ZERO examples/CoT/chat template.

Seed156123 hashes doc IDs for deterministic order; MMLU alternates subjects.
Tasks interleave one question each. First25/task form a timed pilot, then
generations, then all remaining questions; no early stop at100/1000items.
Time limit may leave partial coverage. Report actual n/planned n, not a
full-benchmark claim. Original and hybrid can progress at different rates.
Primary comparison uses their exact common question IDs and prompt hashes.
Accuracy Wilson95% intervals and paired question-bootstrap differences
(2000resamples) are descriptive sampling intervals, not independent-seed
replication. Partial balanced MMLU is not the full-test weighted distribution.
Per-subject MMLU counts/scores retained; no assertion about recovered brain%.

## Language modeling and generations

Original source A/B/C also measured for the original Qwen to get a same-slice
reference; these are reused panels, not fresh tests.

New assessment panels: TinyStories English validation and Russian Wikipedia
20231101, pinned shard/commit/SHA. TinyStories is synthetic children's prose;
do not call it broad real-world English or code ability. Russian shard581MB
downloaded once, read in bounded batches. Deterministic document selection
from first4096rows, EOS between documents, exact token SHA recorded. Each
domain128Г—256windows (32640targets), distinct next8192tokens form8Г—1024 context
windows. Compare lengths128/256/512/1024 on the same last127targets.
Fresh means unused in OUR adaptation/evaluation, not absent from Qwen pretrain.
Actual TinyStories and all six MC files were packed with the actual tokenizer
locally. Russian packing used a synthetic Parquet fixture; the real581MB
Russian file and physical inference remain execution-time checks, not results.
See EXP-156-packing-audit.json.

24 fixed raw-completion prompts:8English,8Russian,8Python function prefixes.
All get greedy continuation up to64newtokens; every second prompt also gets
seeded top-k50/temperature.7 (12more), total36 generations per model. Same
prompts/budgets/seeds, no chat/thinking mode. Save token IDs, full decoded text,
EOS/max-length stop and repeated4gram fraction. These expose loops/broken
prose; they are NOT MBPP/HumanEval pass@1 or automated semantic correctness.
No inference-speed claim: generation recomputes prefixes without KV/SSM cache.

Official Qwen3 report Table8 gives Qwen3-1.7B-Base MMLU62.63 (5shot),
GSM8K75.44 (4shotCoT),MATH43.50 (4shotCoT),MBPP55.40 (3shot).
[Source](https://arxiv.org/html/2505.09388v1#S3). Context only: do not compare
our zero-shot MMLU directly as the same setting or substitute chat Qwen results.
Actual paired original-Qwen measurements are the primary baseline.

## Budget, resume, artifacts

11.5h TOTAL incl30min result reserve, leaving30min notebook setup within12h.
Data downloads/model restoration/JIT/replays/NLL/generations/MC all count.
No performance or full-completion promise without the actual pilot.
Continue remaining items until complete or deadline; do not idle to fill12h.
One worker failure stops peer early. Parent terminates overrun near hard end.

Atomic worker reports at least every60sec and each completed stage;
per-question append journal restores items between snapshots, contract checked.
HF aggregate every hour and final under `experiments/exp156-frozen-quality-v1`.
Upload cap1GB/invocation; no weights uploaded. Existing bounded429/network
retry policy installed before downloads. Own resume strict on source/code/data
and per-role hardware; merge only identical-contract reports. Repeating same
cell resumes remaining work; source replay/reference checks run again.
One narrow recovery exception: a FAILED old code-only contract with NO scored
items, NO NLL panels and NO generations may be archived and restarted cleanly.
The observed first startup report satisfies this condition. Changed source/data
or any recorded scores still reject cross-contract resume. Old traceback and
contract retained in the repaired aggregate; no scientific results mixed.
Quiet parent-only notifications: start,progress/4h,failure,finish,max5.

Download ONE `extent-m3q-quality-156.json`: both roles, per-question losses and
predictions, NLL panels, decoded generations, coverage/status, paired comparisons,
runtime/errors, source/data/code provenance. Local worker logs included on failure.
`deadline_partial` is not a scientific failure; `failed`/skips must stay explicit.
Report-sync error means local output is not confirmed on HF; use Kaggle Output.

See [EXP-156-launch.md](EXP-156-launch.md). No cloud evaluation has run locally;
the prepared suite is not evidence of model capability. Once these tests guide
future tuning, they become exposed validation and a separate final test is needed.


## Local validation evidence

Seven CPU tests pass22.11s. Actual mature BF16 R256, nonzero extraB, choice-only
loss agrees with full model; future-token/padding causality and batch1/2
agreement checked. Original tied-head teacher agrees with its independent
full Flax forward; logits and token losses checked. Embedding is a dynamic
head argument (not duplicated622MB executable constants per bucket).
Task formatting/BPE guards/subject ordering/paired statistics tested.
Actual worker mock exercises deadline, journal recovery, source replay,
36generation records and completion; parent test dispatches distinct GPUs,
merges identical question IDs and publishes one aggregate (mock transport).

Source audit checks parameter-subset metadata216chunks/228,830,208bytes,
57,200,640elements and original numerical code hashes. Local source156 tensor
payload/full1.7B forward/physical T4 peakmemory/throughput NOT tested. Actual
T4 source replay and CPU Transformers original-reference checks mandatory.
No benchmark capability result exists at preparation time.

## Received startup failure and repair, 24 October2026

`EXP-156-startup-failure-1.json` is the byte-preserved HF aggregate. No MC
questions, NLL panels or generations scored. Teacher loaded on TeslaT4 and
failed the CPU Transformers BF16 versus GPU JAX BF16 <=.03 check:
EN max per-token delta .1038608551, RU .0289543867. Peer stopped while loading.
This identifies the failing check, not proof of the numerical cause. BF16
rounding/fusion and backends differ; the old comparison did not isolate a
formula/mapping error from precision effects. Pip dependency warnings were not
the raised exception; both GPU engineering preflights passed.

Repair uses the independent FP32 formula gate described above, retaining the
BF16 disagreement as explicit diagnostics, and prints original traceback/log
tails in Kaggle stdout. Hybrid replay tolerance/source/precision unchanged.
The new CPU test uses actual tiny Qwen weights in Transformers and JAX;
injected BF16-only diagnostic drift remains visible, while deliberately
changed HF output-norm weights must fail the FP32 gate. Full1.7B GPU peak
memory and physical T4 FP32 parity remain runtime checks, not local results.

Repair validation: nine CPU tests pass45.17s; final reference and parent-recovery
tests rerun after the archive-path assertion, two pass22.87s. Actual tiny FP32
maximum per-token discrepancy4.76837e-7; deliberately doubled HF final norm
gives .8924978 and is rejected. Full diagnostic rows in
`EXP-156-reference-repair-audit.json`. Neither number is a full1.7B GPU result.

## Second startup: FP32 GPU duplicate exhausts memory

`EXP-156-startup-failure-2.json` preserves the received HF report. Teacher fails
in the temporary FP32 bridge's vocabulary-head compilation/autotuning:
`RESOURCE_EXHAUSTED`, allocation9.27MiB, surfaced as `NOT_FOUND: All configs
failed during profiling`. No scored items, NLL panels or generations; hybrid
peer stopped loading. This is a memory failure, not a failed FP32 parity value.

The second repair places that entire temporary bridge on CPU as specified
above. GPU-to-CPU transfer precedes FP32 casting; all parameter devices are
checked, and the forward/head run inside the explicit CPU device context.
The same strict .003 formula gate and BF16 benchmark math remain. BF16 actual
GPU vs CPU Transformers diagnostics are still measured on the original engine.
The empty failed preflight recovery also covers this startup. No claim is made
that physical1.7B T4 memory/quality has been validated locally.

# EXP175 · Two-hour TPU objective diagnosis

EXP174 retained source quality at LR1e-6, but its selected full-student branch
finished slightly worse after79.46M new inputs. Higher rates worsened the
matched short endpoints. This experiment tests the learning signals from
clean EXP171 CURRENT5120 before spending another long run on continuation.

## Common settings

Seven retained attention blocks,21 Mamba mixers, all active student weights
trainable, separate immutable original Qwen teacher. Use the verified
600M FineWeb-Edu pack, context1024, effective batch16, physical batch8.
No new speed sweep. Both batches8/16 fit EXP174, and8 was faster.
Only an actual OOM in the disposable initial diagnosis permits retry at4.

All three pilots start from the same source, fresh Adam(.9,.95), stable
global clipping at1,32-update warmup to constant LR1e-6, and data cursor
83,886,080. Objectives are CE, exact full-vocabulary forward KL at T1,
and0.5CE+0.5KL. CE-only and KL-only have coefficient1. Component norms and
clipping therefore need to be considered when interpreting the comparison.
Teacher values and both metrics are computed for every arm, including CE.

## Measurements

Before training, a separate disposable mixed-objective worker computes
CE and KL gradients on the same16 sequences. Report their scaled norms
and cosine globally and for Mamba, stacked Qwen blocks, last block,
tied embedding/head and final norm. Zero-norm cosines have an explicit flag.
Then do32 constant-LR1e-6 updates on that fixed batch with fresh Adam.
Report CE/KL before, after the first step and at the end. Record the angle
between the actual first Adam displacement and its gradient, together with
fractions of changed FP32 coordinates and changed BF16 casts per group.
Fractions use all stored coordinates in each group, including unused
attention/mixer slots. Their zero gradients also contribute to that count.
BF16 casts are a proxy, since recurrence and some other operations use FP32.
These diagnostic updates are disposable and excluded from pilot trajectories.

Each pilot also measures its first and final update on the exact batch used
for that update, before and after. This distinguishes immediate optimization
behavior from validation generalization. An individual increase under BF16
does not by itself prove an incorrect gradient.

Start and endpoint validation cover all128 available Edu1024 windows.
Per-window CE/KL remain in the aggregate report for paired offline analysis.
This expands the previously used panel and includes windows used before.
It is not a new untouched test set. Four-window monitor checks every64 steps.
The starting four-window replay must match the pinned source within tolerance.
No NLL-based early stop is applied.

The target is256 matched updates per arm,4,194,304 input positions each.
After the diagnostic measures full-update speed, the supervisor chooses a
shared budget in blocks of32, up to256, retaining startup/checkpoint time
and an18-minute final reserve. Fewer than64 affordable updates leaves a
partial diagnostic report. All complete endpoints have identical budgets.
An incomplete or failed arm stays in the report and cannot win selection.

Select minimum endpoint CE NLL on the128-window panel. Continue the selected
pilot's own weights, Adam and cursor until the work deadline, even if NLL
has increased. Only its checkpoint is uploaded and verified on HF under
`experiments/exp175-objective-diagnostic-v1/<arm>`. Source code hashes,
data identity, exact budget, failures, and all candidate results are retained.
Selection on this panel needs independent capability validation afterwards.
Downstream task benchmarks are outside this short session.

## Launch

Use `notebooks/extent-tpu-175.ipynb`, TPU v5e-8, internet and the existing
four secrets. The two-hour budget includes installation and downloads.
Large model/state files stay in tmpfs. The parent stays CPU-only and
collects separate worker logs. Three normal Telegram messages cover start,
selection and final status.

For an existing EXP174 notebook, keep its first two cells and replace the
third cell with the third cell of the EXP175 notebook. A fresh cloud run
clones the updated repository. Keep `SESSION_STARTED` from the first cell
so installation time is deducted. Send `extent-results/extent-m3q-tpu175.json`.
If the process was killed, send that aggregate with the affected worker log.
Use a fresh tmpfs work directory for a new matched comparison.

Local verification passed14 CPU tests across EXP173/174/175. They cover
eight-device virtual sharding, matching the EXP174 mixed objective and
gradients, frozen-teacher gradients, finite scaled gradient diagnostics,
actual Adam displacement, checkpoint Adam/cursor round-trip, CPU-only parent
import, failed-arm recovery/selection and notebook cell syntax. Physical
TPU175 throughput, peak memory and scientific outcomes remain unmeasured.

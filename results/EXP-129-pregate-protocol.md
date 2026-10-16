# EXP-129 — competitor-informed internal normalization audit

PREPARED, 16 October 2026. One Kaggle T4×2 session, 11.5 hours including
60 minutes for persistence and final measurements. Single seed exploratory
intervention. No physical GPU outcome yet. Existing T4 notebook, last cell only.

## Source and attribution

Read-only inspection of arianraje/qwen3-4b-mamba3-hybrid-stage2b-kd-bias1,
revision `b946069890837bbe9b5dd75e2d118a7e98d2d193`.
HF metadata declares Apache-2.0. Inspected source:
[modeling_qwen3_mamba3.py](https://huggingface.co/arianraje/qwen3-4b-mamba3-hybrid-stage2b-kd-bias1/blob/b946069890837bbe9b5dd75e2d118a7e98d2d193/modeling_qwen3_mamba3.py),
SHA256 `6ca2cee4665ac88b28cb81470179f635287e983a28c4079afe2b085ceaf55f77`.
Remote Python was read, not imported/executed; no competitor weights downloaded.
New JAX implementation extends our own reference recurrence in a separate module.
Normalization is a standard upstream Mamba option, not claimed as our invention
or as a novel contribution by the competitor.

Confirmed source details: reference lines 454–457 normalize the state readout
PLUS D skip over head channels P, separately for each token, head and MIMO rank;
then multiply shared H×P weights and SiLU gate, then reduce ranks with mimo_o.
Fused path lines 695–697 explicitly enables the same normalization. Epsilon1e-5.
Our historical core has B/C normalization but lacks this readout normalization.

Other observed differences are NOT bundled into129: 4B versus1.7B teacher,
27/36 versus24/28 replaced layers, head/state128 versus64, groups8 versus1,
expand1.6 versus.75, longer-context full-model KL training. Competitor reference
also bounds raw angle projections with tanh; ours currently uses raw angle
increments. Initialization zeros dynamics rows, uses simplex rank value factors
and an inherited-head rank selection. These require separate audits against our
initialization history; this document does not establish that each differs at
every historical Extent checkpoint or that any one causes the quality gap.

## Paired question and arms

Does adding precisely this internal normalization improve continued recovery
from the existing plateau? This is NOT a reproduction of the competitor's
from-conversion training recipe and cannot rule that recipe out if it fails.

| GPU | Method |
| --- | --- |
|0|R32-CONTROL: protected R32, unchanged mixer math|
|1|R32-PREGATE-RMS: same R32 plus headwise pre-gate RMSNorm and learned H×P weights|

Both start at pinned EXP111 seed123 MAIN32768 using EXP114 source pin. Frozen
base BF16 kernels, trained R32 coordinates restored, fresh Adam for both.
Embeddings, retained GQA, MLP and dynamics coordinates remain frozen. No learned
output gains, residual cap, external mixer cap or KD in either arm. Treatment
has 24×1536=36,864 extra norm weights on the real1.7B configuration; actual
coordinate count is recorded. Thus this is a recipe comparison, not a strictly
parameter-count-matched placement-only ablation. A win warrants fixed-weight
and fresh-seed follow-ups.

For rank readout u=(C·state + D·value), apply

    u_new = u * (mean_P(u²)+1e-5)^(-s/2) * (1+s*(w-1))

where s=min(step/2048,1), w starts at1 and is shared across ranks within each
head. At full strength this is the inspected normalization; no detachment or
straight-through gradients. At zero strength the source mixer executable is
used exactly. The ramp is OUR recovery adaptation, not the competitor's method.
Scale-stable evaluation avoids FP32 square overflow. Norm can amplify small
vectors; it is NOT clipping. It does not directly bound the SiLU gate or state.

## Budget, measurements and persistence

32768 updates per arm, same batch1×256, 255 next-token targets/update,
FIT16384×256 and fixed permutation from120–128. Full horizon8,388,608 input
presentations/arm; distinct training pool4,194,304 tokens; earlier preparation
excluded and must remain separately accounted for. Adam .9/.999 eps1e-8,
joint stable gradient clip1, R32 LR3e-4, norm LR1e-4, LR warmup256.
Actual completed horizon depends on GPU throughput; same cell resumes.

Source A64/B32/C64 controls and same-tail127 context128/256/512/1024.
Initial full-strength FIT8 and profile BEFORE training, recorded even if NLL
is poor; only numerical/parity failure stops it. A16/B32 monitor before2048;
A64 selects best ONLY at full strength at2048/4096/8192/16384/24576/32768.
Post-ramp endpoints include B32/C64 and layer profile. Last and eligible best
receive all panels and context probes at exit. A/B/C previously exposed, not
independent tests. Compare the LAST COMMON full-strength endpoint between GPUs;
do not compare mismatched A16/A64/C64 or best/last or unequal step counts.
Exploratory useful signal: paired A64 improvement at least0.1 NLL with B/C
direction reported, not a post-hoc pass/fail replacement for confirmation.

Profiles: per-layer residual/mixer/MLP RMS and peak, fraction of nonzero mixer
and MLP contributions lost in BF16 addition; pre/post internal head-readout
RMS mean/max and SiLU gate RMS mean/max, learned norm weight range/RMS. Internal
readout summaries aggregate P-normalized head/rank groups and time, separately
from whole residual RMS. Source and full-strength diagnostics show the initial
intervention shock. Source replay tolerance.03; same-GPU hook background.01;
ordinary/profile relative discrepancy.02, unchanged from previous campaigns.

Local snapshots128; remote2048 or120min plus exit. Current corrections, Adam,
best coordinates and exact ramp/data cursor restored. Default4GB new uploads
TOTAL across both workers per invocation; cap becomes storage_partial, no
deletion. Clean timeout saves last accepted state and finalizes in reserve;
external kill may lose unsynced cloud progress. Source/base weights never
duplicated into fit checkpoints. Telegram start/progress/end/pending notices.

Legacy124–128 source and contracts remain unchanged. Normalization only lives
in the independent129 core; successful results are needed before default core
or paper conclusions change.

## Last cell in the existing T4 notebook

```python
import os
import subprocess
import sys

repo = "/kaggle/working/extent"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, env=git_env, check=True)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)

from scripts.m3q_t4_pregate_129_campaign import main
result = main([
    "--experiment", "129",
    "--max-wall-hours", "11.5",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "4",
])
```

git_env comes from the existing deploy-key clone cell. Keep original setup,
library paths and Secrets. If setup consumes more than30min of the12h runtime,
reduce runner wall limit. Download ONE aggregate file:
`/kaggle/working/extent-results/extent-m3q-t4-129.json`.

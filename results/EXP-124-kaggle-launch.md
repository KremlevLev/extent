# EXP-124 — short versus long context, paired TPU recovery

Status: PREPARED, physical TPU outcome pending. One seed123, one TPU v5e8,
9h allocation. Uses existing TPU notebook; no new notebook or CUDA setup.

## Question and controlled intervention

EXP117 found almost no benefit from128→1024 inference context for the hybrid,
while original Qwen benefited. EXP115 rank growth gave a small PG19 improvement,
not a plateau breakthrough. This experiment asks whether training only256-token
windows contributes to lost context use. It does not assume activation scale is
the sole cause. It complements scale/KL GPU120–123 rather than repeating them.

Source: pinned EXP111 seed123 MAIN32768, reconstructed by the unchanged EXP115
loader. Protected rank32 corrections retained. Both arms train those coordinates
and24 Mamba output log-gains, initially gain1. Frozen base, MLP, attention,
embeddings and readout. Fresh Adam for BOTH arms (source moments deliberately
discarded): LoRA LR3e-4, gain LR.01,256-update warmup, stable joint clip1.
Gain range exp(clip(raw,-30,3)); no activation normalization or KL objective.

| Arm | Tokens per update | CE targets | Available context |
| --- | --- | --- | --- |
| SHORT | 4 independent windows ×256 | 4×255=1020 | up to256 |
| LONG | 1 contiguous window ×1024 | 1023−3=1020 | up to1024 |

Input token IDs and ordered CE target IDs are identical for each paired update.
LONG omits loss indices255,511,767, which predict token positions256,512,768.
The omitted predictions still exist in the causal computation. SHORT resets
Mamba and attention at each window boundary. LONG keeps the prefix available.
Positions also reset with SHORT boundaries; this is a training-context package,
not a recurrence-only intervention.

Training pool: same pinned EXP115 4,194,304 tokens, reshape4096×1024.
Permutation `(step*7919+120123)%4096`, repeated across epochs, identical per arm.
Planned horizon32768 updates/arm =33,554,432 additional input presentations and
33,423,360 CE targets/arm; only4,194,304 unique pool tokens. This is a long
horizon, not a promise that9h completes it. Preparation and historical training
are excluded from these additional-token counts. Attempted/replayed updates
charged separately in aggregate cost; compare equal updates and report wall cost.

## Measurements

- Source canonical per-window A64 NLL parity against EXP115 fixture ≤1e-4;
  gain1/tiled-head path difference ≤.01 per window before training.
- Alternating256-update blocks; less advanced arm first. A16 NLL at256,1024,
  4096,8192,16384,24576,32768. Log loss, stable gradient norm, gain trajectory,
  attempted updates and training seconds (including first compilation).
- End/reserve: source and both trained arms, context128/256/512/1024 on the
  SAME last127 targets of each contiguous1024-token block. Panels: EXP115
  A64→16 blocks, Wiki64→16 blocks, PG19128→32 blocks. All historically exposed,
  NOT new independent tests. Qwen context reference on the same A blocks.
- Final evaluation uses the last common in-memory pair when a deadline leaves
  one arm ahead. `comparison.evaluated_steps` identifies the scored cursors;
  `latest_steps` identifies saved training cursors. If no common pair exists
  after unequal remote restore, unequal comparisons are explicitly marked.
- Layer profile on one A1024 block: residual/mixer/MLP RMS, maxima, nonzero
  additions absorbed in BF16. Instrumented forward discrepancy must be≤.02
  relative to ordinary forward to accept the profile. This is NOT the rejected
  recurrence trace. Profile scale alone does not establish causality.
- Device memory counters, per-update time, all costs in one aggregate JSON.
  Positive signal: LONG beats SHORT on identical tail targets AND gains more
  from longer inference context at equal updates. A drop only on long windows
  without matched target comparisons is insufficient. One seed is exploratory.

## Session, durability, outputs

8.8h from runner entry includes data/source reconstruction, compilation and
saving. Default60min reserve: training ends at7.8h; final diagnostics can use
the first half, at least30min left for upload. Compilation/single device calls
cannot be interrupted exactly at the deadline. Infrastructure retries may leave
`finalization_partial` or pending sync; these are not completed experiments.

Local current+Adam snapshots every128 updates and arm boundaries. HF snapshots
at paired4096 endpoints or90min, plus exit.2GB new-upload cap per invocation,
explicit pending/durable cursors. Source artifacts are read-only; no full-model
checkpoint uploads. Older HF generations are not deleted. Remote roundtrip
before training; chunk SHA checks plus remote manifest/sample verification.
Same cell resumes local state or latest uploaded state; cloud loss before sync
can require replay. Report cursors roll back to actual restored checkpoints.
Completed runs only retry sync. `--sync-only` uploads local pending state.

Notifications: start, each arm block, training deadline, final saved/pending.
Return ONE file: `/kaggle/working/output/extent-m3q-context-124.json`.
Optional shorter file: `extent-m3q-context-124-summary.txt`.

## Last cell in the existing TPU notebook

```python
import os
import subprocess
import sys

repo = "/kaggle/working/extent"
subprocess.run(
    ["git", "pull", "--ff-only"],
    cwd=repo,
    env=git_env,  # from the existing deploy-key clone cell
    check=True,
)
os.chdir(repo)
if repo not in sys.path:
    sys.path.insert(0, repo)

from scripts.m3q_context_recovery_124_campaign import main
result = main([
    "--max-wall-hours", "8.8",
    "--save-reserve-minutes", "60",
    "--max-new-hf-gb", "2",
])
print("status:", result["status"])
print("pending:", result["pending_slots"])
print("report:", "/kaggle/working/output/extent-m3q-context-124.json")
```

Use TPU v5e8, existing TPU dependencies and enabled HF/Telegram secrets.
`git_env` belongs to the supplied clone cell; it carries the custom SSH-key path.
Do not replace TPU setup with the T4 notebook's CUDA install.

## Engineering validation

Tiny CPU model: exact target-ID matching across all4096 blocks, tiled/dense
head parity, LoRA gradients against monolithic full model, nonzero gain
gradients, sharded update with8 virtual devices, exact serialized optimizer
continuation after restoring the registered layouts, nonfinite rejection,
cursor rollback. Physical1.7B TPU compilation, HBM, speed and scientific result
remain unmeasured until the cloud run.

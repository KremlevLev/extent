# EXP-069 — whole-model allocation with matched Mamba preparation

Status: pre-registered, no TPU result yet.

## Question

At a fixed four-attention/24-Mamba budget, does the EXP-068 allocation improve full-model language-model likelihood relative to the evenly spaced EXP-064 allocation when both receive the same new Mamba preparation recipe?

| Arm | Retained GQA layers (zero-based) | Remaining mixers |
|---|---|---|
| UNIFORM | 6, 13, 20, 27 | 24 Mamba-3 MIMO |
| ATLAS | 0, 1, 26, 27 | 24 Mamba-3 MIMO |

Source: `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
Seeds: 123 and 456. Both arms have identical parameter counts. Execution order is counterbalanced by seed.

## Shared preparation

Each layer/seed endpoint is prepared once and reused by both arms wherever that layer is Mamba. Saved payload hashes document this equality. A bank covers the union of their Mamba positions (27 layers; only layer 27 is attention in both arms).

Preparation uses canonical random initialization keyed by model seed and layer, 1,024 exact SSD-dual preparation steps, readout calibration, and 2,048 recurrent decoder-aware recovery steps. This retains the legacy dual recipe that was measured in EXP-065–068. The control is the same recipe with a different attention placement; EXP-064's random/exact/hidden arms are not repeated.

Preparation sequences contain 64 tokens. Training cache: WikiText-2 train offset 1,179,648, eight calibration windows, 2,048 recovery windows. Local diagnostics: validation offset 196,608, 32 windows. Both allocations consume 24 prepared endpoints per seed, with equal preparation-step budgets; the combined experiment constructs 27 unique endpoints per seed to share work.

## Whole-model recovery and endpoints

- All student parameters train; Qwen3 teacher parameters remain frozen.
- Retained attention remains GQA. MLA is excluded from this placement intervention.
- BF16 model/gradient/Lion momentum; FP32 loss and gradient-norm reductions; clip norm 1.0.
- 3,072 updates at batch size one and sequence length 256 (786,432 input-token presentations per arm).
- Lion LR 3e-5, warmup 128, cosine decay, no weight decay; teacher KL temperature 2 and true-label CE coefficient 0.1.
- Identical shuffled training token stream: WikiText-2 train offset zero, data seed 20260906. Token array SHA-256 is recorded.
- Evaluation: WikiText-2 test offset zero, 64 windows of length 256, identical across arms. This split was not used by EXP-067/068 allocation fitting. Treat it as experiment evaluation and keep a separate untouched corpus for later paper-level confirmation.
- Checkpoints for metrics: steps 0, 256, 1,024, 2,048, 3,072. Record student NLL, teacher NLL, excess NLL, prediction KL, top-1 agreement, and per-window student NLL.
- Primary gate: ATLAS must improve both final NLL and normalized NLL-curve AUC for each of the two seeds. A failed gate remains a result; model parameters and numerical health must still be reported.
- This is a two-seed controlled experiment, not a confidence claim about all model families, context lengths, or attention allocations. No gain percentage from local decoder L2 is interpreted as whole-model quality.

## Resume and runtime

Default session wall target: seven hours, with 20 minutes reserved for final saves/uploads. A new cache build requires another 20 minutes of headroom. XLA compilation, cache creation and network calls are not preemptible; this is a soft limit below the nine-hour session cap.

Prepared endpoints and full parameters/Lion/cursor/metrics use verified checkpoints. Full checkpoints are saved initially, every 1,024 updates, at completion, and on a controlled deadline exit. An unexpected failure resumes from the last durable checkpoint. A partial layer preparation may be regenerated; previously completed layer/seed endpoints are skipped.

Private HF namespace: `experiments/exp069-allocation`. Each remote state upload commits payload and matching metadata together. Local full state defaults to `/dev/shm/extent-exp069-state` to avoid Kaggle's small output disk. A 40-GiB checkpoint workspace budget is checked before computation. CPU host serialization temporarily uses additional RAM. Failed uploads retain local state and are explicitly recorded; a remote restore error does not silently start training again.

Output: `extent-m3q-allocation-campaign.json` and `extent-m3q-allocation-campaign-summary.md`. Telegram announces start and terminal status. Remote synchronization is required by default; explicit `--no-hf-sync` disables it for deliberate local-only use.

## Notebook entry point

After the normal repository update and dependency setup, run in the notebook process:

```python
from scripts.m3q_allocation_campaign import main as run_exp069

result = run_exp069([])
```

Run the same cell in subsequent sessions if the result is `deadline_partial`. Do not transfer large checkpoints through the user's PC. A smaller remaining session budget can be set with `run_exp069(["--max-wall-hours", "4.5"])` without changing the registered training protocol.

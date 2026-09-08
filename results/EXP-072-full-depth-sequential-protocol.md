# EXP-072 — full-depth sequential recovery confirmation (pre-registered)

Date: 2026-09-08. Status: v1 startup failed before training; v2 prepared with a valid fresh-data range. No model metric was observed, so the scientific test remains unconsumed.

## Question

EXP-071 showed that sequential recovery on hybrid-generated inputs prevents a depth-12 collapse and that pure `ONPOLICY` is the best observed arm. EXP-072 asks whether this survives fresh text, four times more updates, twice the training context, and all 24 Mamba positions of the Qwen3-1.7B 15/85 proxy.

## Frozen design

- Source: pinned Qwen3-1.7B-Base with retained source GQA at `[6,13,20,27]`; all other 24 layers are replaced in model order.
- Starting points: exact seed-123/456 EXP-069 prepared endpoints, independently restored and hash-checked for every arm/layer. EXP-071 trained endpoints are not reused.
- Arms: equal-update `TEACHER`, `ONPOLICY`, and `MIXED`. Only the current Mamba subtree trains; all Qwen modules and earlier recovered endpoints are frozen.
- Budget: 4,096 BF16 Lion updates per layer, context 128, 512 training windows, LR `3e-5`, warmup 64, no decay, clipping 1.0. The contribution-relative-MSE objective and conditional Qwen target implementation are unchanged from EXP-071.
- Fresh training data: WikiText-2 train offset 2,424,832, covering `[2,424,832, 2,490,368)`. This is disjoint from EXP-071's range and fits the observed 2,540,999-token capacity of the pinned WikiText/Qwen tokenizer pair. Fresh evaluation: 32 length-256 WikiText-2 test windows at offset 98,304. Token hashes are part of the resume contract.
- Evaluations after 1/4/8/12/16/20/24 replacements. No-extra-update prepared compositions remain references.

## Registered decision

`ONPOLICY` is now the primary arm, selected before EXP-072 from EXP-071's secondary result. It must beat equal-update `TEACHER` in final held-out NLL and milestone NLL-delta AUC at both seeds. `MIXED` remains a fixed secondary control. This is a fresh-data and longer-budget confirmation, not an independent-initialization replication because the same two prepared model seeds are reused.

A pass establishes full-depth sequential recovery value on the 1.7B proxy. It does not establish recovery to original Qwen, 14B scale transfer, MLA interaction, long-context behavior or inference speed.

## Runtime and durability

The campaign contains 589,824 current-Mamba updates and substantially larger caches/evaluation than EXP-071. Its expected range is roughly 5–8 TPU hours, but runtime is not a result. The soft limit is eight hours with a 20-minute reserve; partial completion is resumable.

The v1 launch stopped at `startup` after approximately 0.020 hours because its requested train range ended at token 2,686,976, beyond the observed 2,540,999-token corpus. It produced zero branches, baselines, checkpoints or model metrics. The v2 amendment changes only the unseen training slice and operational namespace. A deterministic range guard now rejects an invalid slice before campaign setup. V2 uses a separate local directory, in-memory state directory and HF prefix so the failed v1 resume record cannot contaminate it. The supplied full v1 artifact has SHA-256 `510cc2365c72af4f98590828e1f3818a9335bfa61dc72ff47eea45bb6a1f7b46`; its summary has SHA-256 `2bec7c0cdb96adf24c14290b0967694b5a0b47a5934efd0a87ccb3e2cce6d710`.

Every completed layer endpoint is immediately uploaded to the private HF dataset. Compact summaries upload only at registered depth milestones and terminal states. Transient checkpoint failures receive six attempts with 2/5/15/30/60-second waits; persistent failure stops before dependent work. Re-running the identical cell restores exact-contract progress.

Kaggle entry point after the existing setup cells:

```python
from scripts.m3q_full_depth_sequential_confirmation import main as run_exp072

result = run_exp072([])
print(result["status"], result["aggregate"])
print("/kaggle/working/output/exp072-v2/extent-m3q-full-depth-sequential-confirmation-summary.md")
```

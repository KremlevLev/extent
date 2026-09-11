# EXP-080 — second on-policy coordinate sweep (pre-registered)

## Question

Can a second local recovery pass reduce the composition error left after EXP-072, and does causal forward order matter? EXP-072 trained each replacement once as the hybrid prefix grew. Earlier layers were never revisited after all later replacements changed the complete model.

## Starting point

- Qwen3-1.7B hybrid with all 24 byte-identified completed EXP-072 `ONPOLICY` endpoints, seeds 123 and 456.
- Four retained source-faithful GQA layers at `6/13/20/27`; no MLA.
- Every arm starts from the same complete full-depth parameter set. No layer is reinitialized.

## Arms

- `FORWARD-SWEEP` (registered primary): revisit Mamba layers `0→26` in causal depth order.
- `REVERSE-SWEEP` (matched order control): revisit the identical layers `26→0`.

For the current layer, the campaign computes inputs produced by the current hybrid prefix, applies the frozen matching Qwen decoder block to those exact inputs, and minimizes FP32 relative MSE of the decoder contribution. Only that Mamba subtree trains. After its endpoint is durably uploaded, the updated block becomes part of the model used to produce later inputs. Thus the arms differ only in coordinate-update order.

## Training and evaluation

- 2,048 BF16 Lion updates per layer; LR `3e-5`, warmup 64, cosine decay, no weight decay, clip 1.0.
- 512 length-128 WikiText training windows at offset 1,310,720, distinct from the original EXP-072 recovery slice; deterministic order.
- Full 24-Mamba model evaluation on 32 length-256 test windows at offset 131,072 before the sweep and after 6/12/18/24 updated layers.
- 196,608 local optimizer updates total across two orders and two seeds. Each completed layer is an independent HF resume boundary.

## Gate and boundary

At both seeds, FORWARD-SWEEP must finish below its own initial NLL, never exceed 1.25 times start at a milestone, and beat matched REVERSE-SWEEP at 24 updated layers. Best-checkpoint or per-seed order selection cannot pass the gate.

A pass establishes iterative causal coordinate recovery as an actionable M3Q method rather than a one-pass initialization trick. A reverse advantage would identify update order as important but require locked confirmation. Both failing would show that another local pass cannot close the remaining full-model gap under this token budget.

Expected TPU v5e-8 runtime is approximately 4–7 hours, dominated by repeated current-prefix materialization. The campaign is designed for a full Kaggle session and resumes from every uploaded layer endpoint.

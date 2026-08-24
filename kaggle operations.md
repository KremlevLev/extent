# Kaggle Operations Runbook

This file is operational documentation, not part of the scientific report or
paper. Kaggle accelerator sessions are assumed to have a 10-hour ceiling.

## Session timing

- Treat 9 hours as the training deadline for a session.
- Stop launching long evaluations or new compiled shapes after 8 h 30 min.
- Begin the final checkpoint no later than 9 h 10 min.
- Reserve the remaining time for checkpoint validation, upload/export, logs,
  and a Telegram success or failure notification.

## Recoverable checkpoint contract

Every resumable checkpoint must contain:

- model parameters and Lion momentum;
- global step and processed-token count;
- learning-rate schedule state;
- JAX training, dropout, and data-order RNG state;
- dataset revision, shard index, sample/token cursor, and shuffle seed;
- model config, mesh, precision, source revision, and code commit;
- latest training/evaluation metrics and checkpoint checksum.

Write checkpoints atomically to a temporary sibling directory, validate that
all expected leaves exist, then promote the completed directory. A checkpoint
is not durable merely because it exists under `/kaggle/working`; before the
session ends it must be exported through a persisted Kaggle output/dataset or
downloaded to other durable storage.

## Resume gate

Before the first resumed update, print and verify the checkpoint commit, config
hash, global step, processed-token count, data cursor, and optimizer state. Run
one finite-loss step and confirm that the counter advances exactly once. Never
silently restart the scheduler or data stream from zero.

## Planned automation

The full trainer will checkpoint periodically and on a time-budget stop signal,
retain at least the latest two validated checkpoints, update a small manifest,
and send Telegram notifications for start, periodic progress, final save,
failure, and successful resume.
# EXP-044: one-shot two-layer composition

Run only on a fresh Kaggle TPU v5e-8 session after installing `requirements-tpu.txt` and restarting the notebook kernel if pip replaced JAX. Do not run `pytest` in a subprocess after JAX has already claimed the TPU.

```python
from scripts.qwen_two_layer_composition_run import main as exp044

result = exp044([])
print("EXP-044 scientific gate:", result["scientific_gate_passed"])
print("result: /kaggle/working/output/exp044-two-layer-composition.json")
```

The runner creates the layer-0 and layer-18 train/validation activation caches, trains the paired three-seed replacements, evaluates all ten branches, prunes consumed Qwen shards, and writes the final JSON into Kaggle output. Download only `exp044-two-layer-composition.json`; the activation arrays and training intermediates are disposable after the run.

EXP-044 resume is enabled by default. Completed caches are verified by content hash, and trained endpoint bundles are restored only when their full compatibility contract matches. If streamed evaluation hangs, cancel only the running cell and execute the same `exp044([])` call again in the same Kaggle session. Look for `RESUME-PASS` for all four caches and both training layers; the rerun will then restart only the streamed decoder evaluation. Progress is recorded in `/kaggle/working/output/exp044-stage-manifest.json`, while endpoint weights live under `/kaggle/working/output/exp044-endpoints/`.

## EXP-048: long-horizon scaling campaign

Run this as one cloud `Save Version` job on a fresh TPU v5e-8 session. The expected duration is roughly 4--8 hours, but Kaggle scheduling and compilation can vary. The campaign runs fresh 1,024-step and 4,096-step comparisons at layers 0 and 18, with three seeds and three objective arms.

```python
from scripts.qwen_long_horizon_campaign import main as run_long_horizon

result = run_long_horizon([])
print("EXP-048 numerical pass:", result["passed"])
print("EXP-048 scientific gate:", result["scientific_gate_passed"])
print("result: /kaggle/working/output/extent-long-horizon-campaign.json")
```

The runner sends one Telegram message at start and one on completion or caught failure. Each completed budget and layer is a restart boundary. It writes raw LM metrics before aggregation and removes large NPY/endpoint payloads only after their complete result JSON exists. If a recoverable Python exception occurs in an interactive session, call `run_long_horizon([])` again without deleting output. For a new Kaggle session with no attached prior output, the campaign necessarily starts fresh.

The only file required for handoff is `/kaggle/working/output/extent-long-horizon-campaign.json`. On failure, retrieve `/kaggle/working/output/extent-long-horizon-campaign-failure.json` plus `/kaggle/working/output/exp048-campaign-stage-manifest.json`; do not download multi-gigabyte endpoint payloads.

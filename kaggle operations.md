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

## Compact artifact handoff

New campaigns automatically create `*-summary.json` and `*-summary.md`. Send the Markdown summary by default; retain the full JSON only as the auditable raw artifact. Existing full artifacts can be summarized without an accelerator:

```python
from scripts.summarize_experiment_json import main as summarize

summarize(["/kaggle/working/output/extent-long-horizon-campaign.json"])
```

## EXP-049: extended-horizon scaling campaign

Run as one cloud `Save Version` job on a fresh TPU v5e-8 session. Expected duration is approximately 6--7 hours. It performs the 2,048-versus-8,192-step comparison sequentially for layers 18 and 0, so only one large activation cache occupies working disk at a time.

```python
from scripts.qwen_extended_horizon_campaign import main as run_extended_horizon

result = run_extended_horizon([])
print("EXP-049 numerical pass:", result["passed"])
print("EXP-049 scientific gate:", result["scientific_gate_passed"])
print("summary: /kaggle/working/output/extent-extended-horizon-campaign-summary.md")
```

For handoff, download only `extent-extended-horizon-campaign-summary.md` and optionally its small `-summary.json` companion. Keep `extent-extended-horizon-campaign.json` in Kaggle output for audit; do not download NPY caches or endpoint payloads. On failure, retrieve `extent-extended-horizon-campaign-failure.json` and `exp049-campaign-stage-manifest.json`.

## EXP-050: depth-scaling atlas

Run as one cloud `Save Version` job on a fresh TPU v5e-8 session. The measured-work estimate is approximately 6.5 hours: 2,048-versus-8,192-step comparisons at layers 6, 12, and 29, with three seeds and all three objective arms. Each layer is completed and pruned before the next one.

```python
from scripts.qwen_depth_scaling_atlas_campaign import main as run_depth_atlas

result = run_depth_atlas([])
print("EXP-050 numerical pass:", result["passed"])
print("EXP-050 depth-trend gate:", result["scientific_gate_passed"])
print("summary: /kaggle/working/output/extent-depth-scaling-atlas-summary.md")
```

For normal handoff, download only `extent-depth-scaling-atlas-summary.md`. Keep the full `extent-depth-scaling-atlas.json` in Kaggle output for audit. On failure retrieve `extent-depth-scaling-atlas-failure.json` and `exp050-campaign-stage-manifest.json`.

## EXP-051: progressive 2/4/8-layer composition

Run as one cloud `Save Version` job on a fresh TPU v5e-8 session. The campaign trains eight standalone Mamba endpoints with the frozen asymmetric schedule (8,192 steps for layer 0 and 2,048 for all other layers), then evaluates nested 2/4/8-layer JOINT compositions. The expected target is 5--8 hours; measured time may vary with Kaggle compilation and downloads.

```python
from scripts.qwen_progressive_composition_campaign import main as run_progressive

result = run_progressive([])
print("EXP-051 numerical pass:", result["passed"])
print("EXP-051 eight-layer gate:", result["scientific_gate_passed"])
print("summary: /kaggle/working/output/extent-progressive-composition-campaign-summary.md")
```

Use this single call; do not run `pytest` or initialize JAX in another process after the notebook has claimed the TPU. Start and final Telegram messages are enabled by default. A failed scientific gate still finishes normally and writes the summary.

For handoff, download only `extent-progressive-composition-campaign-summary.md`. Keep the full campaign JSON in Kaggle Output only if convenient; the runner deletes multi-gigabyte caches and endpoint payloads after the final measurements. On a caught failure retrieve `extent-progressive-composition-failure.json` and `exp051-campaign-stage-manifest.json`. In an interactive session, rerun the identical `run_progressive([])` call to reuse completed layer boundaries.

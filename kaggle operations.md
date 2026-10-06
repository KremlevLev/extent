# Kaggle Operations Runbook

This file is operational documentation, not part of the scientific report or
paper. Current user budget (2026-10-03): **8 hours per TPU v5e-8 session** for
EXP-098--100. The older experiment entries below are historical operations.

## Session timing

- New EXP-098--100 runner cap is 8.0 hours including a 45-minute saving reserve.
- Training stops by 7 h 15 min from runner entry; startup/data/JIT consume that
  budget too. Earlier setup cells are outside this timer.
- Preserve the fixed scientific endpoint when partial; rerun the same cell to
  resume rather than changing step count or restarting Adam.
- Use existing clone/install/secrets cells, with the updated main checkout.
  No JAX subprocess after the notebook has acquired the TPU.
- Ready account-specific final cells and independent HF paths:
  [EXP-098--100 protocol](results/EXP-098-100-three-account-protocol.md).

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

## EXP-052: 8-to-16-layer boundary scaling

Run this as one Kaggle `Save Version` cloud job on a fresh TPU v5e-8 session. Do not run `pytest`, `jax.devices()`, or a second Python process first. The campaign trains 16 independent endpoints, evaluates full 8/16 compositions and layer-0/internal-only counterfactuals, and is sized for approximately 5--7 hours.

```python
from scripts.qwen_boundary_scaling_campaign import main as run_boundary_scaling

result = run_boundary_scaling([])
print("EXP-052 numerical pass:", result["passed"])
print(
    "EXP-052 composition gate:",
    result["aggregate"]["composition_scaling_gate_passed"],
)
print(
    "EXP-052 layer-0 mechanism gate:",
    result["aggregate"]["boundary_mechanism_gate_passed"],
)
print(
    "summary: /kaggle/working/output/"
    "extent-boundary-scaling-campaign-summary.md"
)
```

Telegram start, completion, and caught-failure notifications are enabled. A false scientific gate is a normal completed run. For handoff, download only `extent-boundary-scaling-campaign-summary.md`; do not download checkpoints or caches. On a caught failure retrieve `extent-boundary-scaling-failure.json` and `exp052-campaign-stage-manifest.json`. If the same interactive session remains alive, rerun the identical `run_boundary_scaling([])` call to reuse completed layer boundaries.

## EXP-053: composition-onset localization

Run as one Kaggle `Save Version` cloud job on a fresh TPU v5e-8 session. It regenerates the same 16 frozen endpoints, then evaluates 46 paired branches: nested 8/10/12/14/16 compositions, each new layer individually over base-8, and matched early/balanced/late 12-layer layouts. Expected runtime is approximately 7--8 hours.

```python
from scripts.qwen_composition_onset_campaign import main as run_onset

result = run_onset([])
print("EXP-053 numerical pass:", result["passed"])
print(
    "EXP-053 onset gate:",
    result["aggregate"]["onset_localization_gate_passed"],
)
print(
    "EXP-053 attribution gate:",
    result["aggregate"]["first_order_attribution_gate_passed"],
)
print(
    "earliest onset:",
    result["aggregate"]["incremental_onset"][
        "earliest_detected_upper_count"
    ],
)
print(
    "context-sensitive layers:",
    result["aggregate"]["single_addition_attribution"][
        "context_sensitive_layers"
    ],
)
print(
    "summary: /kaggle/working/output/"
    "extent-composition-onset-campaign-summary.md"
)
```

Use this single call after clone/install. Do not run `pytest`, `jax.devices()`, or another Python process before it. Telegram start/completion/failure messages are enabled, and false gates still produce a successful completed artifact. Download only `extent-composition-onset-campaign-summary.md`. On a caught error retrieve `extent-composition-onset-failure.json` and `exp053-campaign-stage-manifest.json`; do not download endpoint payloads.

## EXP-054: four-hour bridge initialization screen

EXP-053 is deferred when fewer than eight TPU hours remain. On a fresh Kaggle TPU v5e-8 `Save Version` job, run only this cell after cloning the current commit and installing `requirements-tpu.txt`. Do not call `jax.devices()`, `pytest`, `%run`, or start another Python process first.

```python
from scripts.qwen_bridge_ablation_campaign import main as run_bridge_screen

result = run_bridge_screen([
    "--max-wall-hours", "3.5",
])
print("EXP-054 status:", result["status"])
print("EXP-054 numerical pass:", result["passed"])
print("EXP-054 complete:", result["complete"])
print(
    "EXP-054 screening gate:",
    result["aggregate"]["screening_gate_passed"],
)
print(
    "summary: /kaggle/working/output/"
    "extent-bridge-ablation-campaign-summary.md"
)
```

The runner prioritizes layer 18, then layer 0; within each seed it records random, Apple bridge, combined bridge+orientation, MOHAWK orientation, and direct-QKVO controls. It writes a partial JSON after every completed arm and stops cleanly at the 3.5-hour deadline. Telegram reports start, normal completion, deadline-partial completion, or a caught failure.

Download only `extent-bridge-ablation-campaign-summary.md`. If the screen ends at the deadline, also download `extent-bridge-ablation-campaign.json`; the completed arms remain scientifically usable and `complete=false` prevents accidental treatment as the full protocol. On failure download `extent-bridge-ablation-campaign-failure.json` and `exp054-campaign-stage-manifest.json`. Do not download NPY caches or Qwen shards.

## EXP-055: stabilized partial-RoPE bridge confirmation

Run this as the first JAX use in a fresh Kaggle TPU v5e-8 `Save Version` process after cloning the current commit and installing `requirements-tpu.txt`. Do not run EXP-054 again and do not call `jax.devices()`, `pytest`, `%run`, or another Python process first.

```python
from scripts.qwen_stabilized_bridge_campaign import main as run_stabilized_bridge

result = run_stabilized_bridge([])
print("EXP-055 status:", result["status"])
print("EXP-055 numerical pass:", result["passed"])
print("EXP-055 complete:", result["complete"])
print("EXP-055 primary arm:", result["aggregate"]["primary_arm"])
print("EXP-055 primary gate:", result["aggregate"]["screening_gate_passed"])
print(
    "summary: /kaggle/working/output/"
    "extent-stabilized-bridge-campaign-summary.md"
)
```

The locked runner uses partial RoPE `0.5`, cosine-only bridge fitting, 256 initializer updates, and 4,096 matched recovery steps per arm. It has a 3.4-hour internal deadline and sends Telegram start/final/failure notifications. Download both `extent-stabilized-bridge-campaign-summary.md` and the small `extent-stabilized-bridge-campaign.json`; the full JSON is required to audit bridge stability. On failure retrieve `exp055-campaign-failure.json` and `exp055-campaign-stage-manifest.json`. Do not download activation arrays or Qwen shards.

## EXP-056: operator-preserving MIMO lift campaign

Use a fresh Kaggle TPU v5e-8 `Save Version` job. After cloning the current commit and installing `requirements-tpu.txt`, make this the first code that touches JAX. Do not run `pytest`, `jax.devices()`, `%run`, or an earlier experiment first.

```python
from scripts.qwen_mimo_lift_campaign import main as run_mimo_lift

result = run_mimo_lift([])
print("EXP-056 status:", result["status"])
print("EXP-056 numerical pass:", result["passed"])
print("EXP-056 complete:", result["complete"])
print("EXP-056 screening gate:", result["aggregate"]["screening_gate_passed"])
print("summary: /kaggle/working/output/extent-mimo-lift-campaign-summary.md")
```

The hard budget is 7.5 hours. The runner evaluates layers 18, 6, 29, then 0 and saves after every arm, so a deadline-partial run remains auditable. Telegram reports start, completion/deadline, or failure.

Download `extent-mimo-lift-campaign-summary.md` and `extent-mimo-lift-campaign.json`. On failure download `exp056-campaign-failure.json` and `exp056-campaign-stage-manifest.json`. Do not download Qwen shards, NPY caches, or checkpoints.

## EXP-057: two-hour locked exact-lift confirmation

In a fresh TPU v5e-8 `Save Version` process, after clone/install, run this as the first JAX call:

```python
from scripts.qwen_exact_lift_confirmation_campaign import main as run_confirmation

result = run_confirmation([])
print("EXP-057 status:", result["status"])
print("EXP-057 complete:", result["complete"])
print("EXP-057 gate:", result["aggregate"]["confirmation_gate_passed"])
print("summary: /kaggle/working/output/extent-exact-lift-confirmation-summary.md")
```

Do not run `pytest`, `jax.devices()`, `%run`, or another experiment first. Download `extent-exact-lift-confirmation-summary.md` and `extent-exact-lift-confirmation.json`. On failure download `exp057-failure.json` and `exp057-stage-manifest.json`. Do not download caches or weights.

## EXP-058: one-hour exact-lift composition pilot

In a fresh TPU v5e-8 process, after clone/install, immediately run:

```python
from scripts.qwen_exact_lift_composition_pilot import main as run_composition_pilot

result = run_composition_pilot([])
print("EXP-058 numerical pass:", result["passed"])
print("EXP-058 pilot gate:", result["aggregate"]["pilot_gate_passed"])
print("exact minus random NLL:", result["aggregate"]["exact_minus_random_nll"])
print("summary: /kaggle/working/output/extent-exact-lift-composition-pilot-summary.md")
```

Do not run any other JAX command first. Download `extent-exact-lift-composition-pilot-summary.md` and `extent-exact-lift-composition-pilot.json`. On failure download `exp058-failure.json` and `exp058-stage-manifest.json`. Endpoint checkpoints are only needed for debugging a failed streamed stage; do not download them after normal completion.

## EXP-059: multiseed composition confirmation

Run as the first JAX call in a fresh TPU v5e-8 process:

```python
from scripts.qwen_exact_lift_composition_confirmation import main as run_confirmation

result = run_confirmation([])
print("EXP-059 pass:", result["passed"])
print("EXP-059 gate:", result["aggregate"]["scientific_gate_passed"])
print("mean exact-random NLL:", result["aggregate"]["mean_exact_minus_random_nll"])
print("95% CI:", result["aggregate"]["bootstrap_95_ci"])
print("summary: /kaggle/working/output/extent-exact-lift-composition-confirmation-summary.md")
```

Expected runtime is inferred from EXP-058, not guaranteed. Download `extent-exact-lift-composition-confirmation-summary.md` and `extent-exact-lift-composition-confirmation.json`. On failure download `exp059-failure.json` and `exp059-stage-manifest.json`.

## EXP-060: 5–8-hour progressive exact-lift scaling

Use a fresh TPU v5e-8 `Save Version` process and run this as the first JAX call:

```python
from scripts.qwen_exact_lift_scaling_campaign import main as run_scaling

result = run_scaling([])
print("EXP-060 pass:", result["passed"])
print("EXP-060 gate:", result["aggregate"]["scientific_gate_passed"])
for stage in result["aggregate"]["stages"]:
    print(stage["replacement_count"], stage["mean_exact_minus_random_nll"], stage["bootstrap_95_ci"])
print("summary: /kaggle/working/output/extent-exact-lift-scaling-campaign-summary.md")
```

The campaign resumes at completed layer boundaries when the same Kaggle output is still present. Download only `extent-exact-lift-scaling-campaign-summary.md` and `extent-exact-lift-scaling-campaign.json`. On failure download `exp060-failure.json` and `exp060-stage-manifest.json`; endpoint checkpoints are large and only useful for resuming/debugging.


## EXP-101--103: plateau continuation, 8h/session

Use existing notebook preparation/secrets, fresh main, one final import/main cell
from `results/EXP-101-103-three-account-protocol.md`. Source adapters are pinned
EXP098 final16384 protected32, both seeds; never substitute newer/best states.
Per-invocation8h includes45min saving reserve, but preceding notebook setup does
not count: deduct it with `--max-wall-hours`. Same cell resumes partial compatible
new-stage states, including new Adam moments. Send summary AND full JSON.

## EXP-104: paper-inspired full-decoder CE, 8h/session

See `results/EXP-104-kaggle-launch.md`. Keep existing setup/secrets and import
`main` from `scripts.m3q_paper_decoder_recovery_campaign` in the notebook process.
Fixed098 warm starts, two strategies/two seeds,2048 steps each; decoder embeddings
and vocabulary readout fixed, internal FP32 masters and sharded AdamW states.
Eight-hour invocation includes1.5h saving reserve; deduct preceding setup time.
One dense checkpoint is~16.4GB, chunked; final completion within one session is
NOT guaranteed. Pending uploads do not establish remote durability. Same cell
continues the last durable compatible state; gate unavailable for partial runs.
Send both `extent-m3q-paper-decoder-recovery-summary.txt` and corresponding JSON.
Only CPU and virtual-device checks completed locally; real TPU memory/throughput
and upload timing are runtime checks, not prior experimental evidence.

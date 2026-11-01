# EXP170: clean-start audit, 1 November 2026

Code/provenance findings, not a completed 100M-token experiment. No accelerator
run started and no historical scientific files changed.

## Current lineage

`extent/t4_patient.py:113-126` reconstructs EXP069 preparation and EXP072 ONPOLICY
endpoints. Subsequent adaptation changes weights, not the original provenance.

`scripts/m3q_allocation_campaign.py:153` initializes Mamba using random `mamba.init`.
It then fits teacher targets with 1024 dual updates, calibrates the readout, and
runs 2048 recurrent decoder-aware updates. No Q/K/V tensor transplant occurs in
that function. Verified at the received EXP069 revision
`a385256db8cd34dd09b6dc54458c54cec54a9911`, not just today's checkout. The original
EXP069 protocol explicitly documents random initialization. This was a historical
recipe choice, not newly established file corruption or an importer malfunction.
Preparation context was 64. EXP072 subsequently uses 4096 updates/layer on 512
cached windows of 128 tokens: 65,536 distinct input positions.

Therefore, current recovery is genuine recovery of a prepared hybrid, but does
not establish what clean direct head-preserving initialization would achieve.
Historical exact-lift experiments are separate. An exact SISO-to-MIMO lift does
not certify equality to original softmax attention, and is not the initializer
of the bank reconstructed by current Patient.

## Separate geometric issue in the historical lift family

`extent/mamba3_transplant.py:57-75` averages source heads, then `fit_matrix`
tiles/crops/rescales features. Balanced lift uses these pooled projections.
Distinct heads can cancel BEFORE the operator-preserving lift: a synthetic
pair W,-W maps to zero. New tests verify this mathematical counterexample.
This is not a measured loss percentage on actual Qwen weights, nor the claimed
initializer of the random EXP069 bank.

## Inspected arianraje recipe

[Published source](https://huggingface.co/arianraje/qwen3-4b-mamba3-hybrid-stage2b-kd-bias1/blob/b946069890837bbe9b5dd75e2d118a7e98d2d193/modeling_qwen3_mamba3.py),
revision b946069890837bbe9b5dd75e2d118a7e98d2d193, Apache-2.0 metadata.
SHA256: 6ca2cee4665ac88b28cb81470179f635287e983a28c4079afe2b085ceaf55f77.
External Python was read, not executed.

His 4B geometry: 32 query heads, 8 KV heads, P/N128, G8, R4. Copy V to matched
value heads, K to its KV group in each rank, Q to the corresponding rank, and
copy output projection and Q/K norms. One-hot mimo_o selects the own query rank.
Random-simplex mimo_x; zero token-dependent dt/A/trap/angle rows; log-uniform dt
.001-.1; D/mimo_z/readout norm ones. Bias1 is the selected published default.
B/C normalization uses teacher epsilon (1e-6 by default); readout epsilon1e-5.
Angles are tanh-bounded. All mixer weights train, including zero dynamic rows.

[Stage1](https://huggingface.co/arianraje/qwen3-4b-mamba3-hybrid-stage1-align-bias0)
is independent teacher-prefix full-layer hidden-output alignment, 100M tokens
at context4096, LR1e-3, 1526 updates. It is not our sequential ONPOLICY procedure.
Reported final bias1 PPL25.35 gives NLL3.23277873 on his WikiText2 protocol.
Published bias1 weights are reconstructed at step1500, 26 steps before that
reported final endpoint; their PPL25.05 is a different checkpoint.

Gate Z inheritance cannot be recovered from the modeling class constructor alone:
surgery is external. Sampled public init-checkpoint byte ranges, layer0 in_proj
rows0,1024,4095, show Z differs from the corresponding X rows. This does not
identify Z's donor. Our prototype's random Z is an explicit experimental choice,
not an exact reconstruction of all surgery details. Do not apply his reported
99.95% inheritance figure to our adaptation. Only small binary ranges were
compared in memory; no full competitor checkpoint was downloaded.

## Clean Qwen1.7B adaptation

| Field | Current config | Proposed head-preserving config |
|---|---:|---:|
| Inner width |1536|2048|
| Value heads |24|16, matching teacher queries|
| Head/state dimensions |64/64|128/128|
| B/C groups |1|8, matching teacher KV groups|
| MIMO rank |4|4|
| Retained attention |4/28|evaluate4/28 and7/28|

Qwen1.7B has TWO query heads per KV group, versus FOUR in his4B. Our strict
initializer duplicates these two queries across four C slots and selects the
correct own-head slot. It preserves all original projection channels without
head averaging or cropping. Geometry/retention increase capacity and cost;
improvement of this package alone would not isolate initialization. A random
control in the SAME new geometry is required for an initializer comparison.

## Implemented evidence and limits

`extent/clean_transplant_170.py`: strict geometry, Q/K/V/O/norm mapping, clean
initialization, full-strength normalized/tanh recurrence, and frozen-Qwen
full-block alignment loss. No old Patient/adapters/Adam are accepted. The chosen
per-layer target-RMS loss reduction is explicit, not a recovered unpublished
trainer detail.

Five CPU tests pass: projection/norm ownership including two-query/four-rank
adaptation; incompatible geometry rejection; pooled-head cancellation example;
independent NumPy recurrence at zero rotation; real block gradients with live
dt/angle and unchanged frozen teacher weights. This is not full-model NLL,
long-training, fused-kernel/BF16 parity or physical throughput evidence.
No 100M-token runner is implemented yet.

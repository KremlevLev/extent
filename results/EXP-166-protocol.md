# EXP166: whole-batch layer-streamed FULL-KD

Prepared corrective execution experiment, no physical Blackwell result yet.

## Evidence and target

Both received EXP165 invocations failed before any accepted update. The larger-memory invocation still rejected all physical batches1..8 (192s scan). Source CURRENT remains157685, Adam2833, Edu next window16993, pending_slots0. These are engineering failures, not a model-quality plateau result. The previous monolithic executable retained a large reverse-mode workspace; the report discarded detailed allocation exceptions. Raising the pool alone did not make that path usable. Its exact per-trial peak remains unmeasured in the received JSON.

EXP166 retains all1,364,173,952 decoder coordinates, inherited q/Adam/schedule, same .5CE+.5forward-KL, T1, full-scope LRs, clipping and sequential fresh Edu. Embedding/finalnorm/head stay frozen as in165. Only execution and capacity selection change. Same diagnostic A/B/C/D/E, EN/RU16 progress and128 final; no untouched-test or benchmark-capability claim.

## Execution and mandatory planner

Use the existing verified MethodReference full-decoder chain rule on the WHOLE physical batch at once. Forward/backward computations are compiled per layer; intermediate hidden states are staged losslessly in host RAM between layers. The vocabulary head is processed in sequence tiles. Frozen teacher weights remain GPU resident, BEST remains host resident. Gradients across all layers produce ONE Adam update per whole batch. This is not the old six separate one-example forwards/backwards with mean-gradient accumulation used as a slow fallback in163.

The planner measures batches1..8 in this declared streamed mode, ascending, on the actual CURRENT/Adam and same next Edu windows. It excludes compilation/warmup from two synchronized timings and selects maximum finite tokens/s. Proposals do not advance q, optimizer or data. The selected whole-batch gradient is checked against the fixed-q mean of single-example gradients using the existing numerical tolerances. If none fit, stop and retain full error messages and per-trial allocator/NVML snapshots. No monolithic whole-model gradient executable is invoked; MethodStep.accumulated is used only as the numerical reference, not as the selected training engine.

After each accepted update, q/Adam return to planned NamedSharding; data cursor advances by actual batch, global/Adam by1. On resume, the mandatory planner repeats. Batch change is an explicit optimization intervention; compare trajectories by accepted tokens, not nominal step count.

## Source, storage and upload

Freeze unchanged durable165 CURRENT157685. NLL source replay is mandatory using165's actual per-window source controls (its0 accepted updates make these controls its CURRENT). Source manifest is a separate immutable file checked by SHA. Base161 manifest descriptor is compact and expanded in RAM only for reconstruction. Full ancestry manifests are not recursively embedded in new reports.

Bootstrap restores the already durable source without creating another22GB state generation. If an invocation accepts0 updates, no full-state save/upload is performed; an already pending upload may still be retried. Once training advances, atomic/pruned local CURRENT is saved immediately then every30min and on exit. Full HF payload only at exit, metrics during work. Per-session32GB upload cap. New namespace experiments/exp166-full-stream-v1, states/runtime-full-stream, results/runtime-full-stream.json. Historical165 sources remain unchanged.

Default work allowance130GB accounts for retained donor states (~85GB) plus new CURRENT (~22GB) and atomic replacement staging (~22GB); actual limit checks and minimum20GB disk free remain. Only one generation per active slot is retained. No donor checkpoint deletion.

`--leave-free-gib5` queries current physical GPU total/used memory before importing JAX, subtracts5GiB reserve and2GiB CUDA overhead allowance, then configures the BFC pool fraction. This is a startup budget, not a guarantee against other processes subsequently growing. Without this option, manual environment memory settings remain available. Other GPU processes are never stopped or altered.

## Verification

31 local checks passed across targeted suites: real reduced whole-batch versus microgradient comparison, two real optimizer steps, actual planner numerical audit/nonmutation, all8 selection/error logging, source checkpoint restore, no-update checkpoint reuse, compact manifests and no contract mutation, explicit budget calculation, wrapper routing and bot regressions. Physical Blackwell throughput, workspace and long training stability remain unverified until the user's run. Existing Telegram bot registration is automatic, with one read-only listener and start/finish notifications only.

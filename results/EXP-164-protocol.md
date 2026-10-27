# EXP-164: FULL-KD continuation with mandatory capacity scheduling

Prepared experiment, no Blackwell result yet. Single seed123 exploratory continuation.

## Observation and intervention

EXP163 durable FULL-KD CURRENT157684 has2832 accepted fresh-Adam updates and has consumed16992 FineWeb-Edu windows (4,349,952 input positions). Its last shared quality measurement is earlier, at156900; CURRENT quality must be measured before new updates. All three EXP163 methods fell back from a whole batch6 to six streamed microgradients, yielding only about291-315 input tokens/s. A memory rejection must trigger a capacity search rather than silently committing the session to this slow path.

Continue ONLY the full decoder (1,364,173,952 trainable coordinates; embeddings and final norm/head stay frozen). Retain EXP163 q, Adam moments/count2832, LR schedule, BEST and next unused Edu window16992. Objective remains0.5CE+0.5forward-KL, T1, sequence256, stable global clipping1, same full-scope LRs. No rank branch, no upper-MLP branch, no optimizer reset.

## Mandatory planner

Each invocation, after exact state restore and CURRENT baseline measurement, tests every physical batch size1..8, largest first. Each uses the same real CURRENT/Adam and consecutive Edu windows at the saved cursor. Compilation/warmup is separate; two synchronized complete proposal samples determine median input tokens/s. Proposals are discarded, so q, moments, Adam count and data cursor do not advance. Executable caches are released between trial shapes. Memory failures are recorded and the scan continues; other errors stop preflight.

Select maximum measured tokens/s among finite candidates. Audit the selected actual-batch gradient against the mean of fixed-q streamed single-example gradients, with one Adam proposal and the inherited CE/gradient tolerances. Refuse training if none fit or if the audit fails. No silent six-microstep fallback. Store all trials, selected batch, timings, numerical audit and probe input presentations. The eight compilations can take several minutes; there is no hard compilation-time guarantee.

Changing the physical batch changes effective batch and optimization trajectory. This is an explicit new EXP164 intervention, not a numerically identical EXP163 continuation. Repeat the scan on each resume; selected batch may change with available memory. Every accepted step advances global/Adam by1 and data cursor by the selected batch. The cursor is a checkpoint leaf; use explicit accepted input tokens, never `(step-source)*6`. Training reads each new packed window once; probes reuse the next windows without consuming them. Stop at data exhaustion, requested STOP or wall deadline, not an8192-step target.

## Diagnostics and durability

Source controls and progress use the same diagnostic A/B/C/D/E panels, EN/RU16 during progress and128 at exit. These inspected panels are validation, not untouched confirmation or benchmark capability evidence. C4 D selects BEST; CURRENT is independently retained. Prioritize CURRENT final diagnostics before the22GB exit upload; leave10min of the18min reserve for synchronization. Incomplete diagnostics are explicitly partial.

Pruned atomic local state every30min and on stop; full HF state only at exit; reports during work. Separate `states/runtime-full`, `results/runtime-full.json`, HF `experiments/exp164-full-adaptive-v1`. Shared existing token packs, source states and compilation cache; no bulk new corpus. Default32GB upload allowance,80GB work cap,20GB minimum free. A full q/Adam/BEST checkpoint is about21.83GB, so repeated soft pauses incur large uploads. Do not delete source states manually.

## Automatic bot

`engine_full` registers `runtime-full.json` and ensures one read-only listener for the existing token. A shared per-bot registry retains other work directories, updates the same work alias and reloads on each command. A known legacy engine_status listener may be gracefully replaced only after exact module/token verification; unknown lock owners are never killed and never cause a duplicate poller. No token appears in logs. Training start/finish notifications only; `/check` is on demand. Listener stays alive after training exits so the final status remains available. No mutation commands.

## Interpretation

The immediate question is whether FULL-KD continues improving held-out panels on more fresh Edu tokens at usable Blackwell throughput. It is not a seed replication, isolated capacity ablation, pure corpus comparison or proof that NLL implies usable instruction following. CPU tests establish wiring/math on a reduced model; actual96GB Blackwell throughput and memory remain to be measured by the mandatory planner.

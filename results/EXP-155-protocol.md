# EXP-155 вЂ” TPU continuation of trained EXP-153 v2 rank256

Prepared 22 October 2026. One restored Kaggle account, one TPU v5e8 session,
9h available. Execute in the existing TPU notebook, one process/eight devices.
This is a continuation, not a new reset/rank screen. No cloud run launched here.

## Immutable source and clocks

Source: `experiments/exp153-rank-growth-v2/worker-1/checkpoints/GROW-R256`,
HF dataset `levkremlev888/extent`, revision
`db3ae61fc99cb3baa10d6f7236f9895654527dc2`.
Pin: [EXP-155-source-pin.json](EXP-155-source-pin.json).
Manifest SHA: `fe0b63d43df72f04b7d3aec3535c04d8363802115fb652dc948edf79b0ad8864`.

- CURRENT global data cursor **37910**, six Adam/schedule counters **6784**.
- Inherited A-selected BEST **37270**: A/B/C **5.575060/5.611766/5.149767**.
- Preserve CURRENT factors, all trained extra coordinates, pregate, Adam moments,
  schedule counts, best weights and best cursor. No reset, warmup restart,
  additional folding, new capacity, change of rates or new source data.
- NEW = globalв€’37910; Adam = 6784+NEW. R256 means retained32+extra224,
  not MIMO rank. Source MIMO rank remains4.
- Reconstruction of the existing frozen base uses the original136 fold ONCE,
  exactly as validated145/147/153v2. No further147/153 correction folding.

Source report still says `running`. CURRENT37910 has no source NLL evaluation.
Do not label BEST37270 metrics as CURRENT37910 metrics. Replay inherited BEST
against its pinned160 per-window values at tolerance.03 BEFORE training, persist
actual/reference/max/worst values before rejection. Measure CURRENT separately
on all A/B/C. Keep inherited BEST eligible throughout own continuation.

## Execution and scientific gates

New isolated files `extent/rank_continue_155.py` and
`scripts/m3q_rank_continue_155_campaign.py`. Original source runners unchanged.
Restore all871 source chunks with SHA verification and exact optimizer tree.
Require source code hashes, JAX/jaxlib0.11.0,flax0.12.8,optax0.2.8,
matmul precisionNone/PRNGthreefry2x32 and identical FIT/A/B/C metadata.

First actual TPU proposal compares compiled full-chain gradients against
growth-aware streamed full-chain gradients on identical source/state/batch:
relativeL2<=.15,cosine>=.98,CE discrepancy<=.03, finite proposal. Both paths
explicitly use the extra224 coordinates. The legacy streamed factory ignores
block_fn, so fallback MUST be GrowthStreamedStep, not that factory.
If compiled parity fails or its memory guard rejects, use the finite reference
streamed proposal on the SAME8TPU and record the choice. Never loosen gates or
accept a rejected compiled proposal. Charge extra audit input presentations.

Cross-hardware continuation is not bitwise trajectory equivalence. No claim of
25вЂ“30M tokens/session until throughput is measured. Original rates: Mamba3e-4,
normalization1e-4, stable global clip1, Adam.9/.999eps1e-8, batch1/seq256.
Same cyclic FIT permutation with global cursor; no new token offset.

## Budget, persistence, outputs

Runner **8.7h total including45min save reserve**, leaving18min for notebook
setup within9h. Work budget7.95h includes source downloads, model load, JIT,
replay, training, intermediate evaluation and saves. Ceiling262144 NEW updates
is deliberately high, not a promise or stop after8192. Registered work deadline
normally produces `deadline_partial`; final durability is checked separately.

Save locally every128 new updates, remotely every120min plus finalization;
current+Adam+selectedbest only. New HF prefix `experiments/exp155-rank-continue-v1`;
original153 untouched. Upload cap8GB/invocation, no frozen-model uploads.
HF429/network retries use existing deadline-aware policy. Notifications:
start, at most one progress/4h, one failure, finish; maximum5.
Repeat the same cell/contract to resume OWN155, not restart153.

Download one report: **extent-m3q-rank-continue-155.json**.
Inspect source_replay, adam_count_at_restore, first execution_checks, NEW
tokens, A/B/C registered and last endpoints, durable_step and pending_slots.
`storage_partial`/pending slots are not confirmed remote durability.

## Validation and limits

[EXP-155-source-audit.json](EXP-155-source-audit.json): actual source full
current+Adam+best restored locally from backup, all871 chunks checked;
57,200,640 parameter elements, all six optimizer counts6784, selectedbest37270,
30,277,632 nonzero added-B coefficients. Not a full1.7B forward/TPU proof.

Five CPU tests pass, including real trained growth/optimizer restoration,
compiled-versus-streamed gradient audit and forced fallback, actual runner's
first accepted update/deadline save, rejected replay persistence before any
update, dispatch/immutable-source-code hashes. CPU tests do not establish TPU
memory use or speed; physical source replay/gradient gates remain mandatory.
One preparation seed, repeatedly exposed validation panels. No new rank
comparison, general quality recovery, fresh-test, or equal-total-cost claim.

Eight virtual CPU devices also passed the actual1x4x2 distributed tiny BF16
R256 update: gradient relativeL2=.010437,cosine=.999946,CEerror=.000154;
compiled proposal selected; moments/counts preserved. See
[EXP-155-eight-device-audit.json](EXP-155-eight-device-audit.json).
XLA reported resharding/rematerialization warnings; speed/memory on physical
TPU remain unmeasured. This audit is not a physical TPU result.

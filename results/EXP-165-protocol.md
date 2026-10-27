# EXP165: steady-memory continuation after EXP164 engineering failure

Prepared corrective engineering continuation; no new GPU result yet.

## Received164 evidence

EXP164 failed on the second attempted optimizer update with a MemoryCheckedJit HBM guard, after one accepted256-token update. CURRENT157685, Adam2833 and next unused Edu window16993 are durable; pending_slots is empty. It ran0.6045h including preparation and a21.9GB upload. No new learning-quality conclusion is supported by one update. The planner tried all8 shapes in533.2s; batches2..8 were rejected for memory, batch1 measured367tokens/s and passed the gradient gate (relativeL2.02444, cosine.999708, CE error.001863).

The initial fixed-q probe did not reserve the extra CURRENT-versus-BEST weight copy that appears when an aliased BEST remains at the source and CURRENT changes. The second memory check reported28,757,183,232bytes in use plus21,826,797,900 output and8,277,039,104 temporary bytes:58,861,020,236 versus56,086,233,088 limit. The loop also installed raw executable outputs instead of restoring planned NamedSharding, creating a new compilation key on the next step. Previous tests covered one real gradient update and a mocked multi-step controller; they missed this live memory/layout transition.

## Correction

EXP165 freezes the exact durable EXP164 CURRENT157685 as its parent, verifies code and chunk identities, carries Adam2833 and Edu16993, and uses its own HF prefix experiments/exp165-full-steady-v1 and states/runtime-full-next. Old163/164 scientific sources remain byte-identical. The single FULL-KD recipe, frozen head/embedding, all LRs/objective and no-repeat data contract are preserved.

BEST lives as exact NumPy arrays in host RAM before profiling and after every selection. It is included in atomic local/HF checkpoint payloads as before, but no longer consumes a second5.4567GB GPU weight copy when CURRENT changes. Only CURRENT/Adam plus frozen base/teacher remain resident for training. Ordinary compiled proposals return q/Adam/health, dropping the extra5.4567GB gradient output. A separate gradient-only executable audits the selected batch against the streamed fixed-q reference with the same objective. It does not weaken the numerical gate.

The mandatory1..8 scan measures this actual training executable with host-retained BEST. Every accepted output is explicitly placed into its original NamedSharding, so consecutive updates reuse the planned executable key. No silent streamed-batch fallback. Stop before training if no candidate fits. Actual Blackwell fit/speed are still decided by preflight; CPU tests do not establish them.

All other164 timing, soft-stop, data cursor, CURRENT diagnosis-before-upload and one-listener Telegram behavior is retained. `engine_launch --runner full` now routes165; explicit full164 is available for historical use. New report results/runtime-full-next.json; log runtime-full-next.log. The source164 failed report remains runtime-full.json.

Default local steady-state allowance is100GB because historical donor states plus the new current state now occupy about85GB combined. Atomic replacement can transiently stage another~22GB; reports show~200GB free before this change. One pruned generation per slot; no automatic deletion of source checkpoints. GPU budget remains independently controlled by the same fraction. This local storage setting does not mean100GB of VRAM.

## What to test

Real reduced-model two-update test using NamedSharding, independent host BEST, only one cached executable, exact lean-proposal versus old-proposal q/Adam, separately compared gradients and exact Adam increments. Real chunked164-to165 fork preserves data and optimizer; own resume avoids donor. Frozen source tests accept the durable last accepted state, never the failed second proposal. No seed, corpus or quality superiority claim.

# 8 October2026: completed short horizons, pending durability

Supplied raw JSONs preserved in EXP-104-completed.json/EXP-105-completed.json.
Both status completed, four finite trajectories2048, gatesFALSE. Runner-session
durations1.0710358h/0.9703678h. The registered2048 horizon caused completion, not
the8h upper bound;2048x256 tokens per trajectory gives only2.097M input tokens
per campaign. Conservative feasibility scope significantly underused actual
runtime after the user's long Kaggle queue. Future long-session horizons/data/LR
must be registered based on this measured throughput to target6–6.5h useful work,
with saving reserve; do not silently append tokens to this completed protocol.

104 primary Wiki control advantage:-0.000446/+0.189027, own-start gain
-0.003384/+0.021388. 105:-0.002481/+0.192240, own-start gain-0.005419/+0.024602.
No paired0.1 recovery win. Seed456 comparative gain is largely control regression;
not evidence that either primary recovered teacher quality. More training remains
a hypothesis, not an explanation that guarantees success. Shared controls match.

Both JSONs list ALL4 final checkpoint slots pending. 104 records an earlier dense
step512 upload (257.49s), which does not prove final2048 durability. Earlier code
swallowed transient chunk-upload exception details, so the reason cannot be
identified from these files. New logs record safe error class/HTTP status/retry
delay, and explicit pending/durable flags; notifications separate training-done
from fully saved. Exact105-v1 contract archived for operational-fix compatibility;
no objective, data, horizon, optimizer or state reset. 104 guard explicitly permits
operational chunk-sync diagnostics changes while retaining training AST guards.

Telegram had not been wired into the new runner. It is now enabled by default
with existing secrets, reporting start, training-finished/stopped, failure and
post-sync outcome; send results recorded in JSON. No live Telegram message sent
locally. CPU mock notification/continuation/auth/rate-limit checks pass.

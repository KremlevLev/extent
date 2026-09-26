# EXP-091 — first session, conclusive registered-gate failure

Reviewed 2026-09-26 from the user-supplied full JSON and compact summary.
Run revision: `ca31f1ecae398965f8c039c9be35fa700e499f67`.
Status: `deadline_partial`; duration `7.787254` hours on eight TPU devices.
Three of four trajectories finished all 24,576 context-256 updates
(6,291,456 token presentations each). Seed-456 PLAIN stopped durably at
step 7,456. All 14 recorded checkpoint uploads succeeded; no failed upload
or non-finite evaluation appears in the supplied artifact.

| Seed | Arm | Step | Start NLL | Final/last NLL | Final/last excess NLL | Final/last KL |
|---:|---|---:|---:|---:|---:|---:|
| 123 | ONPOLICY-PLAIN | 24,576 | 10.688488 | 15.687073 | 13.040308 | 17.335600 |
| 123 | ONPOLICY-ANCHOR | 24,576 | 10.688488 | 15.764514 | 13.117749 | 18.741549 |
| 456 | ONPOLICY-ANCHOR | 24,576 | 10.866499 | 23.826136 | 21.179371 | 32.303601 |
| 456 | ONPOLICY-PLAIN | 7,456 | 10.866499 | 37.004325 | 34.357560 | 43.199371 |

The complete seed-123 paired endpoint is decisive for the pre-registered
conjunctive gate: ANCHOR minus PLAIN is `+0.077442` excess NLL and `+1.405950`
prediction KL, where positive is worse. It also loses in both metrics at
every nonzero shared registered checkpoint (`3,072`, `8,192`, `16,384`,
`24,576`). No possible seed-456 completion can make ANCHOR win at *both*
seeds. Therefore the registered gate is already impossible and the remaining
seed-456 PLAIN trajectory should not consume another TPU session solely to
close EXP-091. The incomplete seed-456 endpoints must **not** be compared as
an equal-step pair.

More importantly, both complete seed-123 branches worsen substantially from
the common warm start: PLAIN NLL `10.688488→15.687073`, ANCHOR
`10.688488→15.764514`. Seed-456 ANCHOR worsens `10.866499→23.826136`.
The frozen Qwen teacher NLL on the same validation windows is `2.646765`.
The trajectories are strongly non-monotonic (for example, seed-123 ANCHOR
excess NLL reaches `30.828080` at step 8,192 before falling to `13.117749`).
This is evidence against the tested fixed 0.03 post-Lion backbone-update
scale as a reliable protection recipe; it is not proof that all backbone
anchoring or all on-policy warm starts fail. BF16 rounding may make very
small copied-weight updates ineffective. The next method must directly
prevent loss of the step-zero assembled-model quality, not infer success
from local layer gains or a transient intermediate minimum.

Full user-supplied artifact SHA-256:
`2e505e6e2dfa4c074cda505c8b113b36b9569c268252412cabe2afae76c2ad13`.
Compact summary SHA-256:
`2e480d71cdfa854faeb5096a91e0cd4493bf81ed40cebf84d6973711e1fbf0e5`.

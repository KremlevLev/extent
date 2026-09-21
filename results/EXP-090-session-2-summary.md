# EXP-090 — session 2 partial result

Reviewed 2026-09-21 from the user-supplied full artifact, SHA-256 `cdb8a04480aa309b551400168ba161897fef88c5edb1f47f2945262152106543`.

- Status: `deadline_partial`; invocation duration `7.7897` hours.
- `seed 123 / RANDOM-DT-FROZEN`: partial at step 83,360 (21,340,160 tokens). The registered `dt` freeze ended at step 32,768, so 50,592 subsequent updates had trainable `dt`.
- All recorded metrics are finite. All 12 checkpoint upload events passed, including step 83,360.
- The ordinary `seed 123 / RANDOM` control is already durably saved in the EXP-090 HF namespace to step 83,936 from session 1. The control's registered curve exactly matches EXP-089 on all shared checkpoints; it is not an independent replication.

## Matched-step comparison

Negative differences favor early `dt` freezing.

| Step | Random excess NLL | Frozen-dt excess NLL | Frozen − random | Random KL | Frozen-dt KL | Frozen − random KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16.044501 | 16.044501 | 0.000000 | 22.114488 | 22.114488 | 0.000000 |
| 3,072 | 11.814042 | 12.996476 | +1.182434 | 17.189559 | 18.658984 | +1.469425 |
| 8,192 | 14.182362 | 11.400582 | −2.781781 | 19.986859 | 16.609838 | −3.377021 |
| 16,384 | 6.844993 | 13.042907 | +6.197914 | 11.040131 | 18.618632 | +7.578501 |
| 32,768 | 9.817048 | 9.913191 | +0.096143 | 14.495236 | 14.695620 | +0.200384 |
| 65,536 | 9.806951 | 9.495497 | −0.311454 | 14.564223 | 14.205716 | −0.358508 |
| 83,360 | 9.718625 | 9.785252 | +0.066628 | 14.512564 | 14.640417 | +0.127852 |

There is no stable advantage: the sign reverses repeatedly. The improvement at step 65,536 disappears by the matched step 83,360. The registered step-98,304 endpoint and second seed are still pending. Do not claim that freezing `dt` helps or hurts in general from these partial observations.

## Decision

Resume `RANDOM-DT-FROZEN` from the uploaded step-83,360 checkpoint. Finish the registered endpoint, then complete the existing control and the second-seed trajectories before adjudicating the gate. Do not launch another timestep schedule yet.

Run revision: `380c735b0223d5c84fbd1510bb8a533512dae303`.

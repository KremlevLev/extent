# EXP-078 — protected trust-ratio recovery (completed)

- Status: completed in 4.077832 hours at implementation revision `30eab659191959e5acaa09545742231e17061251`.
- Artifacts supplied by the user: full JSON SHA-256 `879f973fc0828f5ddb697fa5b3db173e9c2267ec7b09f919191c627eba12ccaf`; summary SHA-256 `4828e8a517ac7f1b7834e551ed34da44e3a21d212c019a765de89b9e69f88a68`.
- Registered gate: **FAIL**. Primary `TRUST-1E-4` worsens both seeds and violates the excursion bound.

| Seed | Arm | Step-0 NLL | Best NLL (step) | Final NLL | Final delta | Max/start |
|---:|---|---:|---:|---:|---:|---:|
| 123 | TRUST-1E-4 | 10.916678 | 10.916678 (0) | 13.101959 | +2.185281 | 1.372656x |
| 123 | TRUST-3E-5 | 10.916678 | 10.732380 (4096) | 12.044462 | +1.127785 | 1.191264x |
| 456 | TRUST-1E-4 | 9.692907 | 9.692907 (0) | 14.871074 | +5.178166 | 1.594798x |
| 456 | TRUST-3E-5 | 9.692907 | 9.341055 (8192) | 9.341055 | -0.351852 | 1.114758x |

## Interpretation

The primary relative step `1e-4` is too aggressive. The conservative `3e-5` arm is the first joint-recovery recipe to cross below its own starting NLL for both seeds at a registered checkpoint, though at different steps. Seed 123 improves by `0.184297` at step 4096 and then regresses; seed 456 improves by `0.351852` at step 8192. This is a candidate signal, not a passed result or a valid per-seed early-stopping claim.

The 32-window paired differences are weak: seed-123 step-4096 mean delta is about `-0.184` with paired SE `0.624` and 18/32 windows improved; seed-456 final mean delta is about `-0.352` with paired SE `0.266` and 11/32 windows improved. A few windows can drive the mean. The result therefore justifies a structured follow-up but not a recovery claim.

Because full-depth simultaneous trust-ratio training still depends strongly on seed and checkpoint, the next experiment isolates four six-Mamba segments naturally separated by retained attention layers. This tests whether one depth region causes the coupled instability and establishes an order for later staged recovery.

# EXP-114: first TPU diagnostic session, incomplete

Reviewed 14 October2026 from the supplied full JSON; exact source SHA and
derived B-panel values are in `EXP-114-session-1-analysis.json`.

Status failed after1.72734 runner hours.8338 tasks completed; profile, sweep,
FIT, temperature, readout and optimizer modules completed. Both readouts reached
4096 and both1024/4096 endpoints are recorded durable; pending[]. The original
EXP111 endpoint replay matches every window exactly for both seeds on all three
legacy panels. The final patient immutability hash check was not reached.

Failure: dynamics' first traced-scan output parity guard reports maxabs2.0 versus
the original Mamba mixer. No dynamics task was accepted, no context/reset/
precision or locked C measurements exist. This is diagnostic implementation
failure, not evidence of failed training or nonfinite patient weights.

## Measured preliminary findings (A selection, B verification only)

Best A-selected teacher-mixer replacements give B delta NLL-0.005999/-0.001976;
rollbacks give-0.006140/-0.002272. Both are below the registered practical0.1
signal and comparable to permitted hook background. The selected teacher pairs
give+0.004990/-0.002699; there is no substantial paired rescue on B.

Teacher boundary substitution after layer26 yields NLL3.193958 for both seeds,
against same-path baseline7.376453/7.329343. This bypasses nearly the entire
student prefix; it does not identify layer26 as the cause or a deployable fix.
The neighboring final boundary produces an almost identical teacher-level loss.

The trained output-only rank32 readout worsens B:7.376138->7.410157 and
7.329009->7.418798 (delta+0.034018/+0.089790). Temperature1.0 is selected on A;
tested global logit temperatures do not improve the A objective. These results
do not rule out other readout ranks, objectives, schedules or parameter groups.

Mean coordinate gradients on8 train-new windows versus8 A windows have cosine
0.236270/-0.412707. Equal-size Adam/raw-gradient shadow proposal effects on B
are at most a few thousandths NLL. The seed456 directional conflict is a
diagnostic observation, not an established cause of the plateau.

## Repair and continuation

The trace distributed gate/output multiplication over state-readout and skip,
while the official recurrence adds these contributions before multiplication.
Algebraic equivalence does not ensure equal floating-point output. The repair
restores official operation order; the1e-4 guard is NOT relaxed. A256-token,
large-output regression and19 virtual8 CPU tests pass. Actual repaired1.7B TPU
parity remains unverified; a mismatch will still stop the run.

Explicit `--repair-trace` accepts only the pinned original contract, the exact
audited repair code, this failure class, completed pre-dynamics modules, durable
4096-step readouts and absence of affected/locked measurements. Changes in data,
source, tolerance or other dependencies are rejected. Repair history preserves
the old error and both contracts. The saved readout checkpoints retain their
original contract; completed A/B/FIT modules are not repeated. Resume starts
dynamics, then precision and locked verification; source restore/parity still
run and consume time. No new result is declared by the repair itself.

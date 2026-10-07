# Measured update:7 October2026

Original supplied files preserved as `EXP-102-completed.json` and
`EXP-104-initial-parity-failure.json`. Neither contains document instructions to
execute; interpretation follows registered protocols and actual JSON fields.

EXP102 completed4.3377456765h, all6 branches8192, no pending HF, primary/cross-domain
gatesFALSE. Wiki final(seed123/456):3e-4=7.228510/7.216786;
1e-4=7.184306/7.174657;3e-5=7.177410/7.183887. Small-vs-large gains0.05110/0.03290,
below0.1. PG19 final:3e-4=9.202130/8.779207;1e-4=9.397119/8.848477;
3e-5=9.415622/8.859033. High LR has opposite domain preference. Shared3e-5 controls
match101 CE and103 PROTECTED final Wiki/PG19. No universal LR win established.

EXP104 stopped with ValueError `initial adapter/decoder NLL differs >1e-4`.
Control123 completed512 finite steps and uploaded85,782,792B state in22.256s.
Decoder123 remained at0; seed456 not trained. GatesNULL, notFALSE. Logged invocation
0.16228697h (~9.74min), not3h; notebook/prior setup elapsed time is not captured.
Initial dense reconstruction bitwise check passed. Old layout-sensitive NLL
differences(dense-control):validation+0.00010696,Wiki-0.00028217,PG19-0.00020431;
max window differences0.00292492/0.00248957/0.00275230. Layout/rounding is the
working numerical explanation, not an established TPU correction result.

V2 uses one explicit canonical BF16 materializer and evaluator layout. Threshold
unchanged1e-4; original starts re-evaluated before resuming optimizer states.
Old starts/validation diagnostics retained and labelled as noncanonical.
Original104 contract exactly archived in `EXP-104-v1-contract.json`; source digests
were independently matched against Git4401657. Source/data/objective/update masks/
optimizer/schedule/count unchanged; checkpoints retain original contract while
actual runtime implementation hashes and evaluation revision are logged separately.
Only this exact old contract is permitted; arbitrary setting changes rejected.

Measured free disk20,601,634,816B versus raw dense state16,369,669,640B: two dense
trajectories + next atomic generation cannot fit. Measured host available RAM
391,445,880,832B. Default Kaggle checkpoint storage moves to `/dev/shm`, with
capacity/RAM checks and last durable remote cursor. This does not fix sharding
memory or prove dense train-step feasibility. Control compiled temp10,465,362,944B;
full decoder compilation/training still unmeasured. No scientific104 outcome yet.

Local validation after correction:38 tests passed (EXP104/105 + existing plateau
campaign/stable clipping regressions),268.21s, eight virtual CPU devices. Four
targeted continuation/contract tests also pass after adding runtime AST/source
fingerprint guards. Both offline plans run without secrets or model allocation.
No real TPU104-v2/105 run or full old optimizer payload download performed locally.

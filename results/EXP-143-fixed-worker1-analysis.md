# EXP143 repaired worker1: PROJECTIONS continuation improves

Worker1 only supplied. deadline_partial, no error, registered work deadline,
11.2964h. Source12051 ->current24393,12342 accepted updates,3,159,552 input tokens.
One further attempted update was not accepted at deadline. Source all-window
replay MAX error0.0 onA/B/C: empty-coordinate repair worked on physical T4.
No newly added rank, no folding, no optimizer restart; this continues the
protected rank32+normalization+PROJECTIONS-reset recipe with old moments/clocks.

| point | A | B | C |
|---|---:|---:|---:|
|source12051|6.112766|6.179138|5.728322|
|registered best24339(new12288)|5.919346|5.947111|5.493689|
|last24393(new12342)|5.906142|5.915106|5.499488|

All source/final profiles pass consistency checks. Between registered new8192
and12288, A improves.046402,B.086546,C.107864, with intermediate fluctuations.
No evidence of a strict flat plateau over this tail; future progress not assured.
Context256->1024 tail5.683856->5.682413 is tiny and on only8 windows; no meaningful
long-context capability claim.

At matched NEW12288 steps:143PROJ gainA.193420 vs145folded gainA.204947; gainB
.232026 vs.198423; gainC.234633 vs.254065. Most absoluteA endpoint gap already
existed between different starting sources. This is DESCRIPTIVE:143PROJ source
12051 differs from145FULL source12586 in weights, reset recipe and data cursor.
Do not attribute145 absolute advantage to adding rank. Await143worker0(FULL,
same actual initial checkpoint as145), then148freshAdam control. Also do not
call this ordinary full-decoder CE/random-initialization control: rank32,
protection, normalization and prior resets/preparation remain in force.

Report pending with confirmed durable21907. Read-only immutable live HF snapshot
a848e128385229c8cfcf20f1cc0614485c812109 has final manifest24393 with original
contract matching this report. Full payload restoration not performed. Existing
save-readback deadline fix already published. No old/new runner code modified,
no cloud training launched or resumed by this result review.

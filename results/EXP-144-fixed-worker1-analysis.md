# EXP144 repaired worker1: LR/3 continuation near5.80

Only worker1 CE-LR-DIV3 supplied, paired originalLR worker0 not yet available.
Normal deadline_partial/noerror,11.291h. Source138CE current11965 ->23854,
11889 accepted updates/3043584 new input tokens; one attempted update unaccepted
at deadline. All source window replay maxerrors<.000760, tolerance.03. MambaLR
1e-4, pregateLR3.333e-5, all groups/3; original Adam moments/counts retained,
no warmup restart. Same fold130 ancestor from138, no new folding in144.

| point | A | B | C |
|---|---:|---:|---:|
|source11965|6.000852|5.992129|5.566408|
|registered best20157(new8192)|5.818954|5.875902|5.429943|
|last23854(new11889)|5.803475|5.824708|5.409201|

Tail afternew8192 adds3697 updates, improves A.015479/B.051194/C.020742.
Slowdown/noise onA is visible, no demonstrated hard flat plateau. Deadline
endpoint is descriptive, not retroactive registered best selection. Aselected
best remains8192 because no later registeredA beat it before deadline.
All source/final profile consistency checks pass. Tail context256->1024
5.554079->5.560328 does not improve long-context performance meaningfully.

Rate advantage is NOT established from this arm alone. Wait for144worker0,
which shares exact138CE start/moments/data.143PROJECTIONS and145folded stage have
different model ancestry/data/optimizer history despite some starts nearNLL6;
do not rank their final NLL as a causal LR comparison. Existing147/148 rate
screen at145current5.78 remains a separate source-conditioned experiment.

Report pending/durable21565. Read-only live immutable snapshot
 a848e128385229c8cfcf20f1cc0614485c812109 contains final manifest23854 and contract
matches report. Full payload restore not performed. Existing save-readback
transport fix already published. No numerical code modified or cloud job launched.

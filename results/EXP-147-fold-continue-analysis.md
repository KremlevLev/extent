# EXP-147 and EXP-148: continuation learning-rate screen

Both147 workers exit0/deadline_partial at registered work deadline. Parent
wall6.3959h, three Telegram messages. Source replay passes across160 windows;
all best/last profile consistency checks pass. Report pending empty, durable
steps equal finals. Read-only pinned HF manifests confirm steps31126/31095
and contracts plus arm names. Full payload restore not performed.

All three continuation arms restore SAME145 current24965 with Adam12379,
same data hashes/permutation and source model. No additional folding, rank,
optimizer reset or warmup restart; all learning-rate groups divided together.
Full common-checkpoint table is in EXP-147-fold-continue-analysis.json.

## Matched new updates

| New updates | LR | A | B | C |
|---:|---|---:|---:|---:|
|4096|Original|5.725551|5.751581|5.309792|
|4096|/3|5.738466|5.762110|5.315632|
|4096|/10|5.750668|5.770483|5.330003|
|5120|Original|5.728547|5.739723|5.323132|
|5120|/3|5.727711|5.747919|5.329712|
|5120|/10|5.748516|5.764374|5.334363|

At5120 (1,310,720 input tokens) original and /3 are essentially tied onA
(difference0.000836, smaller than maximum source per-window replay error);
original is better onB/C. At4096 original wins all three. Early smaller-rate
advantages occur; there is no uniform dominance or monotonic descent.

## Final checkpoints: nearly, but not exactly, equal work

| LR | New accepted updates | Input tokens | Last A | Last B | Last C |
|---|---:|---:|---:|---:|---:|
|Original|6161|1,577,216|5.692111|5.718191|5.292497|
|/3|6130|1,569,280|5.757107|5.738609|5.346462|
|/10 (148/1)|6158|1,576,448|5.758164|5.762702|5.349371|

Do not present these as exact matched-update contrasts. Registered A-selected
best original global31109/new6144 is5.707909/5.720280/5.315491;
/3 global30085/new5120 is5.727711/5.747919/5.329712;
/10 sameglobal30085 is5.748516/5.764374/5.334363.
Final original31126 is outside the registered grid, improvesA0.015798/C0.022994
over just17 additional updates. Preserve both final and registered values:
short-endpoint fluctuations are material, and final is not relabeled as best.

Original source5.782582/5.804530/5.364205 ->final improves
A0.090471/B0.086339/C0.071708. Atnew4608 A5.709290,5120 A5.728547,
6144 A5.707909: nonmonotonic, but hard plateau at5.8 is not supported.
No extrapolation to teacher quality or eventual convergence.

Practical default: originalLR continuation remains the strongest candidate
in this screen; no demonstrated reason to replace it with /3 or /10.
Evidence is one preparation seed and repeatedly exposed A/B/C validation,
not a universal optimal-rate claim or independent-test confirmation.
This echoes144 original-rate late advantage from a different starting model;
it is not an independent-seed replication.

Tiny8window last-context audit original256:5.482173 versus1024:5.493977;
/3 256:5.503327 versus1024:5.510017. No useful long-context gain established.
Passing layer-profile checks do not establish a unique plateau cause.

Original accepted correction updates since136source:14512379+1476161=18540
(4,746,240 full-hybrid input tokens); since111 through13612586+18540=31126
(7,968,256). These exclude prior111 training, local preparation, evaluation
and search across other arms; they are neither total project cost nor unique
tokens. Do not splice older12->7 on other panels into this curve.

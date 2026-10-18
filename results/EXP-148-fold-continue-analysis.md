# EXP-148: repaired Adam control and LR/10 continuation

Both GPU workers exit0 at the registered work deadline. Parent wall6.3986h;
three Telegram notifications (start,4h,final). Report pending lists are empty,
durable steps equal final accepted steps. Source replay passes all160 windows;
all last/best layer-profile consistency checks pass.

| Worker | Start | New accepted updates | Input tokens | Last A/B/C |
|---|---|---:|---:|---|
|0|136 FULL12586; retained factors, fresh Adam0, originalLR|6328|1,619,968|5.895821 /5.888958 /5.493516|
|1|145 current24965; retained Adam12379, ALL rates/10|6158|1,576,448|5.758164 /5.762702 /5.349371|

These arms have different sources and are NOT a paired endpoint comparison.
Worker1 introduces no further folding, extra rank or optimizer restart.

## LR/10 continuation

Source A/B/C5.782685/5.804455/5.364219; final gains
0.024520/0.041753/0.014847. Registered A-selected best global30085/new5120:
5.748516/5.764374/5.334363. Last global31123/new6158 worsens A/C relative to
that best. This is a small improvement, not evidence of rapid sustained descent.
Await147 originalLR and LR/3 at matched NEW counts to select a rate.

## Missing EXP-145 control

148/0 repairs the failed145 SAME-FACTORS control. Both it and archived145
FOLD-NEW-R32-NEW-ADAM start from136 FULL current12586 with freshAdam0,
same rates, FIT/validation hashes and global data permutation. Reset ramp2048
is already saturated at source12586; it does not restart. Source function
replay passes bounded BF16 tolerances. Full matched table is in analysisJSON.

| New updates | Control A | Fold A | Control B | Fold B | Control C | Fold C |
|---:|---:|---:|---:|---:|---:|---:|
|2048|5.988708|5.957616|5.960760|5.947869|5.596339|5.551483|
|4096|5.921496|5.907920|5.940984|5.910615|5.522973|5.496228|
|5120|5.869815|5.868214|5.942963|5.907475|5.516329|5.497676|
|6144|5.917068|5.890601|5.910803|5.887089|5.521145|5.477891|

Fold recipe wins all three panels at every shared checkpoint512..6144,
but is worse at128; A is nearly tied at5120 (difference0.001601).
At6144 its advantage is A0.026468/B0.023714/C0.043254.
This supports a modest benefit of the tested fold/new-correction stage beyond
Adam reset alone. It does not isolate capacity: folding changes optimization
coordinates and rounds frozen weights in BF16. One seed, repeatedly exposed
validation panels, not independent-test confirmation. Control has not reached
145's12288-update budget. Still await143 FULL for the preserved-Adam contrast.

Tiny context audits (8windows,127targets) show no substantial gain beyond256:
control256->1024 improves0.001467; LR10 worsens0.001214. Passing profiles do not
establish the cause of a quality plateau.

Read-only HF manifest audit confirms finals18914/31123 with matching contracts
plus arm names. Full payload restoration was not performed. No runner edits or
new cloud launches in this review.

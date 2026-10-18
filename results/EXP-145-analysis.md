# EXP145: successful fold/new-R32 trajectory; paired control failed

Aggregate failed after11.30245h because worker0 SAME-FACTORS-NEW-ADAM hit
KeyError('6') in source NLL, before any training. This is the known missing empty
frozen-attention coordinate schema defect. Worker1 FOLD-NEW-R32-NEW-ADAM exited0
at work deadline and completed its final evaluations. Do not discard valid
worker1 measurements or call the paired experiment a successful causal comparison.

| panels | starting folded source | registered best, new12288 | final, new12379 |
|---|---:|---:|---:|
| A |6.000774|5.795827|5.782680|
| B |6.004448|5.806026|5.804467|
| C |5.617387|5.363322|5.364202|

The source136 current12586 was folded into BF16 base; a fresh zero-effectrank32
correction and fresh Adam were attached, learned pregate weights retained, and
data cursor12586 preserved. Source per-window replay maxerror.000740<.03, so
folding/reparameterization preserved measured source quality. No rank64 identity
claim is justified after BF16 folding. All source/final profile consistency checks
pass. Registered best(global24874) is selected onA; arbitrary final(global24965)
is descriptive, not a retroactive replacement of best. C best and last differ.

12379 accepted updates =3169024 new input tokens; one more attempted window was
not accepted at deadline. Together with13612586 prior updates:24965*256=6391040
input tokens, STILL excluding earlier source preparation. Same validation panels
are reused. Controls missing: SAME-FACTORS with fresh Adam and plain unchanged
Adam continuation at matched new tokens. The observed improvement cannot be
attributed uniquely to rank expansion, folding, Adam reset or extra training.
Late registered A10240/11990/12288 new updates5.855947/5.812537/5.795827 and
last5.782680 support continued late improvement within this budget, not proof of
unlimited progress or teacher-quality restoration. Long context256->1024 tail
NLL5.551133->5.553845 has no useful gain.

HF report says pending, durable22442. Targeted immutable live snapshot
9d0a8f95d3df2ae566175fae950fca8f2759a233 contains FINAL manifest24965, original
contract matches report. See remote-checkpoint-audit.json. No full checkpoint
restore performed. Previously fixed save-readback deadline issue explains why
publication and confirmed-readback fields can disagree. No cloud job or new
training/repair runner created in this result review.

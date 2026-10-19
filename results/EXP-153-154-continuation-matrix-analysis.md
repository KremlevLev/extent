# EXP-153/154: ordinary continuation succeeds; added folding fails replay

Received19 October2026. Raw153/154 reports archived byte-for-byte; analysisJSON
contains full registered evaluations and costs. No runner/numerical code changed
in this review, no new cloud launch.

## EXP-154 contains a valid quality result

Aggregate FAILED because worker1 failed; worker0 exited0/deadline_partial,
registered work deadline, parent5.392437h. Ordinary KEEP-ADAM-R32 starts from
147current31126 with all source moments/count18540 preserved, same rates/data.
Source per-window replay passes: Amax.001242/B.000870/C.001105, tolerance.03.
4994 accepted NEW updates /1,278,464 input tokens /1,273,470 CE targets;
4995 attempted updates (one last attempt not accepted at deadline).
Lastglobal36120, corresponding Adam23534; no fresh rank/folding/reset.

| Panel | Source | Last | Source-minus-last |
|---|---:|---:|---:|
|A|5.692100|5.616958|0.075142|
|B|5.718168|5.675724|0.042444|
|C|5.292490|5.247005|0.045485|

Registered A-selected bestglobal35734/new4608:5.619403/5.672699/5.231323.
Last has slightly betterA but worseB/C; final is not relabeled as registeredbest.
A atnew2048/3072/3434/4096/4608/last:
5.686343/5.647643/5.675643/5.636984/5.619403/5.616958.
Progress is nonmonotonic; hard plateau at5.69 is not demonstrated. One seed,
reused validation panels, not independent-test or teacher recovery evidence.
Source and both last/best28-layer profile consistency checks pass. Tiny8window
tail context256->1024 worsens5.422374->5.438579; no useful long-context benefit.
Pending[],durable36120. Targeted read-only pinned HF audit at
8d4763d25a522ff2f0f1fdabe26ce81b4dd5b230 confirms manifest36120 and exact
report contract plus arm. Full payload restoration was not performed.
Four notifications: start, worker failure,4h progress, finish; no notification flood.

## Folding intervention: engineering failure before learning

153 both workers FOLD-FRESH-R64/R256 fail before first update, parent1.035045h.
154 worker1 FOLD-FRESH-R32 also fails before first update. Error is identical:
`immutable source NLL replay differs; refusing update`.
All three branch cursors remain31126, attempted/accepted training updates ZERO.
The failures are not evidence that these ranks train poorly. The registered
source-function replay gate rejected the initial intervention state.

154 worker0 validates original source loading, original136->145/147 base
reconstruction and retained-factor path on physical T4. Combined with folded
R32/R64/R256 failures, this narrows the additional problem to the fold/new-
correction path; it is not specific to large ranks. It does not establish the
exact arithmetic cause. Do not blame raw checkpoint corruption or capacity
from this evidence, or weaken the.03 tolerance to make the runs proceed.

The runner raises before storing source_controls/source_replay; actual panel
NLLs, worst windows and error magnitudes were lost from the reports. This is a
diagnostic gap. Future repair needs persisted measured-vs-reference windows
before rejection, direct source-versus-folded projection/layer output comparison,
and BF16 host-versus-device folding checks using the actual trained correction.
The33 previous CPU tests and real payload/synthetic optimizer checks did not
verify full-model BF16 function preservation for this new fold stage.

149/150 use the ordinary retained-factor base path;151/152 and154worker1 use
the added-fold path. Actual TPU statuses/results still unknown without reports;
do not mark them successful or failed from153 alone. Do not blindly rerun153
until this folding/replay discrepancy is diagnosed and a repair is verified.

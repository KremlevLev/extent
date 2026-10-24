# EXP157–158: capacity intervention and out-of-distribution checkpoint diagnosis

Prepared24October2026. No physical jobs launched by this preparation.

Observation: received EXP155 session2 CURRENT118304 has A/B/C4.975879/5.061456/4.457787; previous80260 had5.109496/5.150924/4.665559. New9.739264M input positions buy modest reused-panel improvement. EXP156 tested80260, not118304, and found approximately chance-level capabilities on807–808common questions/task; fresh EN/RU NLL7.439/11.447 vs donor1.858/1.848. Domain specialization is a hypothesis, not an established cause. Reduced source-panel NLL is not evidence of donor-equivalent capabilities.

## EXP157 question and interventions

Does freeing the decoder, starting from the latest recovered model, improve recovery and transfer beyond protected rank256 adaptation?

One9h8-device TPU session with18min final reserve. Two arms from the same immutable CURRENT118304, identical cyclic FIT stream and source cursor, fresh Adam with512-update warmup, global stable clip1. R256 keeps protected in/out correction factors and pregate; full decoder trains all28block parameters including Mamba dynamics, four GQA mixers, allMLPs and block norms, with embeddings and final norm frozen. Mamba architectural MIMO rank remains4. Adaptation rank256 is a separate concept.

Dense initialization materializes both trained rank32 and extra224 factors with the source FP32 addition order and one BF16 cast, then creates FP32 masters of the resulting decoder parameters. No new memory reset or source normalization change. The source factors are removed from dense trainable coordinates; no redundant low-rank update is applied twice. Source per-window replay<=.03 on all160windows is mandatory for each arm before any accepted update.

LRs: rank256 mixer3e-4/pregate1e-4; dense Mamba+GQA1e-4/MLP1e-5/blocknorm+pregate1e-5. Thus LR and parameterization both differ: this tests practical recovery recipes, not capacity alone. There is no claim these are optimal rates. A failure to improve would not establish impossibility of dense recovery. Both use CE, no teacher distillation.

Arms first get8updates each, then alternate blocks<=2048updates, choosing the least advanced arm. Compare registered common new-step endpoints and wall time separately; last tails may not be matched. No fixed8192-step early stop;262144 ceiling per arm intentionally beyond session reach. Every accepted proposal requires finite gradients/parameters/optimizer. Actual first compiled/streamed gradient comparison uses existing relativeL2<=.15, cosine>=.98, CEerror<=.03, otherwise same-source streamed fallback. Eight-device CPU checks are engineering evidence, not TPU throughput or HBM proof.

A/B/C logged at registered endpoints. Fresh EN TinyStories/RUWiki panels are exactly the pinned156 text windows and are never added to FIT. First16windows/language at source and every8192updates; all128 at final when reserve permits. Source A selects best; fresh panels are descriptive. Since these fresh results have now been inspected, they are diagnostic validation, not an untouched final test set. Future confirmatory tests need another locked split.

Resource/storage: shape-only dense trainable count1,364,173,952; current+Adam+selectedbest checkpoint ~21.83GB. Local atomic generations prune obsolete local state; do not retain thousands of snapshots. HF checkpoint sync roughly every4hours plus exit;80GB per-session upload cap, pending state and errors explicit. Frozen donor base not uploaded. Full decoder checkpoint payload is intrinsically larger than an adapter.18min is requested reserve, not an upload-time guarantee.

## EXP158 checkpoint diagnosis

TwoT4 workers compare frozen80260 and118304 on identical156-pinned fresh text, context tails, first256questions/task in existing fixed order and24greedy completions. Both source replays required. No training, model selection or published official-score comparison. Resume accepts only same own code/source/data contract. Paired bootstrap intervals compare common IDs and complete identical NLL panels. Stop after planned subset even if wall budget remains; this is deliberately a smaller diagnostic. Existing156 donor measurements provide context but are not silently merged into a different question subset.

Useful outcomes: old-panel improvement plus fresh-text improvement supports broader recovery; old-only improvement suggests limited transfer but does not identify its cause. Dense improvement beyond rank256 supports expanding trainable scope under this recipe. Failure/divergence requires examining numerical health, LR and data before abandoning the architecture.

## Provenance

Source HF revision5e0d5026d03da0320c72162f5216215b137e43c6, manifest step118304, SHA and160reference windows in EXP-157-source-pin.json. Prior80260 immutable pin remains EXP-156-source-pin.json. EXP155 received session2 archived; EXP156 received metrics/generations archived in a compact summary with raw-file SHA. Original155/156numerical files remain unchanged.

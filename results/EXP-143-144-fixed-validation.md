# EXP143/144 source-container repair validation

39 CPU tests passed in63.51s:6 new fixed-source/launcher cases,15 existing long
runner cases,18 HF download/artifact tests. After descriptive contract-message
cleanup, fixed plan and parent-dispatch cases passed again(2cases,.83s).

For EACH source recipe136FULL,136PROJECTIONS,138CE: tiny REAL hybrid model,
nonzero Adam moments/three clocks, real chunk save -> old reader/restore -> real
NLL reproduces KeyError on empty GQA. Fixed reader/restore returns the exact
original complete pytree and byte-identical tensor arrays. Real jitted ordinary
NLL(current AND best), per-layer profile, streamed full-chain update vscompiled
objective, own checkpoint save/restore all pass. No mocked NLL in these tests.
Missing Mamba and unexpected coordinate keys rejected; existing learned attention
coordinates retained. Two-process launcher dispatch tested with subprocess stub
and old report preserved; real GPU execution remains unavailable locally.

Three REAL pinned1.7B-source coordinate/Adam/best checkpoints restored from local
SHA-verified caches: source136FULL12586/best12288,136PROJ12051/best10240,
138CE11965/best10240.7,182,336q tensor elements each; allsix original Adam/schedule
counts retained, empty6/13/20/27 present in q and best. See fixed-source-audit.
This checks actual checkpoint restoration, not full-size model NLL/gradients.

All complete ORIGINAL143..146 contracts still equal frozen original engineering
plan. No old script, numerical module or source pin modified. New143/144 recipes,
model, source hashes, sequence/batch, selection and evaluation/data clocks equal
original contracts. New fixed code hashed into separate exp143-144-long-fixed-v2
contract/namespace. No migration or overwriting of old failed/running states.
Old145/146 remain on their old runner and namespace. Freeze fixed-plan.json for
review; fixed launches are in EXP-143-fixed-launch.md / EXP-144-fixed-launch.md.

No repaired Kaggle job was launched by this local work. No new NLL result.

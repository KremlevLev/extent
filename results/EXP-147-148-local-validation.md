# EXP147/148 engineering validation

27 CPU tests passed in82.47s (7new continuation cases +20existing HF retry/artifact
cases). After metadata-only checkpoint-description cleanup, real CLI/dispatch
case passed again(.82s). No physical T4 execution performed in this preparation.

Four actual tiny-model paths147/0,147/1,148/0,148/1 reconstruct the correct frozen
base and source q, preserve or reset Adam as prescribed, calculate real NLL,
perform streamed full-chain next updates against compiled-objective reference,
check finite proposals and six counter increments, save/restore own full payload.
A separate real chunk source restore uses global24965 vsAdam3 and verifies that
using global24965 as expected Adam count fails. All q/Adam/best leaves equal
saved original, including empty GQA dictionaries. LR1,/3,/10 change only update
scale with identical moments and counters. CLI two-worker dispatch,6.8h total,
6.3h work,7h upper bound, namespace and no-child-Telegram exercised.

ACTUAL source145 current24965 at immutable9d0a8f95d3df2ae566175fae950fca8f2759a233:
487 chunks/114,979,740bytes downloaded and SHA checked. Complete q/moments/best
restored through NEW loader, allsix counters12379, best24874,7,182,336q elements,
empty GQA6/13/20/27 retained. Source136 q-only ancestor reconstructed via actual
selective path, all q chunks checked, no unused ancestor moments needed. Actual
source synthetic-gradient optimizer next update equals original145 optimizer
BITWISE for every update/state array; six counters12379->12380. This is a
synthetic-gradient CPU optimizer check, NOT a1.7B model gradient/GPU run.
Audit contains exact manifests and parameter/best tree hashes.

All complete original143..146 contracts and repaired143/144 contracts equal
existing frozen plans after these additions. No shared numerical/runner file
changed, no old checkpoint overwritten/deleted. New plan registers new scripts,
source/schema helper hashes with canonicalLF; raw tensor SHA remains byte-exact.
No cloud job started and no new scientific result claimed. Initial all-window
GPU NLL replay/finite/HBM checks stay mandatory before first accepted update.

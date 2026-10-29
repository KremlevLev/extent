# Extent working preprint

[Read the draft](draft.md) · [Русская версия](draft.ru.md). Updated 29 October 2026; **not submitted or peer reviewed**.

The draft describes the evidence chain: local transfer versus full composition,
hybrid-prefix input shift, numerical norm overflow, and protected recovery with
an unresolved quality plateau, and a single-seed combined dynamics/factor reset that improves matched continuation NLL. Historical results still require a complete
raw-artifact audit before submission; the mathematical lift derivation and
detailed related-work comparison remain unfinished.

EXP-111 and EXP-113 are completed. EXP-112 is partial at 12288/8642 MAIN steps,
with recorded durable states. The draft now includes received EXP-125–132 sessions. EXP-133/134 outcomes are not included.
The draft does not claim Qwen-equivalent quality, isolated initializer superiority,
an equal-total-cost speedup, 14B recovery, or faster inference.

## Figures and provenance

`figures/` includes PNG previews, PDF/SVG exports, extracted CSV points and source
provenance. Figures 1–2 contain 116 recorded points from EXP-072-v2 and EXP-097.
Figure 3 contains 308 recorded points from the latest received EXP-111–113 JSONs.
Seeds are shown separately; missing endpoints are not imputed and spikes are
not smoothed. MAIN-token axes exclude LR pilots and warm-start preparation.

Figure 3 can be rebuilt with Python and matplotlib:

```sh
python paper/make_efficiency_figure.py
```

Large run JSONs and model checkpoints are not duplicated here. Figure provenance
records raw-artifact SHA256 values and source contracts. `draft_sources.json`
identifies historical document snapshots; its hashes need not match later ledger
edits. The repository's `results/` and technical report remain the research ledger.

## Recent figures (4–7)

Four additional figures are rebuilt directly from archived EXP-129–132 JSONs:

- Figure 4: side-by-side scale, absorption, and quality comparisons using accepted layer-26 profiles.
- Figure 5: annotated NLL/change matrix for norm/tanh, dynamics unlock, MLP adaptation, and reset at 3,072 updates.
- Figure 6: A64-only post-ramp trajectories, with independent subplot axes and arbitrary final endpoints distinguished.
- Figure 7: CE versus pure teacher KL at 2,560 updates, with measured training cost shown separately.

Run `python paper/make_recent_figures.py` with matplotlib installed. This is offline analysis only. The script checks source/data consistency, durable cursors, completion of finalization, and source replay before extracting points. All four new figures have PNG, PDF, and SVG exports. `figures/recent_points.csv` contains 87 measured panel points; `figures/recent_provenance.json` and `../results/EXP-129-132-session-1-analysis.json` record raw hashes, costs, and failed profile checks. Raw JSONs are archived once under `results/`, with no checkpoint weights.

The late runs are exploratory single-seed results on reused panels. The reset changes several components together and does not establish which caused the improvement or whether earlier preparation was harmful. The draft also reconstructs selected-path preparation presentations separately from full-model recovery and unmeasured teacher/search compute.

## Latest recovery evidence (through EXP-166)

The English and Russian drafts now include the extended EXP-155 lineage, the
Wiki-versus-fresh-C4 domain tradeoff, matched EXP-163 adaptation recipes, the
48.05M-input full-decoder continuation, and bounded downstream/hidden-loss checks.
EXP-167 remains a prepared LR comparison, not a result.

Figure11 now covers the selected lineage to EXP-166, with the alternative Wiki
branch explicitly separated. Selected-path full-model presentations144,501,248;
local block-training preparation17,301,504 separately. Figure12 shows matched
EXP-163 quality and EXP-166 changes from its own source. Offline rebuild:

```sh
python -m paper.make_progress_overview
python -m paper.make_latest_figures
```

`latest_recovery_data.json` archives a compact numeric extraction with original
received-report hashes, domain panels, stage offsets and a checked token ledger.
It contains no weights, secrets or runtime-machine details. Reported panels are
reused diagnostic validation; an independent full-model capability result for
CURRENT181145 has not been received.

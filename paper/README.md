# Extent working preprint

[Read the draft](draft.md). Updated 13 October 2026; **not submitted or peer reviewed**.

The draft describes the evidence chain: local transfer versus full composition,
hybrid-prefix input shift, numerical norm overflow, and protected recovery with
an unresolved quality plateau. Historical results still require a complete
raw-artifact audit before submission; the mathematical lift derivation and
detailed related-work comparison remain unfinished.

EXP-111 and EXP-113 are completed. EXP-112 is partial at 12288/8642 MAIN steps,
with recorded durable states. EXP-114 is prepared; no TPU outcome is included.
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

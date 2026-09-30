# EXP-093 completed: foldable mixer gains

Reviewed 2026-10-01. User artifact: `extent-m3q-mixer-gains.txt`, SHA256
`5aaa8a4b79459c53338448b5611006aa732d14e3754430544f9d8bacb7f080fc`.
All four registered 2,048-step branches completed; scientific gate **failed**.

| Seed | Unchanged test NLL | Global gain | Per-layer gains |
|---:|---:|---:|---:|
| 123 | 13.261727 | 17.616229 | 12.451787 |
| 456 | 9.161675 | 12.788677 | 10.170573 |

Per-layer gains beat the global gain in both seeds, but improve the unchanged
model only in seed123. Output scaling alone is not a reliable recovery recipe.
The `0.150883` hours field describes the last resumed invocation, **not total
campaign compute**; earlier session durations were not cumulatively recorded.

Next intervention: richer foldable corrections, with frozen base weights and
paired whole-model objectives (EXP-094--096). This does not establish that a
random transplant is preferable, nor estimate tokens for full 14B recovery.

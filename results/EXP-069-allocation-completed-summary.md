# EXP-069: completed allocation comparison

Reviewed 2026-09-07. User artifact `extent-m3q-allocation-campaign.txt`, SHA256 `1c06241bb3312011d8657cff8cd8162a7d119a6bba12b8dc0f502abbbd015128`; run revision `a385256db8cd34dd09b6dc54458c54cec54a9911`. Numerical version: `fp32-scaled-overflow-fallback-v1`.

Session duration 1.604789 hours; four arms completed 3,072 updates each, two complete pairs. This is the resumed session duration, not the total cost including preparation and failed runs. All arms report finite execution. Final checkpoint uploads passed for all four arms; 55 restore and 15 upload events passed in the supplied log. No live HF inventory was independently checked.

## Whole-model NLL (lower is better)

| Seed | Placement | Initial | Step 256 | Step 1024 | Step 2048 | Step 3072 |
|---|---|---:|---:|---:|---:|---:|
| 123 | UNIFORM | 19.6557 | 38.1513 | 22.3332 | 14.7348 | 13.2689 |
| 123 | ATLAS | 18.9281 | 46.15 | 22.99 | 31.87 | 25.7932 |
| 456 | UNIFORM | 19.9360 | 40.8294 | 19.3534 | 22.9250 | 21.1412 |
| 456 | ATLAS | 19.4930 | 46.44 | 47.29 | 19.23 | 16.3975 |

Original Qwen NLL is 2.805075 in UNIFORM evaluations and 2.805911 in ATLAS evaluations. This small systematic discrepancy (~0.000836) is recorded, not rounded away as exact teacher identity. Its cause is not established here; it is much smaller than the observed placement differences. Audit teacher execution parity before precision claims.

| Seed | Final NLL difference ATLAS minus UNIFORM | Normalized NLL AUC difference |
|---|---:|---:|
| 123 | +12.524302 | +9.292229 |
| 456 | -4.743678 | +7.041444 |

Mean final difference: +3.890312, favoring UNIFORM in this two-seed sample. The registered gate requires ATLAS to improve both final NLL and AUC in both seeds; it fails. Opposite final signs and only two seeds do not establish universal superiority of either placement. Window measurements are not independent training seeds.

## Interpretation

- The norm-overflow correction removed the reproduced stopping condition: all four arms completed. This is engineering success, not successful capability recovery.
- Local layer compatibility ranking did not demonstrate a reliable whole-model placement advantage under this preparation/recovery recipe.
- Both placements remain far from the teacher. Two arms finish worse than their own initial NLL; all four worsen sharply by step 256. Finite execution is not stable or effective recovery.
- Final training gradient norms range from approximately 1.26e13 to 6.07e19. These are pre-clipping norms; they do not mean optimizer updates have that magnitude. The concentration and source of these gradients have not been measured.
- Full recovery budget is only 786,432 input-token positions per arm (3,072 x 256, batch one), excluding local preparation. These findings do not prove that longer recovery cannot work, but current non-monotonic curves do not justify simply extending the same run as the next test.

## Next question, not another allocation sweep

Keep one placement fixed for diagnosis. Locate which parameters/layers dominate gradients and measure activation scales as replacements are composed, before spending another long training budget. Separate the hypotheses of composition/input-distribution shift, local context 64 versus full context 256, and damage from updating all preserved Qwen weights immediately. Then register a matched stabilization comparison (for example mixer-only warmup versus immediate joint recovery) based on that evidence. Neither a mechanism nor a new method win is claimed yet.

Do not rerun unchanged EXP-069: the four completed checkpoints and preparation bank are reusable from HF. MLA, long-context recovery and inference speed were not tested here.

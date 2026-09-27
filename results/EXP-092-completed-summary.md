# EXP-092 — completed whole-model parameter-path probe

Reviewed 2026-09-27. Protocol/code revision `79d41fc620089f09096480bde818676ff634ae39`. Status: all 3 registered branches completed on TPU v5e-8 in 0.113 hours. Scientific gate: **FAIL**.

The input EXP-091 artifact SHA-256 matches the recorded source hash `2e505e6e2dfa4c074cda505c8b113b36b9569c268252412cabe2afae76c2ad13`. Full result SHA-256: `e4859b15c253589e4d4f418f669c511700414557e859cb7cc1ca9dd9d39bdfb1`. The user-supplied summary SHA-256 is `257f6313004626cadf878c5ffcc52f4731ddccb465dcd891b11b676b91fb5982`.

## Frozen comparison and observed result

For each completed EXP-091 endpoint, interpolate the *entire model* from its exact EXP-072 ONPOLICY start with alpha in `[0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]`. Select on 16 fresh length-256 validation windows; report a separate locked 32-window test split. No training occurred in EXP-092. NLL is lower-is-better.

| Seed | EXP-091 endpoint | Validation NLL alpha 0 | Validation NLL alpha 0.03125 | Selected alpha | Locked-test NLL alpha 0 | Locked-test NLL alpha 1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 123 | PLAIN | 10.2786 | 16.0712 | 0 | 13.2617 | 16.4545 |
| 123 | ANCHOR | 10.2786 | 13.4539 | 0 | 13.2617 | 16.6815 |
| 456 | ANCHOR | 9.6712 | 12.9432 | 0 | 9.1617 | 21.7363 |

Alpha zero is the minimum validation NLL over **all seven** registered candidates in every branch, not merely a fallback under the 0.01 acceptance threshold. Therefore selected locked-test NLL equals the unchanged start in all three cases. The raw trained endpoints increase locked-test NLL by `+3.1928`, `+3.4197`, and `+12.5746` respectively. The two-seed ANCHOR gate required nonzero selected alpha and a >0.01 locked-test gain at both seeds; neither seed meets it.

## Interpretation and boundary

The specific EXP-091 full-model update directions do not contain a useful point on this registered *straight global interpolation path*. Even 1/32 of each direction damages validation NLL substantially. This rejects using these endpoints for a simple global fractional-update or staged path-training proposal. It does **not** prove that every path, smaller alpha, different optimizer, or Transformer-to-Mamba method fails. It also does not test the 14B model or MLA conversion. A further long run along the same EXP-091 direction is not justified by these data. Keep EXP-091/092 as negative controls; any new training campaign needs a distinct mechanism and an equal-compute comparator.

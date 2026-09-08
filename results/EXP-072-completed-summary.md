# EXP-072 v2 — completed full-depth sequential confirmation

Reviewed 2026-09-08. Status: completed; the pre-registered `ONPOLICY` gate passes at all 24 Mamba replacements.

The supplied full artifact has SHA-256 `7ebe6a13ed888d5145cd1210e6dfc73d66b3760e7f8f238f009a4aa439b80b81`; its compact summary has SHA-256 `4fcd1c95fbf3a0307c1394466dcaa5300cae8dd47e8fb3c66a7cebceae4da6fc`. The resumed session took `0.350796` TPU hours at revision `5c3dc46`; its endpoints extend the earlier `1.033192`-hour partial run.

## Registered result

| Seed | TEACHER NLL | ONPOLICY NLL | NLL delta | Delta AUC | ONPOLICY excess NLL |
|---:|---:|---:|---:|---:|---:|
| 123 | 22.346362 | 10.545254 | -11.801108 | -6.854519 | 7.477798 |
| 456 | 22.037049 | 10.266002 | -11.771046 | -5.862882 | 7.198546 |

Both the final-NLL and trajectory-AUC requirements pass at both seeds. Mean final `ONPOLICY-minus-TEACHER` NLL is `-11.786077`. Relative to equal-update `TEACHER`, `ONPOLICY` reduces total NLL by `52.81%/53.42%` and excess NLL above the cached Qwen teacher by `61.21%/62.05%` for seeds 123/456.

The effect grows with replacement depth through 20 layers. `ONPOLICY-minus-TEACHER` NLL is `0.000/0.000` at depth 1, `-0.679/-0.406` at 4, `-1.347/-0.846` at 8, `-6.280/-4.863` at 12, `-10.579/-9.737` at 16, and `-16.427/-13.491` at 20. This shape is mechanism evidence: upstream hybrid input shift accumulates, so clean-Qwen-only local training becomes increasingly mismatched.

All sequential variants deteriorate in the final four replacements. `ONPOLICY` rises from `7.004257/6.894241` at depth 20 to `10.545254/10.266002` at depth 24. It remains decisively better than `TEACHER`, but sequential current-layer-only recovery does not solve full-depth quality by itself. Original Qwen NLL on this slice is approximately `3.067456`.

`MIXED` is not uniformly ranked: it beats `ONPOLICY` by `0.480467` at seed 123 depth 24 but loses by `1.655679` at seed 456. Its mean final advantage over `TEACHER` is slightly smaller than the registered primary's (`-11.198472` versus `-11.786077`). No method selection is changed.

## Decision

Accept deployment-distribution sequential recovery as a confirmed full-depth mechanism on Qwen3-1.7B. Proceed to EXP-073 to test whether its advantage persists during equal-budget joint full-model recovery. Do not claim recovery to Qwen, compute-normalized superiority to random, 14B transfer, MLA compatibility, or long-context speed from EXP-072.

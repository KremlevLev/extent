# EXP-069: first full-model attempt failed

User-supplied result reviewed 2026-09-06. Source: `extent-m3q-allocation-campaign.txt`, SHA256 `97e848eb188e4e23dd6394bc19a6072c238deb42d7eb17eeab655f4839b4ba1b`; run revision `75267024596a571eead2bb4c9d709f571dafdf57`.

- Duration: 1.234301 hours. Completed allocation pairs: 0/2.
- First arm: seed 123, UNIFORM. Completed updates: 105; failure on attempted update 106.
- Only full-model evaluation: step 0. Student NLL 19.655661; teacher NLL 2.805075; excess NLL 16.850586; prediction KL 21.620184; top-1 agreement 0.003799.
- No post-training evaluation; no ATLAS result. The false gate is an incomplete experiment, not evidence against ATLAS.
- The 54 prepared endpoints were constructed. Latest recorded upload outcomes confirm 51 prepared endpoints and full seed-123/UNIFORM step 0. Seed-456 layers 21, 22, 23 have no later successful upload event. This is log evidence, not a live Hub inventory.
- Checkpoint 0 contains weights and optimizer. Updates 1–105 were not durably checkpointed. A fresh session can restore confirmed artifacts; missing preparations may need recomputation. No PC transfer required.

## Failure interpretation

The old guard merged non-finite gradient elements, scalar loss failures, and norm overflow into one exception. Training metrics were recorded only at evaluation milestones, so the exact failed metric is unavailable. The old `finite=True` arm flag was stale and is not evidence that the failed update was finite. Do not attribute this to TPU hardware or conclude the architecture cannot recover.

Diagnostic patch records attempted step, deterministic batch index, last durable step, all failed scalar metrics (NaN/Inf encoded as strings), and the last successful training metrics. It sets the failed arm flag correctly and leaves the last good checkpoint untouched. No optimizer, data, initialization or allocation hyperparameters are changed. This improves observability; it does not claim to fix the unknown numerical cause.

Next invocation uses the same entry point and HF resume. If the failure repeats, inspect `failed_update` in the campaign JSON/summary before another long run. Do not blindly spend the full remaining quota repeating it.

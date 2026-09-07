# EXP-070 v2: overly strict FP32 control

Reviewed 2026-09-07. User artifact SHA256 `c944e547ffe0c4b3a164af2cebf9b4277d95d7344b20895f3aaab25d0b6ce483`; run revision `2ef38be21419480e9531a0f81699839d2e4a8fde`. Runtime 0.0388615 hours (~2.33 minutes). Status failed, accepted probes 0/40. No Mamba replacement was evaluated, hence no evidence about input shift or sequential calibration.

The preserved all-GQA seed-123/length-64/window-0 BF16 measurement repeats v1 exactly: student NLL 2.7808144093, teacher NLL 2.7824058533, excess -0.0015914440, KL 0.0031070698, top-1 agreement 0.9841270; gradients finite with norm 6.6149521 and max absolute 1.142578125.

The new FP32 joint control reported student/teacher NLL 2.7789120674/2.7778170109 (difference +0.0010950565), KL 0.0001315110, top-1 agreement 1.0, logits error RMS 0.02275832 against target RMS 10.75228977, and logits relative L2 0.0021166021. It failed v2 thresholds NLL <=0.001, KL <=0.0001 and relative L2 <=0.0001. The exact parameter-tree contract passed before execution.

These small but nonzero differences show that FP32 on TPU is not bitwise/effectively identical across the two wrapper computations under the original lowering. They do not establish the cause as compiler scheduling, nor prove the wrappers semantically identical. Conversely, 100% top-1 agreement and 0.21% logit relative L2 do not look like gross parameter mis-mapping. It was correct for v2 to stop under its registered bounds; those bounds were unsuitable as a corruption gate.

V3 uses highest matmul precision, records per-layer hidden relative L2 to localize divergence, and sets explicit observed-background sanity bounds (NLL .005, KL .001, logit and maximum-hidden relative L2 .005). These post-v2 bounds are not independent confirmation criteria and must not be presented as equivalence. Raw BF16 controls remain part of every baseline probe. V3 uses a separate output/HF namespace and requires new TPU confirmation.

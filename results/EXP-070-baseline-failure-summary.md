# EXP-070 v1: baseline guard stopped the campaign

Source: user-provided `extent-m3q-input-shift-campaign.txt`, SHA256 `ba80b7fbba3d21aa4a492fec85e4882f42912744b69f52fac62b58be1bf8ec52`. Revision `ec26585c566a88c7ee127f084f5ea157de9461c8`. Duration 0.0378495 hours (~2.27 minutes). Only 1/40 probes recorded; no Mamba replacement tested.

Recorded all-GQA seed-123, length-64, window-0 metrics:

- Student NLL 2.7808144093; teacher NLL 2.7824058533; difference -0.0015914440.
- Prediction KL 0.0031070698, top-1 agreement 0.9841270.
- Gradient norm 6.6149521; all gradient leaves finite; maximum absolute gradient 1.142578125.

The next probe failed the all-GQA baseline guard, but its metrics were thrown away before persistence. Exact failed values are unavailable. The first recorded probe already demonstrates nonzero discrepancies between the two BF16 evaluation paths. Compilation/precision is a plausible explanation, not an established cause from these artifacts. No inference about input shift, Mamba dynamics, or sequential calibration is supported.

## Version-2 correction

Do not blindly increase the BF16 tolerance. Add a separate FP32 forward parity control with identical checkpoint weights and explicit NLL/KL/logit tolerances; keep original BF16 discrepancies as numerical-background measurements. Both architectures are evaluated in a common FP32 control function. If that check fails, preserve raw BF16 and FP32 diagnostics and stop. A FP32 pass does not establish gradient parity or eliminate BF16 execution differences.

Save baseline data before running the control/raising. Rejected baselines are not resumable completed probes. Version-2 artifacts use a separate output subdirectory and HF prefix to avoid mixing changed acceptance rules with v1. Prepared EXP-069 checkpoints are unchanged. TPU confirmation of the new control is still pending.

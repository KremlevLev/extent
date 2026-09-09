# EXP-075 — protected joint recovery (completed)

- Status: completed in 3.587978 hours; both registered seeds and both arms finished with finite metrics.
- Artifacts supplied by the user: full JSON SHA-256 `6c916c0ec759759a49fc524fb14efc9a4f18994623b1c6e94767058e9c0f08a6`; summary SHA-256 `f286f481d46279b5943455c7f565492bbcc579d8496a0e9f75743137024f2281`.
- Registered gate: **FAIL**. No trained checkpoint in either arm/seed beats its own step-zero NLL.

| Seed | Arm | Step-0 NLL | Final NLL | Final delta | Max/start |
|---:|---|---:|---:|---:|---:|
| 123 | MAMBA-ONLY | 10.916678 | 11.694606 | +0.777928 | 1.543568x |
| 123 | MAMBA-NORMS | 10.916678 | 11.472961 | +0.556283 | 1.378560x |
| 456 | MAMBA-ONLY | 9.692907 | 11.304210 | +1.611303 | 1.483537x |
| 456 | MAMBA-NORMS | 9.692907 | 11.904981 | +2.212074 | 1.373714x |

## Interpretation

Protecting copied Qwen parameters materially reduces the damage relative to EXP-074's matched all-parameter `LR3E-6` arm, whose final degradation was `+4.755375/+6.251433`. The corresponding Mamba-only degradation is only `+0.777928/+1.611303`. This identifies copied-backbone movement as a major amplifier of instability, but not its complete cause.

Every protected curve rises sharply during early Lion updates and then recovers as the cosine schedule decays, yet remains worse than the untouched sequential endpoint. Training gradients are finite but extremely large before clipping. Norm training has no consistent advantage across seeds. Therefore the selected result is to keep Qwen frozen and test the optimization objective before spending compute on random-init comparisons or 14B scaling.

This experiment does not show that the transplant is worse than random. It only shows that the current global output-KL/CE joint-recovery recipe cannot improve an already useful sequentially recovered transplant.

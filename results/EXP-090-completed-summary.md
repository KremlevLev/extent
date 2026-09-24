# EXP-090 — completed long-horizon timestep-freeze comparison

Reviewed 2026-09-25 from the user-supplied full artifact. The last invocation ran `5.772` hours, completed all four 98,304-step trajectories, and uploaded all 12 checkpoints successfully. Every endpoint is finite.

Each arm received 25,165,824 training-token presentations per seed. `RANDOM-DT-FROZEN` froze the raw timestep projection and `dt_bias` through step 32,768, then trained them through step 98,304. Both arms otherwise used the same random initialization, data and optimizer.

| Seed | Random excess NLL | Frozen-dt excess NLL | Frozen − random | Random KL | Frozen-dt KL | Frozen − random KL |
|---:|---:|---:|---:|---:|---:|---:|
| 123 | 9.763291 | 9.538814 | −0.224476 | 14.612423 | 14.329857 | −0.282565 |
| 456 | 7.282903 | 9.657643 | +2.374740 | 11.010735 | 14.382832 | +3.372097 |

Negative differences favor freezing. The direction reverses between seeds, and the loss on seed 456 is much larger than the win on seed 123. The descriptive mean frozen-minus-random difference is `+1.075132` excess NLL and `+1.544766` prediction KL, favoring ordinary random. The pre-registered gate required the frozen-dt arm to beat random in both metrics at both seeds; it failed.

This rejects early hard freezing of `dt` for 32,768 updates as a reliable improvement under the locked recipe. It does not rule out other schedules or timestep parameterizations. The result does not support a positive transplant claim, since neither arm transfers Qwen attention weights into Mamba mixers.

The two seeds share one held-out validation slice, so the seed-level endpoint reversal does not establish data-domain robustness. The evaluation curves were non-monotonic. Do not select the best intermediate checkpoint after observing it and present that as the registered endpoint.

Full artifact SHA-256: `9dfdedcf11b7e453ba1501087dd7f1a0d419b622b8f9995f2b7e8d7924f05a37`.
Run revision: `96052976e5b5cb27bbd7b5f5ca1dedd04e3a0b11`.

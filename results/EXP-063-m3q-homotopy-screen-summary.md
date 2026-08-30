# EXP-063 — M3Q homotopy screen

- Source: `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`
- Runtime: 2.777 hours on TPU v5e-8
- Completion: 5/5 layers, all numerical checks passed
- Globally selected schedule: `M3Q-EXACT-COSINE`
- Advancement gate: FAIL (1/5 layer gates)

| Layer | Exact final L2 | Cosine final L2 | Final gain | AUC gain | Decision |
|---:|---:|---:|---:|---:|---|
| 0 | 0.7340133 | 0.7489481 | -2.03% | -2.50% | worse |
| 6 | 0.0056889 | 0.0056891 | -0.00% | -0.01% | tied |
| 13 | 0.0101628 | 0.0101622 | +0.01% | -0.01% | tied |
| 20 | 0.0940224 | 0.0935862 | +0.46% | -2.22% | mixed |
| 27 | 2.9270126 | 2.5644604 | +12.39% | +17.50% | win |

The fixed attention-to-Mamba homotopy is rejected as a universal recovery
recipe. Its only material win is at layer 27, which is a retained-attention
layer in the registered 24-Mamba/4-attention Qwen3-1.7B hybrid. It harms layer
0, the one screened layer that is actually replaced by Mamba in that plan.

The result does not reject the exact MIMO lift or full-model distillation.
It rejects adding one globally fixed homotopy schedule to them. The next
experiment therefore moves to whole-model staged hidden-state and prediction
distillation, with exact lift as the initialization and standard end-to-end KL
as the paired baseline.

Artifact SHA-256:

- Full JSON text: `2e7003eb4804efb095ee6e4dcf2ae4cbe98162eb6af3aaad116497642aa7bcca`
- User summary: `d8e420a1384923e85bb2a901750b8c5b8da7e0a8653bb28135c2fe11c944e5e8`

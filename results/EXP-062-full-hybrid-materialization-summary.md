# EXP-062 full-hybrid materialization bring-up

- Duration: `0.1534` hours (`9.20` minutes)
- Numerical pass: `True`
- Source tensors: `443/443`
- Direct mappings: `203`
- Mamba mixers: `34/34` using `INIT-K-balanced-qkvo-lift`
- RoRoPE/BKV mixers: `6/6` using rank-448 activation PCA
- BF16 weights per TPU: `3.4513 GiB`
- BF16 Lion state per TPU: `3.4513 GiB`

| Context | Probe loss | Finite | Compile + execute |
|---:|---:|---:|---:|
| 8 | 15.9241 | True | 15.69 s |
| 32 | 19.6923 | True | 16.46 s |
| 128 | 18.6049 | True | 19.10 s |

All six retained attention layers use 512 calibration tokens and report the
expected 71.875% retained-attention cache reduction. Calibration reconstruction
relative L2 ranges from approximately 0.025 at layer 5 to 0.047 at layer 39.

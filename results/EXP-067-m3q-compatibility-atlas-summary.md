# EXP-067 — Qwen3-1.7B full-depth Mamba compatibility atlas

- Completed all `28/28` layers on TPU v5e-8 in `1.749` hours.
- Numerical and registered predictor gates passed.
- Step-1024 versus step-2048 difficulty ranking: Spearman `0.9978106`.
- Frozen 14.3% retained-attention proposal: layers `[0, 1, 20, 26]`.
- Legacy exact-dual preparation improved final decoder relative L2 strongly at layers 0–2 and 19–27, but generally lost at layers 3–18.
- The largest relative gains over canonical random occur at layers 24–27 (`94.29%`, `97.01%`, `97.87%`, and `94.85%`).
- Absolute post-recovery dual error, not relative gain, defines retained attention; layers 0 and 1 remain hardest despite positive transplant gains.
- This is a layer-allocation result, not yet full-model quality evidence.

Artifacts supplied by the user:

- Full SHA-256: `754b815bbd29d855acd3eda08ce6b269ae8d25c68f61edb932feae9a9edda503`
- Summary SHA-256: `acc7cda6c1ac063817cf8364e6fb2021688dff8a2a15f91100cba93df418f8f5`

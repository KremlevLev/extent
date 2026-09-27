# EXP-093 — foldable Mamba contribution calibration

Status: prepared, **not yet run on TPU**. Protocol frozen before seeing validation or test results.

## Question

EXP-091's all-parameter updates harmed the assembled model, and EXP-092 found no useful point on their straight global parameter path. Before another multi-billion-parameter update, ask a different, narrower question: are the 24 Mamba branches in the completed EXP-072 ONPOLICY hybrid merely mis-scaled *as a composed network*? Scale each Mamba branch by folding a gain into its existing output projection. This leaves the Qwen and Mamba feature-producing weights fixed and adds no inference-time operation or KV state.

## Frozen comparison

- Source: pinned Qwen3-1.7B-Base and the exact EXP-072-v2 ONPOLICY endpoints for seeds 123 and 456, including all 24 Mamba replacements and four unchanged GQA layers. EXP-069/072 checkpoint hashes must validate before use. No EXP-091 trained endpoint is used.
- Paired arms: `GLOBAL` learns one gain shared by all 24 Mamba layers; `PER-LAYER` learns 24 gains. Both start at gain 1.0 from identical model weights, see identical ordered text, optimize the same objective and receive exactly 2,048 updates per seed.
- Gain parameterization: `1 + 0.5*tanh(raw)`; FP32 raw coordinates initialized to zero; every effective gain stays in `(0.5, 1.5)`. The gain multiplies the Mamba `out_proj/kernel` before BF16 storage. It can be folded into that kernel for deployment. This is a gain-calibration comparison, not an equal-parameter-count comparison.
- Optimizer/objective: FP32 Adam at constant LR `0.01` on gain coordinates only; all 1.7B model weights are frozen. True-token causal cross-entropy on 2,048 unique length-256 WikiText-103 train windows beginning at offset 7,340,032. Four trajectories consume 2,097,152 student training tokens. No online Qwen teacher runs during optimization.
- Validation: 16 length-256 WikiText-103 validation windows beginning at offset 16,384, recorded every 128 steps for diagnostics. It does **not** select an endpoint or tune hyperparameters within this experiment.
- Locked test: 32 length-256 WikiText-103 test windows beginning at offset zero, evaluated only after the fixed 2,048-step endpoint. The unchanged start and both final arms are compared on the same windows.
- Primary gate: at **both seeds**, `PER-LAYER` final test NLL is at least `0.05` lower than `GLOBAL` and lower than the unchanged start. No one-seed win, best validation checkpoint, or later hyperparameter choice can satisfy this gate retrospectively.

## Execution and limits

Run `scripts.m3q_mixer_gain_campaign.main([])` once in the already configured Kaggle v5e-8 notebook. HF source preflight occurs before loading Qwen. The runner publishes a small result JSON and Markdown summary plus gain/Adam state every 128 steps to `experiments/exp093-foldable-mixer-gains/`; a fresh session restores the same protocol and code revision. It reserves 20 minutes of the 8.25-hour default wall-time cap for final persistence, and Telegram announces start, completion, deadline partial, or caught failure. The actual TPU runtime is unknown until execution; 8.25 hours is a ceiling, not a promised duration.

If the gate fails, do not turn this into another gain/lr sweep on the same locked text. If it passes, validate on fresh domains/seeds and compare with an equal-compute global-training baseline before claiming a general transplantation method. This experiment does not test 14B, MLA, long context, generation throughput, or recovered Qwen-level quality.

# EXP169 received failure — 3 November2026

Source: [received aggregate](EXP-169-received-session-1.json).

Both batch1 probes timed out: compiled607.556s and streamed608.095s,
returncode-15 after supervisor termination. No measured fit, no training arm
launched. The aggregate does not establish OOM, numerical failure, or a model
quality result. Corpus restore succeeded; reported available tmpfs176.09GB and
effective host RAM351.15GB do not diagnose per-chip TPU memory.

The supervisor's600-second timeout covers complete worker startup, source
reconstruction, compilation and measurement. It is not a steady-state update
timeout. Without worker logs, the stalled phase remains unknown. Do not
automatically label this as insufficient batch capacity or fix it solely by
increasing the timeout. Inspect probe-compiled-b1.log and probe-streamed-b1.log.

The run belongs to the old EXP168 random-prepared lineage. A clean QKVO TPU
experiment requires an independent original-Qwen loader and clean geometry;
changing the source filename in engine_tpu169 is not sufficient. PyTorch170
uses CUDA-specific execution and cannot simply be pasted into a TPU notebook.
No clean TPU runner or physical TPU compatibility result is claimed here.

Received worker tails subsequently confirm supervisor SIGTERM and show
unexpected torch_xla import in both JAX workers. The runner imports
Transformers AutoTokenizer after JAX device discovery; Transformers optional
backend discovery is a plausible import path, not a proved deadlock cause.
The environment helper now disables Transformers torch/TF/Flax backends for
these tokenizer-only workers. This does not disable explicit JAX/Flax imports.
Metric-server port and hugepages warnings alone do not establish a fatal error.
No Python-level stack or phase markers identify the exact stall; no physical
TPU rerun has verified the environment isolation fix.

# Singularity: JAX Mamba-3/MLA hybrid prototype

This repository contains a research bring-up path for a parameter-matched Qwen
14B-class hybrid: 36 Mamba-3 MIMO mixers, 12 MLA mixers, and the Qwen SwiGLU
MLPs. It targets one TPU v5e-8 slice, but all reference code and tests run on CPU.

## What is implemented

- A reusable Flax `Mamba3MIMO` block with data-dependent decay, the
  exponential-trapezoidal update, complex/rotary state channels, rank-4 MIMO,
  and an FP32 recurrent state.
- A reference MLA module whose configuration and projection names follow
  MaxText (`q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, and friends).
- A pre-norm hybrid decoder and causal LM head.
- BF16 parameters and gradients, with FP32 recurrence/softmax/logits for
  numerical stability.
- Lion with a single BF16 momentum buffer, optional gradient accumulation, full remat,
  and conservative `data/fsdp/tensor = 1/4/2` partitioning.
- Streaming-friendly Qwen mapping plans, SVD helpers for MLA, and an explicitly
  labelled heuristic Q/K/V -> Mamba input-projection transplant.

The `lax.scan` recurrence is the correctness implementation, not the final
throughput kernel. It is JIT/autodiff compatible on TPU. Before long runs,
replace `mamba3_reference_scan` with a chunked Pallas kernel and retain its
public tensor/parameter contract.

## Smoke test

```bash
python -m pytest -q tests/test_research_prototype.py
python -m scripts.smoke_prototype --sequence-length 8
```

The legacy `tests/test_qwen_base.py` is excluded from collection because it
downloads Qwen3-14B and runs work at import time. Treat it as a manual checkpoint
integration script, not an offline unit test.

On a fresh Kaggle TPU runtime:

```bash
pip install -r requirements-tpu.txt
python -m scripts.smoke_prototype --sequence-length 8
```

Start real bring-up with `config/hybrid_14b_v5e8.yaml`. Its sequence length is
deliberately 1024 and micro-batch size is one. Compile a single step and inspect
HBM before raising sequence length. The analytical model size is approximately
14.7B parameters; BF16 weights + BF16 gradients + one BF16 Lion moment consume
about 10.3 GiB per device under ideal eight-way sharding, before activations,
XLA temporaries, executable buffers, and unevenly replicated parameters.
`gradient_accumulation_steps` starts at one: Optax accumulation would add a
further full BF16 buffer (roughly 3.4 GiB/device for this model).

## MaxText boundary

`config/maxtext_mla_v5e8.yml` and `singularity.maxtext_adapter` expose the
upstream MaxText MLA configuration surface. Upstream MaxText does not currently
provide a Mamba-3 decoder block, so `singularity.model` owns layer scheduling.
This avoids claiming a drop-in MaxText model that cannot compile; the MLA names
are kept compatible so its optimized kernel can replace the reference attention
without another checkpoint conversion.

## Research caveats

The Qwen -> Mamba map is an initialization hypothesis, not exact functional
equivalence. Compare it against random mixer initialization, copied MLP/norms,
and MLA-only conversion under the same recovery-token budget. The tests here
cover contracts and differentiability; numerical parity with the official
CUDA Mamba-3 kernels still needs a dedicated cross-framework fixture.

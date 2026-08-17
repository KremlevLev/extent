# Singularity: JAX Mamba-3/MLA hybrid prototype

This repository contains a research bring-up path for a parameter-matched Qwen
14B-class hybrid: 41 Mamba-3 MIMO mixers, 7 MLA mixers, and the Qwen SwiGLU
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

To verify real multi-device parameter and optimizer sharding on two GPUs:

```bash
python -m scripts.sharded_smoke --sequence-length 8 --steps 1
```

For a two-device `tensor=2` mesh, the tiny model should report approximately
`partitioned_param_arrays=20/47` and
`partitioned_train_state_arrays=40/97`. The second number includes the sharded
Lion momentum. Exact loss can vary slightly between CUDA and TPU.
The step also reports `grads_finite` and `nonfinite_grad_leaves`. Gradient norm
and clipping use FP32 reductions even though stored gradients remain BF16. A
non-finite step is reported and skipped instead of corrupting parameters.
When CUDA reports non-finite gradients, the launcher recompiles a diagnostic
backward and prints the exact parameter paths. On pre-Ampere GPUs, compare with
`--compute-dtype float32`; parameters and stored gradients remain BF16.
The sharded launcher defaults to `--compute-dtype auto`: it selects FP32 compute
on T4/V100-class GPUs and BF16 on TPU v5e or BF16-capable modern GPUs. Passing
`--compute-dtype bfloat16` explicitly keeps the failure-reproduction path.

Do not pass `--config config/hybrid_14b_v5e8.yaml --allow-full-model` on ordinary
two-GPU Kaggle instances: the ideal BF16 weights + gradients + Lion state alone
would require about 41 GiB per GPU. The full configuration is intended for eight
devices and should still begin with a zero/one-step HBM bring-up.

Before reserving v5e-8, run the exact shape-only audit on any machine (including
v5e-1). It traces all parameter shapes but allocates no 14B arrays:

```python
from scripts.full_model_preflight import main as full_preflight
full_preflight([])
```

Only on v5e-8, after reviewing the report, initialize parameter shards with:

```python
full_preflight(["--initialize-params"])
```

The guarded initialization refuses to run unless the actual eight-device mesh
matches `data=1, fsdp=4, tensor=2`. It creates weights only—not Lion state or a
full training step.

After parameter initialization passes, restart with a fresh TPU runtime and
probe the real persistent allocation for weights plus Lion momentum:

```python
from scripts.full_model_preflight import main as full_preflight
full_preflight(["--initialize-optimizer"])
```

This still does not run forward/backward and does not allocate gradients or
activations. It prints parameter, optimizer, and combined bytes for every TPU.
Run it in the notebook kernel; do not use `!python`, `%run`, or a subprocess
after JAX has initialized libtpu.

On a fresh Kaggle TPU runtime:

```bash
pip install -r requirements-tpu.txt
python -m scripts.smoke_prototype --sequence-length 8
```

In Google Colab, libtpu is exclusive to one Python process. After importing JAX
in the notebook kernel, do not launch `!python` or `!pytest`. Run validation in
that same process instead:

```python
import pytest
pytest.main(["-q"])

from scripts.sharded_smoke import main as sharded_smoke
sharded_smoke(["--sequence-length", "8", "--steps", "1"])
```

## Kaggle Telegram TPU notification

Create a bot with Telegram's `@BotFather`, open the bot, press **Start**, and
send it any message. Store the token in Kaggle **Add-ons > Secrets** as
`TELEGRAM_BOT_TOKEN`; never paste it into a notebook or commit it.

To discover the destination chat id, enable notebook access for that secret and
run in the notebook process:

```python
from scripts.telegram_tpu_notifier import main as telegram_tpu_notifier
telegram_tpu_notifier(["--show-chat-ids"])
```

Store the printed id as a second Kaggle secret named `TELEGRAM_CHAT_ID`. On the
next fresh TPU session, after installing requirements, run:

```python
from scripts.telegram_tpu_notifier import main as telegram_tpu_notifier

NOTIFIER_ENABLED = True
telegram_tpu_notifier(
    ["--message", "experiment runtime initialized"],
    enabled=NOTIFIER_ENABLED,
)
```

The notifier reports whichever runtime JAX actually sees: TPU, GPU, or CPU. If
JAX initialization itself fails, it attempts to send that failure too. Set
`NOTIFIER_ENABLED = False` to disable all checks and network calls. It cannot
notify before Kaggle starts the notebook kernel, because no project code is
running before that point. Do not launch it through `!python` or `%run`;
libtpu must stay in the notebook process.

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

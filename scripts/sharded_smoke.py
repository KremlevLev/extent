from __future__ import annotations

import argparse
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from extent import HybridForCausalLM, tiny_config
from extent.config import load_config
from extent.estimate import parameter_count, training_state_gib
from extent.hardware import recommended_compute_dtype
from extent.optimizer import create_lion
from extent.sharding import count_partitioned_arrays, create_v5e_mesh
from extent.train_step import initialize_sharded_runtime, shard_host_batch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Real multi-device init + training smoke test.")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--config", type=str, default=None, help="YAML model config; default is the tiny model.")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "bfloat16", "float32", "float16"),
        default="auto",
        help="Override activation compute dtype without changing BF16 parameter/gradient storage.",
    )
    parser.add_argument(
        "--allow-full-model",
        action="store_true",
        help="Required with --config because full initialization can exhaust accelerator memory.",
    )
    args = parser.parse_args(argv)
    if args.config and not args.allow_full_model:
        parser.error("--config requires --allow-full-model; run the tiny distributed smoke test first")

    config = load_config(args.config)[0] if args.config else tiny_config()
    dtype_decision = recommended_compute_dtype(requested=args.compute_dtype)
    config = replace(config, compute_dtype=dtype_decision.dtype)
    mesh = create_v5e_mesh()
    counts = parameter_count(config)
    ideal_per_device = training_state_gib(config) / mesh.size
    print(f"devices={jax.devices()}")
    print(f"mesh={dict(mesh.shape)}")
    print(f"compute_dtype={config.compute_dtype} reason={dtype_decision.reason}")
    print(f"parameters={counts['total']:,} ideal_weight_grad_lion_gib_per_device={ideal_per_device:.3f}")

    batch_size = mesh.shape["data"]
    tokens = np.arange(batch_size * args.sequence_length, dtype=np.int32).reshape(batch_size, args.sequence_length)
    tokens %= config.vocab_size
    host_batch = {
        "input_ids": tokens,
        "labels": tokens.copy(),
        "attention_mask": np.ones_like(tokens, dtype=np.bool_),
        "loss_mask": np.ones_like(tokens, dtype=np.bool_),
    }
    batch = shard_host_batch(host_batch, mesh)
    model = HybridForCausalLM(config)
    tx = create_lion(total_steps=max(args.steps, 10), warmup_steps=1, accumulation_steps=1)
    runtime = initialize_sharded_runtime(model, tx, jax.random.key(0), batch, mesh)
    partitioned_params, total_params = count_partitioned_arrays(runtime.state.params)
    partitioned_state, total_state = count_partitioned_arrays(runtime.state)
    print(f"partitioned_param_arrays={partitioned_params}/{total_params}")
    print(f"partitioned_train_state_arrays={partitioned_state}/{total_state}")

    state = runtime.state
    for step in range(args.steps):
        state, metrics = runtime.train_step(state, batch)
        jax.block_until_ready(metrics)
        print(
            f"step={step} loss={float(metrics['loss']):.4f} "
            f"grad_norm={float(metrics['grad_norm']):.4f} "
            f"max_abs_grad={float(metrics['max_abs_grad']):.4f} "
            f"grads_finite={bool(metrics['grads_finite'])} "
            f"nonfinite_grad_leaves={int(metrics['nonfinite_grad_leaves'])}"
        )
        if not bool(metrics["grads_finite"]):
            print("nonfinite gradient parameters:")
            _, diagnostics = runtime.diagnose_gradients(state.params, batch)
            jax.block_until_ready(diagnostics)
            finite = traverse_util.flatten_dict(diagnostics["finite"])
            max_abs = traverse_util.flatten_dict(diagnostics["max_abs"])
            for path, is_finite in finite.items():
                if not bool(is_finite):
                    print(f"  {'/'.join(path)} finite_max_abs={float(max_abs[path]):.6g}")
            break


if __name__ == "__main__":
    main()

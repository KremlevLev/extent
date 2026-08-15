from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from singularity import HybridForCausalLM, tiny_config
from singularity.estimate import parameter_count, training_state_gib
from singularity.optimizer import create_lion
from singularity.sharding import create_v5e_mesh
from singularity.train_step import create_train_state, make_train_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a tiny hybrid forward/backward step.")
    parser.add_argument("--sequence-length", type=int, default=8)
    args = parser.parse_args()
    config = tiny_config()
    model = HybridForCausalLM(config)
    tokens = (jnp.arange(args.sequence_length)[None, :] % config.vocab_size).astype(jnp.int32)
    variables = model.init(jax.random.key(0), tokens)
    tx = create_lion(total_steps=10, warmup_steps=1, accumulation_steps=1)
    state = create_train_state(model, variables["params"], tx)
    state, metrics = make_train_step()(state, {"input_ids": tokens})
    counts = parameter_count(config)
    print(f"devices={jax.devices()}")
    print(f"mesh={create_v5e_mesh().shape}")
    print(f"logits={model.apply({'params': state.params}, tokens).shape}")
    print(
        f"loss={float(metrics['loss']):.4f} grad_norm={float(metrics['grad_norm']):.4f} "
        f"grads_finite={bool(metrics['grads_finite'])}"
    )
    print(f"tiny_params={counts['total']:,} training_state_gib={training_state_gib(config):.4f}")


if __name__ == "__main__":
    main()

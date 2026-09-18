"""EXP-088 engineering preflight only: synthetic targets, no scientific recovery claim."""
import argparse
from pathlib import Path
import time
import jax
import jax.numpy as jnp

from extent import HybridForCausalLM, tiny_config
from extent.config import load_config
from extent.campaign_checkpoint import write_json_atomic
from extent.downstream_recovery import HybridDecoderSuffix, suffix_parameters, make_downstream_recovery_step, layout_stable_downstream_step
from extent.initialization import initialize_sharded_parameters, initialize_sharded_optimizer_state
from extent.optimizer import create_lion
from extent.sharding import create_v5e_mesh, batch_sharding, replicated_sharding
from scripts.qwen_extended_horizon_campaign import _safe_notify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    _safe_notify(args.telegram, "Extent EXP-088 downstream engineering preflight started")
    if args.steps < 1 or not 2 <= args.sequence_length <= 128:
        raise ValueError("preflight requires positive steps and context 2-128")
    config = tiny_config() if args.tiny else load_config(
        Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml"
    )[0]
    mesh = create_v5e_mesh()
    model = HybridForCausalLM(config)
    tokens = jax.device_put(jnp.arange(args.sequence_length)[None] % config.vocab_size,
                            batch_sharding(mesh))
    params = initialize_sharded_parameters(model, jax.random.key(88), tokens, mesh).params
    pair = tuple(config.mamba_layer_indices[:2])
    first = pair[0]
    if first != 0:
        raise ValueError("initial engineering probe requires first Mamba layer 0")
    hidden = jnp.take(params["embed_tokens"]["embedding"], tokens, axis=0)
    frozen = suffix_parameters(params, first)
    candidate = {f"layers_{index}": params[f"layers_{index}"]["mamba"] for index in pair}
    suffix = HybridDecoderSuffix(config, first)
    tx = create_lion(learning_rate=1e-5, total_steps=10, warmup_steps=1)
    candidate_layout = jax.tree.map(lambda value: value.sharding, candidate)
    state = initialize_sharded_optimizer_state(
        tx, candidate, candidate, candidate_layout, mesh,
    ).opt_state
    # Deliberately synthetic nonzero-gradient target; never treated as experiment evidence.
    targets = jax.device_put(
        jnp.zeros((1, args.sequence_length, config.vocab_size), jnp.float32),
        replicated_sharding(mesh),
    )
    step = layout_stable_downstream_step(
        make_downstream_recovery_step(suffix, tx),
        candidate, state, frozen, hidden, targets,
    )
    result = {"protocol": "exp088-downstream-engineering-preflight-v1",
              "scientific_experiment": False, "status": "running", "steps": [],
              "pair": list(pair), "sequence_length": args.sequence_length,
              "devices": [str(x) for x in jax.devices()]}
    path = Path(args.output_dir) / "extent-m3q-downstream-preflight.json"
    write_json_atomic(path, result)
    try:
        started = time.monotonic()
        compiled = step.lower(candidate, state, frozen, hidden, targets).compile()
        result["compile_seconds"] = time.monotonic() - started
        memory = compiled.memory_analysis()
        result["compiler_memory"] = {
            key: int(getattr(memory, key)) for key in (
                "argument_size_in_bytes", "output_size_in_bytes",
                "temp_size_in_bytes", "alias_size_in_bytes",
            ) if memory is not None and hasattr(memory, key)
        }
        write_json_atomic(path, result)
        for index in range(args.steps):
            started = time.monotonic()
            candidate, state, metrics = compiled(candidate, state, frozen, hidden, targets)
            jax.block_until_ready(metrics)
            row = {"step": index, "seconds": time.monotonic() - started,
                   "loss": float(metrics["loss"]), "grad_norm": float(metrics["grad_norm"]),
                   "grads_finite": bool(metrics["grads_finite"])}
            result["steps"].append(row)
            write_json_atomic(path, result)
            print(row, flush=True)
            if not row["grads_finite"]:
                raise FloatingPointError("nonfinite downstream gradient")
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json_atomic(path, result)
        print(f"result_json={path}", flush=True)
        _safe_notify(args.telegram, f"Extent EXP-088 preflight {result['status']}")
    return result


if __name__ == "__main__":
    main()

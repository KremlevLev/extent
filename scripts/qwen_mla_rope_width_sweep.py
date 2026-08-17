from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np

from scripts.qwen_mla_shock import _run_layer
from singularity.layers.common import RMSNorm
from singularity.layers.mla import MultiHeadLatentAttention
from singularity.mla_conversion import (
    convert_qwen3_gqa_to_mla_joint_svd,
    factorize_qwen3_joint_kv,
    qwen3_mla_conversion_config,
)
from singularity.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_attention_params,
    load_layer_arrays,
    parity_metrics,
)
from singularity.qwen3_teacher import Qwen3GQAAttention, Qwen3TeacherConfig
from singularity.qwen_source import QWEN3_14B, validate_source_metadata
from singularity.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_widths(value: str) -> tuple[int, ...]:
    widths = tuple(sorted({int(item) for item in value.split(",")}))
    if not widths or widths[0] < 2 or any(width % 2 for width in widths):
        raise ValueError("RoPE widths must be positive even comma-separated integers")
    return widths


def _measure(reference, candidate, tail_tokens: int) -> dict:
    tail = min(tail_tokens, reference.shape[1])
    return {
        "all_tokens": asdict(parity_metrics(reference, candidate)),
        "tail_tokens": asdict(
            parity_metrics(reference[:, -tail:], candidate[:, -tail:])
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep partial-RoPE width at a fixed shared-MLA cache budget."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--rope-widths", default="32,64,96,128")
    parser.add_argument("--target-cache-elements", type=int, default=576)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--svd-seed", type=int, default=0)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)
    widths = _parse_widths(args.rope_widths)

    spec = QWEN3_14B
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32", compute_dtype="float32", remat_policy="none"
    )
    if widths[-1] > source.head_dim:
        raise ValueError("RoPE width exceeds the source head dimension")
    if args.sequence_length > source.max_position_embeddings:
        raise ValueError("sequence length exceeds the pinned Qwen3 context")
    target_ranks = {
        width: args.target_cache_elements - width for width in widths
    }
    if min(target_ranks.values()) < 1:
        raise ValueError("fixed cache budget leaves no positive KV latent rank")

    model_dir, shards = ensure_layer_checkpoint(
        args.cache_dir,
        index_payload["weight_map"],
        source,
        args.layer_index,
        repo_id=spec.repo_id,
        revision=spec.revision,
    )
    arrays = load_layer_arrays(
        QwenCheckpointReader(model_dir), source, args.layer_index
    )
    rng = np.random.default_rng(args.seed)
    hidden = jnp.asarray(
        rng.standard_normal(
            (1, args.sequence_length, source.hidden_size), dtype=np.float32
        )
    )
    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None, :]
    mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
    input_scale = jnp.asarray(
        arrays[f"model.layers.{args.layer_index}.input_layernorm.weight"],
        dtype=jnp.float32,
    )
    normalized = RMSNorm(
        source.hidden_size, source.rms_norm_eps, jnp.float32
    ).apply({"params": {"scale": input_scale}}, hidden)
    reference = _run_layer(
        Qwen3GQAAttention(source),
        jax_attention_params(arrays, source, args.layer_index),
        normalized,
        positions,
        mask,
    )
    print(
        f"source={spec.repo_id}@{spec.revision} layer={args.layer_index} "
        f"shards={list(shards)} length={args.sequence_length} "
        f"target_cache={args.target_cache_elements}"
    )

    variants = {}
    factorization_seconds = {}
    for width in widths:
        maximum_rank = source.num_key_value_heads * (
            2 * source.head_dim - width
        )
        started = time.perf_counter()
        factors = factorize_qwen3_joint_kv(
            arrays,
            source,
            args.layer_index,
            rope_dim=width,
            max_rank=maximum_rank,
            seed=args.svd_seed,
        )
        elapsed = time.perf_counter() - started
        factorization_seconds[str(width)] = elapsed
        print(f"rope_{width}_factorization=DONE seconds={elapsed:.2f}")

        configurations = (
            ("grouped_no_svd_loss", maximum_rank, True),
            ("shared_no_svd_loss", maximum_rank, False),
            ("shared_fixed_cache", target_ranks[width], False),
        )
        for label, rank, grouped_rope in configurations:
            mla = qwen3_mla_conversion_config(
                source,
                kv_lora_rank=rank,
                rope_dim=width,
                grouped_rope=grouped_rope,
            )
            params, report = convert_qwen3_gqa_to_mla_joint_svd(
                arrays,
                source,
                mla,
                args.layer_index,
                seed=args.svd_seed,
                factors=factors,
            )
            candidate = _run_layer(
                MultiHeadLatentAttention(
                    source.hidden_size,
                    mla,
                    dtype=jnp.float32,
                    param_dtype=jnp.float32,
                ),
                params,
                normalized,
                positions,
                mask,
            )
            name = f"rope_{width}_{label}_rank_{rank}"
            variants[name] = {
                "conversion": asdict(report),
                "mixer_output": _measure(
                    reference, candidate, args.tail_tokens
                ),
            }
            print(
                f"{name}=DONE all_l2="
                f"{variants[name]['mixer_output']['all_tokens']['relative_l2']:.6g} "
                f"tail_l2="
                f"{variants[name]['mixer_output']['tail_tokens']['relative_l2']:.6g}"
            )

    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "layer_index": args.layer_index,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "svd_seed": args.svd_seed,
        "compute_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "input_kind": "deterministic_gaussian",
        "rope_widths": list(widths),
        "target_cache_elements": args.target_cache_elements,
        "target_ranks": {str(key): value for key, value in target_ranks.items()},
        "tail_tokens": args.tail_tokens,
        "factorization_seconds": factorization_seconds,
        "variants": variants,
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"result_json={output.resolve()}")


if __name__ == "__main__":
    main()

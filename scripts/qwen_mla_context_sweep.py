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
from extent.layers.common import RMSNorm
from extent.layers.mla import MultiHeadLatentAttention
from extent.mla_conversion import (
    convert_qwen3_gqa_to_mla_joint_svd,
    factorize_qwen3_joint_kv,
    qwen3_mla_conversion_config,
)
from extent.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_attention_params,
    load_layer_arrays,
    parity_metrics,
)
from extent.qwen3_teacher import Qwen3GQAAttention, Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(sorted({int(item) for item in value.split(",")}))
    if not lengths or lengths[0] < 1:
        raise ValueError("lengths must be positive comma-separated integers")
    return lengths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure MLA mixer shock as context positions grow."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--lengths", default="4,64,256,1024")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--svd-seed", type=int, default=0)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--compressed-rank", type=int, default=512)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)
    lengths = _parse_lengths(args.lengths)

    spec = QWEN3_14B
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32", compute_dtype="float32", remat_policy="none"
    )
    if lengths[-1] > source.max_position_embeddings:
        raise ValueError("requested length exceeds the pinned Qwen3 context")
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
    maximum_joint_rank = source.num_key_value_heads * (
        2 * source.head_dim - args.rope_dim
    )
    if args.compressed_rank >= maximum_joint_rank:
        raise ValueError("compressed rank must be below the no-SVD-loss control rank")
    started = time.perf_counter()
    factors = factorize_qwen3_joint_kv(
        arrays,
        source,
        args.layer_index,
        rope_dim=args.rope_dim,
        max_rank=maximum_joint_rank,
        seed=args.svd_seed,
    )
    factorization_seconds = time.perf_counter() - started

    rng = np.random.default_rng(args.seed)
    hidden_prefix = jnp.asarray(
        rng.standard_normal(
            (1, lengths[-1], source.hidden_size), dtype=np.float32
        )
    )
    teacher_attention_params = jax_attention_params(
        arrays, source, args.layer_index
    )
    input_norm_scale = jnp.asarray(
        arrays[
            f"model.layers.{args.layer_index}.input_layernorm.weight"
        ],
        dtype=jnp.float32,
    )
    input_norm = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)
    teacher_attention = Qwen3GQAAttention(source)
    references = {}
    normalized_inputs = {}
    print(
        f"source={spec.repo_id}@{spec.revision} layer={args.layer_index} "
        f"shards={list(shards)} lengths={list(lengths)}"
    )
    print(f"max_rank_factorization=DONE seconds={factorization_seconds:.2f}")
    for length in lengths:
        hidden = hidden_prefix[:, :length]
        positions = jnp.arange(length, dtype=jnp.int32)[None, :]
        mask = jnp.ones((1, length), dtype=jnp.bool_)
        normalized = input_norm.apply(
            {
                "params": {
                    "scale": input_norm_scale
                }
            },
            hidden,
        )
        normalized_inputs[length] = normalized
        references[length] = _run_layer(
            teacher_attention,
            teacher_attention_params,
            normalized,
            positions,
            mask,
        )
        print(f"teacher_length_{length}=DONE")

    variants = {}
    for rank in (args.compressed_rank, maximum_joint_rank):
        for grouped_rope in (True, False):
            mla = qwen3_mla_conversion_config(
                source,
                kv_lora_rank=rank,
                rope_dim=args.rope_dim,
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
            module = MultiHeadLatentAttention(
                source.hidden_size,
                mla,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
            )
            name = f"rank_{rank}_{'grouped' if grouped_rope else 'shared'}_rope"
            measurements = {}
            for length in lengths:
                positions = jnp.arange(length, dtype=jnp.int32)[None, :]
                mask = jnp.ones((1, length), dtype=jnp.bool_)
                candidate = _run_layer(
                    module,
                    params,
                    normalized_inputs[length],
                    positions,
                    mask,
                )
                tail = min(args.tail_tokens, length)
                measurements[str(length)] = {
                    "all_tokens": asdict(
                        parity_metrics(references[length], candidate)
                    ),
                    "tail_tokens": asdict(
                        parity_metrics(
                            references[length][:, -tail:], candidate[:, -tail:]
                        )
                    ),
                }
                print(
                    f"{name}_length_{length}=DONE all_l2="
                    f"{measurements[str(length)]['all_tokens']['relative_l2']:.6g} "
                    f"tail_l2="
                    f"{measurements[str(length)]['tail_tokens']['relative_l2']:.6g}"
                )
            variants[name] = {
                "conversion": asdict(report),
                "lengths": measurements,
            }

    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "layer_index": args.layer_index,
        "seed": args.seed,
        "svd_seed": args.svd_seed,
        "compute_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "input_kind": "shared_prefix_deterministic_gaussian",
        "lengths": list(lengths),
        "tail_tokens": args.tail_tokens,
        "rope_dim": args.rope_dim,
        "compressed_rank": args.compressed_rank,
        "control_rank": maximum_joint_rank,
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

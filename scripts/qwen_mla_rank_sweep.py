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

from scripts.qwen_mla_shock import _hybrid_config, _run_layer
from singularity.layers.common import RMSNorm
from singularity.layers.mla import MultiHeadLatentAttention
from singularity.mla_conversion import (
    convert_qwen3_gqa_to_mla_joint_svd,
    factorize_qwen3_joint_kv,
    qwen3_decoder_common_params,
    qwen3_mla_conversion_config,
)
from singularity.model import HybridDecoderLayer
from singularity.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_layer_params,
    load_layer_arrays,
    parity_metrics,
)
from singularity.qwen3_teacher import (
    Qwen3DecoderLayer,
    Qwen3GQAAttention,
    Qwen3TeacherConfig,
)
from singularity.qwen_source import QWEN3_14B, validate_source_metadata
from singularity.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_ranks(value: str) -> tuple[int, ...]:
    ranks = tuple(sorted({int(item) for item in value.split(",")}))
    if not ranks or ranks[0] < 1:
        raise ValueError("ranks must be positive comma-separated integers")
    return ranks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Qwen3 GQA-to-MLA KV ranks with one reusable SVD."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--svd-seed", type=int, default=0)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--ranks", default="256,512,1024,1536")
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)
    ranks = _parse_ranks(args.ranks)

    spec = QWEN3_14B
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32", compute_dtype="float32", remat_policy="none"
    )
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
    attention_mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
    teacher_params = jax_layer_params(arrays, source, args.layer_index)
    teacher_output = _run_layer(
        Qwen3DecoderLayer(source),
        teacher_params,
        hidden,
        positions,
        attention_mask,
    )
    normalized_hidden = RMSNorm(
        source.hidden_size, source.rms_norm_eps, jnp.float32
    ).apply(
        {"params": {"scale": teacher_params["input_layernorm"]["scale"]}},
        hidden,
    )
    teacher_mixer = _run_layer(
        Qwen3GQAAttention(source),
        teacher_params["self_attn"],
        normalized_hidden,
        positions,
        attention_mask,
    )

    print(
        f"source={spec.repo_id}@{spec.revision} layer={args.layer_index} "
        f"shards={list(shards)} ranks={list(ranks)}"
    )
    started = time.perf_counter()
    factors = factorize_qwen3_joint_kv(
        arrays,
        source,
        args.layer_index,
        rope_dim=args.rope_dim,
        max_rank=max(ranks),
        seed=args.svd_seed,
    )
    factorization_seconds = time.perf_counter() - started
    print(f"max_rank_factorization=DONE seconds={factorization_seconds:.2f}")

    common = qwen3_decoder_common_params(arrays, args.layer_index)
    variants = {}
    for rank in ranks:
        for grouped_rope in (True, False):
            mla = qwen3_mla_conversion_config(
                source,
                kv_lora_rank=rank,
                rope_dim=args.rope_dim,
                grouped_rope=grouped_rope,
            )
            attention_params, report = convert_qwen3_gqa_to_mla_joint_svd(
                arrays,
                source,
                mla,
                args.layer_index,
                seed=args.svd_seed,
                factors=factors,
            )
            decoder_output = _run_layer(
                HybridDecoderLayer(_hybrid_config(source, mla), 0),
                {**common, "self_attn": attention_params},
                hidden,
                positions,
                attention_mask,
            )
            mixer_output = _run_layer(
                MultiHeadLatentAttention(
                    source.hidden_size,
                    mla,
                    dtype=jnp.float32,
                    param_dtype=jnp.float32,
                ),
                attention_params,
                normalized_hidden,
                positions,
                attention_mask,
            )
            name = f"rank_{rank}_{'grouped' if grouped_rope else 'shared'}_rope"
            variants[name] = {
                "conversion": asdict(report),
                "mixer_output": asdict(parity_metrics(teacher_mixer, mixer_output)),
                "decoder_output": asdict(
                    parity_metrics(teacher_output, decoder_output)
                ),
            }
            print(
                f"{name}=DONE reconstruction_l2="
                f"{report.joint_reconstruction_relative_l2:.6g} mixer_l2="
                f"{variants[name]['mixer_output']['relative_l2']:.6g}"
            )

    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "layer_index": args.layer_index,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "svd_seed": args.svd_seed,
        "compute_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "rope_dim": args.rope_dim,
        "ranks": list(ranks),
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

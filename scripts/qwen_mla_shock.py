from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from singularity.config import HybridConfig
from singularity.mla_conversion import (
    convert_qwen3_gqa_to_mla_joint_svd,
    qwen3_decoder_common_params,
    qwen3_mla_conversion_config,
)
from singularity.layers.common import RMSNorm
from singularity.layers.mla import MultiHeadLatentAttention
from singularity.model import HybridDecoderLayer
from singularity.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_layer_params,
    load_layer_arrays,
    parity_metrics,
    required_layer_shards,
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


def _hybrid_config(source: Qwen3TeacherConfig, mla) -> HybridConfig:
    return HybridConfig(
        vocab_size=source.vocab_size,
        hidden_size=source.hidden_size,
        intermediate_size=source.intermediate_size,
        num_layers=1,
        attention_layer_indices=(0,),
        rms_norm_eps=source.rms_norm_eps,
        max_position_embeddings=source.max_position_embeddings,
        param_dtype="float32",
        compute_dtype="float32",
        logits_dtype="float32",
        remat_policy="none",
        mla=mla,
    )


def _run_layer(layer, params, hidden, positions, attention_mask) -> np.ndarray:
    output = jax.jit(layer.apply)(
        {"params": params}, hidden, positions, attention_mask
    )
    jax.block_until_ready(output)
    return np.asarray(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure immediate Qwen3 GQA-to-MLA decoder-layer shock."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--kv-rank", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--svd-seed", type=int, default=0)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)

    spec = QWEN3_14B
    source_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(source_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32", compute_dtype="float32", remat_policy="none"
    )
    shards = required_layer_shards(
        index_payload["weight_map"], source, args.layer_index
    )
    model_dir, _ = ensure_layer_checkpoint(
        args.cache_dir,
        index_payload["weight_map"],
        source,
        args.layer_index,
        repo_id=spec.repo_id,
        revision=spec.revision,
    )
    print(
        f"source={spec.repo_id}@{spec.revision} layer={args.layer_index} "
        f"shards={list(shards)}"
    )
    arrays = load_layer_arrays(
        QwenCheckpointReader(model_dir), source, args.layer_index
    )
    rng = np.random.default_rng(args.seed)
    hidden_np = rng.standard_normal(
        (1, args.sequence_length, source.hidden_size), dtype=np.float32
    )
    hidden = jnp.asarray(hidden_np)
    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None, :]
    attention_mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)

    teacher_params = jax_layer_params(arrays, source, args.layer_index)
    teacher = Qwen3DecoderLayer(source)
    teacher_output = _run_layer(
        teacher,
        teacher_params,
        hidden,
        positions,
        attention_mask,
    )
    normalized_hidden = RMSNorm(
        source.hidden_size,
        source.rms_norm_eps,
        jnp.float32,
    ).apply(
        {
            "params": {
                "scale": teacher_params["input_layernorm"]["scale"]
            }
        },
        hidden,
    )
    teacher_mixer_output = _run_layer(
        Qwen3GQAAttention(source),
        teacher_params["self_attn"],
        normalized_hidden,
        positions,
        attention_mask,
    )
    common = qwen3_decoder_common_params(arrays, args.layer_index)
    variants = {}
    reports = {}
    converted_params = {}
    for grouped_rope in (True, False):
        mla = qwen3_mla_conversion_config(
            source,
            kv_lora_rank=args.kv_rank,
            rope_dim=args.rope_dim,
            grouped_rope=grouped_rope,
        )
        attention_params, report = convert_qwen3_gqa_to_mla_joint_svd(
            arrays, source, mla, args.layer_index, seed=args.svd_seed
        )
        name = report.method
        params = {**common, "self_attn": attention_params}
        layer = HybridDecoderLayer(_hybrid_config(source, mla), 0)
        output = _run_layer(
            layer, freeze(params), hidden, positions, attention_mask
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
        variants[name] = {
            "mixer_output": asdict(
                parity_metrics(teacher_mixer_output, mixer_output)
            ),
            "decoder_output": asdict(parity_metrics(teacher_output, output)),
        }
        reports[name] = asdict(report)
        converted_params[name] = attention_params
        print(
            f"{name}=DONE mixer_relative_l2="
            f"{variants[name]['mixer_output']['relative_l2']:.6g} "
            f"decoder_relative_l2={variants[name]['decoder_output']['relative_l2']:.6g}"
        )

    target_mla = qwen3_mla_conversion_config(
        source,
        kv_lora_rank=args.kv_rank,
        rope_dim=args.rope_dim,
        grouped_rope=False,
    )
    target_layer = HybridDecoderLayer(_hybrid_config(source, target_mla), 0)
    random_params = unfreeze(
        target_layer.init(
            jax.random.key(args.seed), hidden, positions, attention_mask
        )["params"]
    )
    random_params.update(common)
    random_output = _run_layer(
        target_layer, freeze(random_params), hidden, positions, attention_mask
    )
    random_mixer_output = _run_layer(
        MultiHeadLatentAttention(
            source.hidden_size,
            target_mla,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        ),
        random_params["self_attn"],
        normalized_hidden,
        positions,
        attention_mask,
    )
    variants["random_mla"] = {
        "mixer_output": asdict(
            parity_metrics(teacher_mixer_output, random_mixer_output)
        ),
        "decoder_output": asdict(parity_metrics(teacher_output, random_output)),
    }

    projection_params = unfreeze(random_params)
    joint_params = converted_params["joint_svd_shared_rope"]
    for name in ("query_proj", "out_proj", "q_norm", "k_norm"):
        projection_params["self_attn"][name] = unfreeze(joint_params[name])
    projection_output = _run_layer(
        target_layer, freeze(projection_params), hidden, positions, attention_mask
    )
    projection_mixer_output = _run_layer(
        MultiHeadLatentAttention(
            source.hidden_size,
            target_mla,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        ),
        projection_params["self_attn"],
        normalized_hidden,
        positions,
        attention_mask,
    )
    variants["projection_copy_random_kv"] = {
        "mixer_output": asdict(
            parity_metrics(teacher_mixer_output, projection_mixer_output)
        ),
        "decoder_output": asdict(
            parity_metrics(teacher_output, projection_output)
        ),
    }
    print(
        "random_mla=DONE "
        "mixer_relative_l2="
        f"{variants['random_mla']['mixer_output']['relative_l2']:.6g}"
    )
    print(
        "projection_copy_random_kv=DONE "
        "mixer_relative_l2="
        f"{variants['projection_copy_random_kv']['mixer_output']['relative_l2']:.6g}"
    )

    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "layer_index": args.layer_index,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "svd_seed": args.svd_seed,
        "compute_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "kv_rank": args.kv_rank,
        "rope_dim": args.rope_dim,
        "variants": variants,
        "conversion_reports": reports,
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"result_json={output.resolve()}")


if __name__ == "__main__":
    main()

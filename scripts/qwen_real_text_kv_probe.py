from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np

from scripts.qwen_balanced_kv_probe import _attend_components, _relative_l2
from scripts.qwen_mla_shock import _run_layer
from scripts.qwen_rorope_shock import _project_qkv
from singularity.activation_compression import (
    bkv_balance_ratio,
    fit_activation_pca_jax,
)
from singularity.calibration_data import (
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
    load_wikitext2_tokens,
)
from singularity.layers.common import RMSNorm
from singularity.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_attention_params,
    load_layer_arrays,
    parity_metrics,
)
from singularity.qwen3_teacher import Qwen3GQAAttention, Qwen3TeacherConfig
from singularity.qwen_source import QWEN3_14B, validate_source_metadata
from singularity.rorope import (
    apply_rorope,
    fit_rorope_rotations,
    rotate_rorope_key,
)
from singularity.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Repeat the balanced KV probe on pinned real-text token embeddings."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/singularity-calibration-cache"
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--calibration-tokens", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--latent-rank", type=int, default=448)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--svd-seed", type=int, default=0)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)

    spec = QWEN3_14B
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32", compute_dtype="float32", remat_policy="none"
    )
    total_tokens = args.calibration_tokens + args.sequence_length
    if args.calibration_tokens <= args.latent_rank:
        raise ValueError("calibration token count must exceed the latent rank")
    if args.sequence_length > source.max_position_embeddings:
        raise ValueError("evaluation length exceeds the pinned Qwen3 context")
    tokens = load_wikitext2_tokens(
        total_tokens,
        args.dataset_cache_dir,
        tokenizer_repo=spec.repo_id,
        tokenizer_revision=spec.revision,
    )
    model_dir, shards = ensure_layer_checkpoint(
        args.cache_dir,
        index_payload["weight_map"],
        source,
        args.layer_index,
        repo_id=spec.repo_id,
        revision=spec.revision,
    )
    embedding_shard = index_payload["weight_map"]["model.embed_tokens.weight"]
    if embedding_shard not in shards:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            filename=embedding_shard,
            local_dir=model_dir,
        )
    reader = QwenCheckpointReader(model_dir)
    arrays = load_layer_arrays(reader, source, args.layer_index)
    hidden = reader.read_rows("model.embed_tokens.weight", tokens)
    calibration_hidden = jnp.asarray(hidden[: args.calibration_tokens][None])
    evaluation_hidden = jnp.asarray(hidden[args.calibration_tokens :][None])
    input_scale = jnp.asarray(
        arrays[f"model.layers.{args.layer_index}.input_layernorm.weight"]
    )
    input_norm = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)
    calibration = input_norm.apply(
        {"params": {"scale": input_scale}}, calibration_hidden
    )
    evaluation = input_norm.apply(
        {"params": {"scale": input_scale}}, evaluation_hidden
    )

    prefix = f"model.layers.{args.layer_index}.self_attn"
    key_kernel = jnp.asarray(arrays[f"{prefix}.k_proj.weight"].T)
    value_kernel = jnp.asarray(arrays[f"{prefix}.v_proj.weight"].T)
    calibration_key = (calibration @ key_kernel).reshape(
        1, args.calibration_tokens, source.num_key_value_heads, source.head_dim
    )
    calibration_key = RMSNorm(
        source.head_dim, source.rms_norm_eps, jnp.float32
    ).apply(
        {"params": {"scale": jnp.asarray(arrays[f"{prefix}.k_norm.weight"])}},
        calibration_key,
    )
    calibration_value = (calibration @ value_kernel).reshape(
        1, args.calibration_tokens, source.num_key_value_heads, source.head_dim
    )
    rotations, _ = fit_rorope_rotations(np.asarray(calibration_key))
    calibration_key = rotate_rorope_key(
        calibration_key, jnp.asarray(rotations)
    )
    key_width = (source.num_key_value_heads - 1) * source.head_dim
    value_width = source.num_key_value_heads * source.head_dim
    calibration_k_nope = calibration_key[:, :, 1:].reshape(-1, key_width)
    calibration_v_flat = calibration_value.reshape(-1, value_width)
    auto_ratio = bkv_balance_ratio(
        np.asarray(calibration_k_nope), np.asarray(calibration_v_flat)
    )

    query, key, value = _project_qkv(evaluation, arrays, source, args.layer_index)
    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None, :]
    mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
    mapping = jnp.arange(source.num_attention_heads, dtype=jnp.int32) // (
        source.num_attention_heads // source.num_key_value_heads
    )
    query, key = apply_rorope(
        query,
        key,
        positions,
        jnp.asarray(rotations),
        mapping,
        1,
        source.rope_theta,
    )
    reference = _run_layer(
        Qwen3GQAAttention(source),
        jax_attention_params(arrays, source, args.layer_index),
        evaluation,
        positions,
        mask,
    )
    out_kernel = jnp.asarray(arrays[f"{prefix}.o_proj.weight"].T)

    def mixer_output(candidate_key, candidate_value):
        attended = _attend_components(
            query, candidate_key, candidate_value, positions, mapping, mask
        )
        return attended.reshape(
            1, args.sequence_length, source.num_attention_heads * source.head_dim
        ) @ out_kernel

    tail = min(args.tail_tokens, args.sequence_length)

    def metrics(candidate):
        return {
            "all_tokens": asdict(parity_metrics(reference, candidate)),
            "tail_tokens": asdict(
                parity_metrics(reference[:, -tail:], candidate[:, -tail:])
            ),
        }

    variants = {
        "rorope_uncompressed": {
            "total_cache_elements_per_token": 2
            * source.num_key_value_heads
            * source.head_dim,
            "mixer_output": metrics(mixer_output(key, value)),
        }
    }
    evaluation_k_nope = key[:, :, 1:].reshape(-1, key_width)
    evaluation_v_flat = value.reshape(-1, value_width)
    print(
        f"source={spec.repo_id}@{spec.revision} layer={args.layer_index} "
        f"dataset={WIKITEXT_REPO}@{WIKITEXT_REVISION} "
        f"calibration_tokens={args.calibration_tokens} latent_rank={args.latent_rank} "
        f"auto_bkv_ratio={auto_ratio:.6g}"
    )
    for name, ratio in (("activation_pca", 1.0), ("bkv_balanced_pca", auto_ratio)):
        calibration_joint = jnp.concatenate(
            (calibration_k_nope / ratio, calibration_v_flat), axis=-1
        )
        basis = fit_activation_pca_jax(
            calibration_joint, args.latent_rank, seed=args.svd_seed
        )
        evaluation_joint = jnp.concatenate(
            (evaluation_k_nope / ratio, evaluation_v_flat), axis=-1
        )
        reconstructed = (evaluation_joint @ basis.T) @ basis
        reconstructed_k = reconstructed[:, :key_width] * ratio
        reconstructed_v = reconstructed[:, key_width:]
        candidate_key = key.at[:, :, 1:].set(
            reconstructed_k.reshape(
                1,
                args.sequence_length,
                source.num_key_value_heads - 1,
                source.head_dim,
            )
        )
        candidate_value = reconstructed_v.reshape(
            1, args.sequence_length, source.num_key_value_heads, source.head_dim
        )
        calibration_reconstructed = (calibration_joint @ basis.T) @ basis
        variants[name] = {
            "latent_rank": args.latent_rank,
            "bkv_balance_ratio": ratio,
            "total_cache_elements_per_token": source.head_dim + args.latent_rank,
            "cache_reduction_fraction": 0.71875,
            "calibration_joint_reconstruction_relative_l2": _relative_l2(
                calibration_joint, calibration_reconstructed
            ),
            "evaluation_key_reconstruction_relative_l2": _relative_l2(
                evaluation_k_nope, reconstructed_k
            ),
            "evaluation_value_reconstruction_relative_l2": _relative_l2(
                evaluation_v_flat, reconstructed_v
            ),
            "mixer_output": metrics(mixer_output(candidate_key, candidate_value)),
        }
        print(
            f"{name}=DONE all_l2="
            f"{variants[name]['mixer_output']['all_tokens']['relative_l2']:.6g} "
            f"tail_l2="
            f"{variants[name]['mixer_output']['tail_tokens']['relative_l2']:.6g}"
        )

    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "dataset": f"{WIKITEXT_REPO}@{WIKITEXT_REVISION}",
        "dataset_split": "wikitext-2-raw-v1/train contiguous token prefix",
        "method": "real_text_RoRoPE_then_activation_PCA_with_BKV_ablation",
        "layer_index": args.layer_index,
        "compute_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "calibration_tokens": args.calibration_tokens,
        "evaluation_tokens": args.sequence_length,
        "svd_seed": args.svd_seed,
        "latent_rank": args.latent_rank,
        "target_cache_elements_per_token": source.head_dim + args.latent_rank,
        "auto_bkv_balance_ratio": auto_ratio,
        "variants": variants,
        "notes": [
            "Calibration and evaluation use disjoint contiguous token ranges.",
            "This remains an activation-subspace diagnostic for layer 0.",
            "A deployable weight mapping and multi-layer calibration are not yet implemented.",
        ],
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"result_json={output.resolve()}")


if __name__ == "__main__":
    main()

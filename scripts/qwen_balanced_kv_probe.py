from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np

from scripts.qwen_mla_shock import _run_layer
from scripts.qwen_rorope_shock import _project_qkv
from singularity.activation_compression import (
    bkv_balance_ratio,
    fit_activation_pca,
    project_activations,
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
from singularity.rorope import apply_rorope, fit_rorope_rotations
from singularity.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _relative_l2(reference, candidate) -> float:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float32).tiny)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _attend_components(query, key, value, positions, mapping, mask):
    head_dim = value.shape[-1]
    logits = jnp.einsum(
        "bqhcd,bkcd->bhqk", query, key, preferred_element_type=jnp.float32
    ) * (head_dim**-0.5)
    causal = positions[:, None, :, None] >= positions[:, None, None, :]
    supplied = mask[:, None, None, :] if mask.ndim == 2 else mask
    logits = jnp.where(causal & supplied, logits, jnp.finfo(jnp.float32).min)
    probabilities = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum(
        "bhqk,bkhd->bqhd", probabilities, value[:, :, mapping, :]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe cache-matched activation-PCA and BKV-balanced compression."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--calibration-length", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--latent-rank", type=int, default=448)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=123)
    parser.add_argument("--evaluation-seed", type=int, default=124)
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
    joint_width = (source.num_key_value_heads - 1) * source.head_dim
    joint_width += source.num_key_value_heads * source.head_dim
    if not 0 < args.latent_rank < min(args.calibration_length, joint_width):
        raise ValueError("latent rank must fit the calibration activation matrix")
    if max(args.calibration_length, args.sequence_length) > source.max_position_embeddings:
        raise ValueError("requested length exceeds the pinned Qwen3 context")
    model_dir, shards = ensure_layer_checkpoint(
        args.cache_dir,
        index_payload["weight_map"],
        source,
        args.layer_index,
        repo_id=spec.repo_id,
        revision=spec.revision,
    )
    arrays = load_layer_arrays(QwenCheckpointReader(model_dir), source, args.layer_index)
    input_scale = jnp.asarray(
        arrays[f"model.layers.{args.layer_index}.input_layernorm.weight"]
    )
    input_norm = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)

    def normalized_hidden(seed, length):
        hidden = jnp.asarray(
            np.random.default_rng(seed).standard_normal(
                (1, length, source.hidden_size), dtype=np.float32
            )
        )
        return input_norm.apply({"params": {"scale": input_scale}}, hidden)

    calibration = normalized_hidden(args.calibration_seed, args.calibration_length)
    calibration_q, calibration_k, calibration_v = _project_qkv(
        calibration, arrays, source, args.layer_index
    )
    rotations, _ = fit_rorope_rotations(np.asarray(calibration_k))
    calibration_positions = jnp.arange(
        args.calibration_length, dtype=jnp.int32
    )[None, :]
    mapping = jnp.arange(source.num_attention_heads, dtype=jnp.int32) // (
        source.num_attention_heads // source.num_key_value_heads
    )
    calibration_q, calibration_k = apply_rorope(
        calibration_q,
        calibration_k,
        calibration_positions,
        jnp.asarray(rotations),
        mapping,
        1,
        source.rope_theta,
    )
    calibration_k_nope = np.asarray(calibration_k[:, :, 1:]).reshape(
        -1, (source.num_key_value_heads - 1) * source.head_dim
    )
    calibration_v_flat = np.asarray(calibration_v).reshape(
        -1, source.num_key_value_heads * source.head_dim
    )
    auto_ratio = bkv_balance_ratio(calibration_k_nope, calibration_v_flat)

    evaluation = normalized_hidden(args.evaluation_seed, args.sequence_length)
    query, key, value = _project_qkv(evaluation, arrays, source, args.layer_index)
    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None, :]
    mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
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
    out_kernel = jnp.asarray(
        arrays[f"model.layers.{args.layer_index}.self_attn.o_proj.weight"].T
    )

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
            "latent_rank": joint_width,
            "total_cache_elements_per_token": source.head_dim + joint_width,
            "mixer_output": metrics(mixer_output(key, value)),
        }
    }
    evaluation_k_nope = np.asarray(key[:, :, 1:]).reshape(
        -1, (source.num_key_value_heads - 1) * source.head_dim
    )
    evaluation_v_flat = np.asarray(value).reshape(
        -1, source.num_key_value_heads * source.head_dim
    )
    print(
        f"source={spec.repo_id}@{spec.revision} layer={args.layer_index} "
        f"shards={list(shards)} latent_rank={args.latent_rank} "
        f"auto_bkv_ratio={auto_ratio:.6g}"
    )
    for name, ratio in (("activation_pca", 1.0), ("bkv_balanced_pca", auto_ratio)):
        calibration_joint = np.concatenate(
            (calibration_k_nope / ratio, calibration_v_flat), axis=-1
        )
        basis = fit_activation_pca(
            calibration_joint, args.latent_rank, seed=args.svd_seed
        )
        evaluation_joint = np.concatenate(
            (evaluation_k_nope / ratio, evaluation_v_flat), axis=-1
        )
        reconstructed = project_activations(evaluation_joint, basis)
        reconstructed_k = reconstructed[:, : calibration_k_nope.shape[-1]] * ratio
        reconstructed_v = reconstructed[:, calibration_k_nope.shape[-1] :]
        candidate_key = key.at[:, :, 1:].set(
            jnp.asarray(reconstructed_k).reshape(
                1,
                args.sequence_length,
                source.num_key_value_heads - 1,
                source.head_dim,
            )
        )
        candidate_value = jnp.asarray(reconstructed_v).reshape(
            1, args.sequence_length, source.num_key_value_heads, source.head_dim
        )
        candidate = mixer_output(candidate_key, candidate_value)
        variants[name] = {
            "latent_rank": args.latent_rank,
            "bkv_balance_ratio": ratio,
            "total_cache_elements_per_token": source.head_dim + args.latent_rank,
            "cache_reduction_fraction": 1.0
            - (source.head_dim + args.latent_rank)
            / (2 * source.num_key_value_heads * source.head_dim),
            "calibration_joint_reconstruction_relative_l2": _relative_l2(
                calibration_joint, project_activations(calibration_joint, basis)
            ),
            "evaluation_key_reconstruction_relative_l2": _relative_l2(
                evaluation_k_nope, reconstructed_k
            ),
            "evaluation_value_reconstruction_relative_l2": _relative_l2(
                evaluation_v_flat, reconstructed_v
            ),
            "mixer_output": metrics(candidate),
        }
        print(
            f"{name}=DONE all_l2="
            f"{variants[name]['mixer_output']['all_tokens']['relative_l2']:.6g} "
            f"tail_l2="
            f"{variants[name]['mixer_output']['tail_tokens']['relative_l2']:.6g}"
        )

    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "method": "TransMLA_RoRoPE_then_activation_PCA_with_BKV_ablation",
        "layer_index": args.layer_index,
        "compute_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "input_kind": "independent_deterministic_gaussian_calibration_and_evaluation",
        "calibration_length": args.calibration_length,
        "sequence_length": args.sequence_length,
        "calibration_seed": args.calibration_seed,
        "evaluation_seed": args.evaluation_seed,
        "svd_seed": args.svd_seed,
        "tail_tokens": args.tail_tokens,
        "rope_cache_elements_per_token": source.head_dim,
        "latent_rank": args.latent_rank,
        "target_cache_elements_per_token": source.head_dim + args.latent_rank,
        "auto_bkv_balance_ratio": auto_ratio,
        "variants": variants,
        "notes": [
            "RoRoPE fold 1 is frozen from EXP-015/016.",
            "Activation PCA is an oracle-style compression diagnostic, not yet a deployable weight map.",
            "Qwen3 per-head Q/K normalization remains before RoRoPE and compression.",
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

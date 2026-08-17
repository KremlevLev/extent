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
from singularity.layers.common import RMSNorm
from singularity.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_attention_params,
    load_layer_arrays,
    parity_metrics,
)
from singularity.qwen3_teacher import Qwen3GQAAttention, Qwen3TeacherConfig
from singularity.qwen_source import QWEN3_14B, validate_source_metadata
from singularity.rorope import fit_rorope_rotations, rorope_attend
from singularity.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_components(value: str) -> tuple[int, ...]:
    components = tuple(sorted({int(item) for item in value.split(",")}))
    if not components or components[0] < 1:
        raise ValueError("components must be positive comma-separated integers")
    return components


def _project_qkv(normalized, arrays, source, layer_index):
    prefix = f"model.layers.{layer_index}.self_attn"
    kernel = lambda name: jnp.asarray(arrays[f"{prefix}.{name}.weight"].T)
    query = (normalized @ kernel("q_proj")).reshape(
        normalized.shape[0], normalized.shape[1], source.num_attention_heads, source.head_dim
    )
    key = (normalized @ kernel("k_proj")).reshape(
        normalized.shape[0], normalized.shape[1], source.num_key_value_heads, source.head_dim
    )
    value = (normalized @ kernel("v_proj")).reshape(
        normalized.shape[0], normalized.shape[1], source.num_key_value_heads, source.head_dim
    )
    query = RMSNorm(source.head_dim, source.rms_norm_eps, jnp.float32).apply(
        {"params": {"scale": jnp.asarray(arrays[f"{prefix}.q_norm.weight"])}}, query
    )
    key = RMSNorm(source.head_dim, source.rms_norm_eps, jnp.float32).apply(
        {"params": {"scale": jnp.asarray(arrays[f"{prefix}.k_norm.weight"])}}, key
    )
    return query, key, value


def _measure(reference, candidate, tail_tokens):
    tail = min(tail_tokens, reference.shape[1])
    return {
        "all_tokens": asdict(parity_metrics(reference, candidate)),
        "tail_tokens": asdict(parity_metrics(reference[:, -tail:], candidate[:, -tail:])),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure standard TransMLA RoRoPE positional-decoupling shock."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--calibration-length", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--rope-components", default="1,2,4,8")
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=123)
    parser.add_argument("--evaluation-seed", type=int, default=124)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)
    components = _parse_components(args.rope_components)

    spec = QWEN3_14B
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32", compute_dtype="float32", remat_policy="none"
    )
    if components[-1] > source.num_key_value_heads:
        raise ValueError("RoPE components exceed the source KV-head count")
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
    _, calibration_key, _ = _project_qkv(
        calibration, arrays, source, args.layer_index
    )
    rotations, eigenvalues = fit_rorope_rotations(np.asarray(calibration_key))
    evaluation = normalized_hidden(args.evaluation_seed, args.sequence_length)
    query, key, value = _project_qkv(evaluation, arrays, source, args.layer_index)
    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None, :]
    mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
    reference = _run_layer(
        Qwen3GQAAttention(source),
        jax_attention_params(arrays, source, args.layer_index),
        evaluation,
        positions,
        mask,
    )
    mapping = jnp.arange(source.num_attention_heads, dtype=jnp.int32) // (
        source.num_attention_heads // source.num_key_value_heads
    )
    out_kernel = jnp.asarray(
        arrays[f"model.layers.{args.layer_index}.self_attn.o_proj.weight"].T
    )
    total_energy = float(np.maximum(eigenvalues.sum(), np.finfo(np.float64).tiny))
    variants = {}
    print(
        f"source={spec.repo_id}@{spec.revision} layer={args.layer_index} "
        f"shards={list(shards)} calibration_length={args.calibration_length} "
        f"evaluation_length={args.sequence_length}"
    )
    for count in components:
        attended = rorope_attend(
            query,
            key,
            value,
            positions,
            jnp.asarray(rotations),
            mapping,
            count,
            source.rope_theta,
            mask,
        )
        candidate = attended.reshape(
            1, args.sequence_length, source.num_attention_heads * source.head_dim
        ) @ out_kernel
        metrics = _measure(reference, candidate, args.tail_tokens)
        captured = float(eigenvalues[:, :count].sum() / total_energy)
        name = f"rorope_components_{count}"
        variants[name] = {
            "rope_cache_elements_per_token": count * source.head_dim,
            "positional_energy_fraction": captured,
            "mixer_output": metrics,
        }
        print(
            f"{name}=DONE energy={captured:.6f} "
            f"all_l2={metrics['all_tokens']['relative_l2']:.6g} "
            f"tail_l2={metrics['tail_tokens']['relative_l2']:.6g}"
        )

    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "method": "standard_TransMLA_RoRoPE_without_FreqFold_or_KV_compression",
        "layer_index": args.layer_index,
        "compute_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "input_kind": "independent_deterministic_gaussian_calibration_and_evaluation",
        "calibration_length": args.calibration_length,
        "sequence_length": args.sequence_length,
        "calibration_seed": args.calibration_seed,
        "evaluation_seed": args.evaluation_seed,
        "tail_tokens": args.tail_tokens,
        "rope_components": list(components),
        "variants": variants,
        "notes": [
            "All KV components remain present; only leading PCA components retain RoPE.",
            "The all-component variant is the exact-invariance implementation control.",
            "No FreqFold, BKV-PCA, or latent KV compression is applied in this experiment.",
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

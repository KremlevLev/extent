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
from extent.calibration_data import (
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
    load_wikitext2_tokens,
)
from extent.layers.common import RMSNorm
from extent.layers.rorope_bkv import Qwen3RoRoPEBKVAttention
from extent.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_attention_params,
    load_layer_arrays,
    parity_metrics,
)
from extent.qwen3_teacher import Qwen3GQAAttention, Qwen3TeacherConfig
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.rorope_bkv_conversion import map_qwen3_to_rorope_bkv
from extent.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit and validate the deployable Qwen3 RoRoPE-BKV attention module."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/extent-calibration-cache"
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--calibration-tokens", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--latent-rank", type=int, default=448)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--svd-seed", type=int, default=0)
    parser.add_argument("--max-relative-l2", type=float, default=0.30)
    parser.add_argument("--min-cosine", type=float, default=0.95)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)
    jax.config.update("jax_default_matmul_precision", "high")

    spec = QWEN3_14B
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32", compute_dtype="float32", remat_policy="none"
    )
    total_tokens = args.calibration_tokens + args.sequence_length
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
    input_scale = jnp.asarray(
        arrays[f"model.layers.{args.layer_index}.input_layernorm.weight"]
    )
    input_norm = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32)
    calibration = input_norm.apply(
        {"params": {"scale": input_scale}},
        jnp.asarray(hidden[: args.calibration_tokens][None]),
    )
    evaluation = input_norm.apply(
        {"params": {"scale": input_scale}},
        jnp.asarray(hidden[args.calibration_tokens :][None]),
    )
    params, mapping_report = map_qwen3_to_rorope_bkv(
        arrays,
        source,
        args.layer_index,
        calibration,
        latent_rank=args.latent_rank,
        svd_seed=args.svd_seed,
    )
    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None, :]
    mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)
    reference = _run_layer(
        Qwen3GQAAttention(source),
        jax_attention_params(arrays, source, args.layer_index),
        evaluation,
        positions,
        mask,
    )
    module = Qwen3RoRoPEBKVAttention(
        source,
        latent_rank=args.latent_rank,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    candidate, cache = module.apply(
        {"params": params},
        evaluation,
        positions,
        mask,
        return_cache=True,
    )
    jax.block_until_ready((candidate, cache))
    tail = min(args.tail_tokens, args.sequence_length)
    all_metrics = parity_metrics(reference, candidate)
    tail_metrics = parity_metrics(reference[:, -tail:], candidate[:, -tail:])
    arrays_finite = bool(
        np.all(np.isfinite(np.asarray(candidate)))
        and np.all(np.isfinite(np.asarray(cache["kv_latent"])))
        and np.all(np.isfinite(np.asarray(cache["k_rope"])))
    )
    cache_shapes_pass = (
        cache["kv_latent"].shape == (1, args.sequence_length, args.latent_rank)
        and cache["k_rope"].shape == (1, args.sequence_length, source.head_dim)
    )
    quality_gate_pass = (
        all_metrics.relative_l2 <= args.max_relative_l2
        and all_metrics.cosine_similarity >= args.min_cosine
    )
    passed = arrays_finite and cache_shapes_pass and quality_gate_pass
    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "dataset": f"{WIKITEXT_REPO}@{WIKITEXT_REVISION}",
        "method": "deployable_Qwen3_RoRoPE_BKV_attention_mapping",
        "layer_index": args.layer_index,
        "compute_dtype": "float32",
        "matmul_precision": "high",
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "calibration_tokens": args.calibration_tokens,
        "evaluation_tokens": args.sequence_length,
        "mapping": asdict(mapping_report),
        "mixer_output": {
            "all_tokens": asdict(all_metrics),
            "tail_tokens": asdict(tail_metrics),
        },
        "cache_shapes": {
            "kv_latent": list(cache["kv_latent"].shape),
            "k_rope": list(cache["k_rope"].shape),
        },
        "gates": {
            "arrays_finite": arrays_finite,
            "cache_shapes_pass": cache_shapes_pass,
            "max_relative_l2": args.max_relative_l2,
            "min_cosine": args.min_cosine,
            "quality_gate_pass": quality_gate_pass,
        },
        "passed": passed,
        "notes": [
            "The cache tensors are explicit, but this is a correctness reference path.",
            "Absorbed incremental decode and optimized kernels remain future work.",
            "Quality gates are post-pilot engineering gates, not final paper thresholds.",
        ],
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"result_json={output.resolve()}")
    if not passed:
        raise SystemExit("DEPLOYABLE-MLA-FAIL")
    print("DEPLOYABLE-MLA-PASS")


if __name__ == "__main__":
    main()

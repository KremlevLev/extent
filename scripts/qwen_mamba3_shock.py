from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib import request

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze

from scripts.qwen_mla_shock import _run_layer
from extent.calibration_data import (
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
    load_wikitext2_tokens,
)
from extent.config import HybridConfig, Mamba3Config
from extent.layers.common import RMSNorm
from extent.layers.mamba3 import Mamba3MIMO
from extent.mamba3_transplant import build_qwen3_to_mamba3_transplant_variants
from extent.mla_conversion import qwen3_decoder_common_params
from extent.model import HybridDecoderLayer
from extent.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_layer_params,
    load_layer_arrays,
    parity_metrics,
)
from extent.qwen3_teacher import (
    Qwen3DecoderLayer,
    Qwen3GQAAttention,
    Qwen3TeacherConfig,
)
from extent.qwen_source import QWEN3_14B, validate_source_metadata
from extent.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _hybrid_config(source: Qwen3TeacherConfig, mamba: Mamba3Config) -> HybridConfig:
    # Layer zero is Mamba; the unused second layer only satisfies the hybrid
    # config invariant that at least one retained attention layer exists.
    return HybridConfig(
        vocab_size=source.vocab_size,
        hidden_size=source.hidden_size,
        intermediate_size=source.intermediate_size,
        num_layers=2,
        attention_layer_indices=(1,),
        rms_norm_eps=source.rms_norm_eps,
        max_position_embeddings=source.max_position_embeddings,
        param_dtype="float32",
        compute_dtype="float32",
        logits_dtype="float32",
        remat_policy="none",
        mamba=mamba,
    )


def _scale_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(candidate) / denominator)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure controlled Qwen3 attention-to-Mamba3 initialization shock."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/extent-calibration-cache"
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
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
    mamba_config = Mamba3Config()
    tokens = load_wikitext2_tokens(
        args.sequence_length,
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
    hidden = jnp.asarray(reader.read_rows("model.embed_tokens.weight", tokens)[None])
    positions = jnp.arange(args.sequence_length, dtype=jnp.int32)[None]
    mask = jnp.ones((1, args.sequence_length), dtype=jnp.bool_)

    teacher_params = jax_layer_params(arrays, source, args.layer_index)
    teacher_mixer = _run_layer(
        Qwen3GQAAttention(source),
        teacher_params["self_attn"],
        RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32).apply(
            {"params": {"scale": teacher_params["input_layernorm"]["scale"]}},
            hidden,
        ),
        positions,
        mask,
    )
    teacher_decoder = _run_layer(
        Qwen3DecoderLayer(source), teacher_params, hidden, positions, mask
    )

    mamba = Mamba3MIMO(
        source.hidden_size,
        mamba_config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    normalized_hidden = RMSNorm(
        source.hidden_size, source.rms_norm_eps, jnp.float32
    ).apply(
        {"params": {"scale": teacher_params["input_layernorm"]["scale"]}}, hidden
    )
    base_params = mamba.init(jax.random.key(args.seed), normalized_hidden)["params"]
    variants, reports = build_qwen3_to_mamba3_transplant_variants(
        base_params, arrays, source, mamba_config, args.layer_index
    )
    hybrid_config = _hybrid_config(source, mamba_config)
    hybrid_layer = HybridDecoderLayer(hybrid_config, 0)
    common = qwen3_decoder_common_params(arrays, args.layer_index)
    run_mixer = jax.jit(
        lambda params: mamba.apply({"params": params}, normalized_hidden)
    )
    run_decoder = jax.jit(
        lambda params: hybrid_layer.apply(
            {"params": {**common, "mamba": params}}, hidden, positions, mask
        )
    )

    results = {}
    passed = True
    for name, params in variants.items():
        mixer_output = run_mixer(params)
        decoder_output = run_decoder(params)
        jax.block_until_ready((mixer_output, decoder_output))
        mixer_np = np.asarray(mixer_output)
        decoder_np = np.asarray(decoder_output)
        finite = bool(np.all(np.isfinite(mixer_np)) and np.all(np.isfinite(decoder_np)))
        passed &= finite
        results[name] = {
            "mapping": asdict(reports[name]),
            "mixer_output": asdict(parity_metrics(teacher_mixer, mixer_np)),
            "decoder_output": asdict(parity_metrics(teacher_decoder, decoder_np)),
            "mixer_norm_ratio_to_teacher": _scale_ratio(teacher_mixer, mixer_np),
            "finite": finite,
        }
        print(
            f"{name}=DONE mixer_relative_l2="
            f"{results[name]['mixer_output']['relative_l2']:.6g} "
            f"mixer_cosine={results[name]['mixer_output']['cosine_similarity']:.6g}"
        )

    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "dataset": f"{WIKITEXT_REPO}@{WIKITEXT_REVISION}",
        "method": "controlled_Qwen3_to_Mamba3_MIMO_initialization_shock",
        "layer_index": args.layer_index,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "compute_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "mamba_config": asdict(mamba_config),
        "variants": results,
        "passed": passed,
        "notes": [
            "All variants share the identical random base and stability controllers.",
            "INIT-D versus INIT-E isolates copied versus distinct MIMO B/C channels.",
            "This is immediate layer shock before any recovery training.",
        ],
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"result_json={output.resolve()}")
    if not passed:
        raise SystemExit("MAMBA3-SHOCK-FAIL")
    print("MAMBA3-SHOCK-PASS")


if __name__ == "__main__":
    main()

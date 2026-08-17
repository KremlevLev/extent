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
from singularity.calibration_data import (
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
    load_wikitext2_tokens,
)
from singularity.config import Mamba3Config
from singularity.layers.common import RMSNorm
from singularity.layers.mamba3 import Mamba3MIMO
from singularity.mamba3_transplant import build_qwen3_to_mamba3_transplant_variants
from singularity.qwen3_parity import (
    ensure_layer_checkpoint,
    jax_layer_params,
    load_layer_arrays,
    parity_metrics,
)
from singularity.qwen3_teacher import Qwen3GQAAttention, Qwen3TeacherConfig
from singularity.qwen_source import QWEN3_14B, validate_source_metadata
from singularity.readout_calibration import fit_dual_ridge_readout
from singularity.weight_mapping import QwenCheckpointReader


def _read_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _select_and_fit_readout(
    features: jax.Array,
    targets: jax.Array,
    calibration_tokens: int,
    selection_tokens: int,
    ridge_values: tuple[float, ...],
) -> tuple[jax.Array, dict]:
    fit_end = calibration_tokens - selection_tokens
    selection = []
    best_ridge = None
    best_l2 = float("inf")
    for ridge in ridge_values:
        kernel, report = fit_dual_ridge_readout(
            features[:fit_end], targets[:fit_end], relative_ridge=ridge
        )
        metrics = parity_metrics(
            np.asarray(targets[fit_end:calibration_tokens]),
            np.asarray(features[fit_end:calibration_tokens] @ kernel),
        )
        selection.append(
            {
                "relative_ridge": ridge,
                "fit": asdict(report),
                "validation": asdict(metrics),
            }
        )
        if metrics.relative_l2 < best_l2:
            best_l2 = metrics.relative_l2
            best_ridge = ridge
    kernel, final_report = fit_dual_ridge_readout(
        features[:calibration_tokens],
        targets[:calibration_tokens],
        relative_ridge=float(best_ridge),
    )
    return kernel, {
        "selection": selection,
        "selected_relative_ridge": best_ridge,
        "final_fit": asdict(final_report),
    }


def _fit_residualized_context_readout(
    raw_features: jax.Array,
    context_features: jax.Array,
    targets: jax.Array,
    calibration_tokens: int,
    selection_tokens: int,
    ridge_values: tuple[float, ...],
    raw_ridge: float,
) -> tuple[jax.Array, jax.Array, dict]:
    """Fit raw-token readout plus a separately regularized context residual."""
    fit_end = calibration_tokens - selection_tokens
    raw_selection_kernel, _ = fit_dual_ridge_readout(
        raw_features[:fit_end], targets[:fit_end], relative_ridge=raw_ridge
    )
    raw_selection_prediction = raw_features @ raw_selection_kernel
    residual_selection_target = targets - raw_selection_prediction
    selection = []
    best_ridge = None
    best_l2 = float("inf")
    for ridge in ridge_values:
        residual_kernel, report = fit_dual_ridge_readout(
            context_features[:fit_end],
            residual_selection_target[:fit_end],
            relative_ridge=ridge,
        )
        combined = (
            raw_selection_prediction[fit_end:calibration_tokens]
            + context_features[fit_end:calibration_tokens] @ residual_kernel
        )
        metrics = parity_metrics(
            np.asarray(targets[fit_end:calibration_tokens]), np.asarray(combined)
        )
        selection.append(
            {
                "relative_ridge": ridge,
                "residual_fit": asdict(report),
                "combined_validation": asdict(metrics),
            }
        )
        if metrics.relative_l2 < best_l2:
            best_l2 = metrics.relative_l2
            best_ridge = ridge

    raw_kernel, raw_report = fit_dual_ridge_readout(
        raw_features[:calibration_tokens],
        targets[:calibration_tokens],
        relative_ridge=raw_ridge,
    )
    residual_target = targets - raw_features @ raw_kernel
    residual_kernel, residual_report = fit_dual_ridge_readout(
        context_features[:calibration_tokens],
        residual_target[:calibration_tokens],
        relative_ridge=float(best_ridge),
    )
    return raw_kernel, residual_kernel, {
        "raw_relative_ridge": raw_ridge,
        "context_selection": selection,
        "selected_context_relative_ridge": best_ridge,
        "final_raw_fit": asdict(raw_report),
        "final_residual_fit": asdict(residual_report),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Test held-out teacher-signal recovery from frozen Mamba-3 features."
    )
    parser.add_argument("--cache-dir", default="/kaggle/working/qwen3-layer-parity")
    parser.add_argument(
        "--dataset-cache-dir", default="/kaggle/working/singularity-calibration-cache"
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--calibration-tokens", type=int, default=256)
    parser.add_argument("--selection-tokens", type=int, default=64)
    parser.add_argument("--evaluation-tokens", type=int, default=128)
    parser.add_argument("--ridge-values", default="1e-2,1e-3,1e-4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)
    if not 0 < args.selection_tokens < args.calibration_tokens:
        raise ValueError("selection-tokens must be within the calibration prefix")
    ridge_values = tuple(float(value) for value in args.ridge_values.split(","))
    if not ridge_values or any(value <= 0 for value in ridge_values):
        raise ValueError("ridge-values must contain positive numbers")
    jax.config.update("jax_default_matmul_precision", "high")

    spec = QWEN3_14B
    config_payload = _read_json(spec.resolve_url("config.json"))
    index_payload = _read_json(spec.resolve_url("model.safetensors.index.json"))
    validate_source_metadata(config_payload, index_payload, spec)
    source = Qwen3TeacherConfig(
        param_dtype="float32", compute_dtype="float32", remat_policy="none"
    )
    total_tokens = args.calibration_tokens + args.evaluation_tokens
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
    hidden = jnp.asarray(reader.read_rows("model.embed_tokens.weight", tokens)[None])
    positions = jnp.arange(total_tokens, dtype=jnp.int32)[None]
    mask = jnp.ones((1, total_tokens), dtype=jnp.bool_)
    teacher_params = jax_layer_params(arrays, source, args.layer_index)
    normalized = RMSNorm(source.hidden_size, source.rms_norm_eps, jnp.float32).apply(
        {"params": {"scale": teacher_params["input_layernorm"]["scale"]}}, hidden
    )
    teacher = _run_layer(
        Qwen3GQAAttention(source),
        teacher_params["self_attn"],
        normalized,
        positions,
        mask,
    )[0]

    mamba_config = Mamba3Config()
    mamba = Mamba3MIMO(
        source.hidden_size,
        mamba_config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    base = mamba.init(jax.random.key(args.seed), normalized)["params"]
    variants, reports = build_qwen3_to_mamba3_transplant_variants(
        base, arrays, source, mamba_config, args.layer_index
    )
    run_features = jax.jit(
        lambda params: mamba.apply(
            {"params": params}, normalized, return_features=True
        )
    )
    evaluation = slice(args.calibration_tokens, total_tokens)
    raw_features = normalized[0]
    raw_kernel, raw_calibration = _select_and_fit_readout(
        raw_features,
        jnp.asarray(teacher),
        args.calibration_tokens,
        args.selection_tokens,
        ridge_values,
    )
    raw_prediction = np.asarray(raw_features[evaluation] @ raw_kernel)
    selected_raw_ridge = float(raw_calibration["selected_relative_ridge"])
    results = {}
    passed = True
    for name, params in variants.items():
        original, features = run_features(params)
        jax.block_until_ready((original, features))
        features = features[0]
        kernel, calibration = _select_and_fit_readout(
            features,
            jnp.asarray(teacher),
            args.calibration_tokens,
            args.selection_tokens,
            ridge_values,
        )
        calibrated = features[evaluation] @ kernel
        residual_raw_kernel, residual_kernel, residual_calibration = (
            _fit_residualized_context_readout(
                raw_features,
                features,
                jnp.asarray(teacher),
                args.calibration_tokens,
                args.selection_tokens,
                ridge_values,
                selected_raw_ridge,
            )
        )
        residualized = (
            raw_features[evaluation] @ residual_raw_kernel
            + features[evaluation] @ residual_kernel
        )
        calibrated_np = np.asarray(calibrated)
        residualized_np = np.asarray(residualized)
        finite = bool(
            np.all(np.isfinite(calibrated_np))
            and np.all(np.isfinite(residualized_np))
        )
        passed &= finite
        results[name] = {
            "mapping": asdict(reports[name]),
            "original_heldout": asdict(
                parity_metrics(teacher[evaluation], np.asarray(original[0, evaluation]))
            ),
            "calibration": calibration,
            "calibrated_heldout": asdict(
                parity_metrics(teacher[evaluation], calibrated_np)
            ),
            "residualized_context_calibration": residual_calibration,
            "raw_plus_mamba_heldout": asdict(
                parity_metrics(teacher[evaluation], residualized_np)
            ),
            "finite": finite,
        }
        print(
            f"{name}=DONE original_l2={results[name]['original_heldout']['relative_l2']:.6g} "
            f"calibrated_l2={results[name]['calibrated_heldout']['relative_l2']:.6g} "
            f"raw_plus_mamba_l2={results[name]['raw_plus_mamba_heldout']['relative_l2']:.6g}"
        )
    result = {
        "source": f"{spec.repo_id}@{spec.revision}",
        "dataset": f"{WIKITEXT_REPO}@{WIKITEXT_REVISION}",
        "method": "frozen_Mamba3_readout_and_residualized_context_probe",
        "layer_index": args.layer_index,
        "calibration_tokens": args.calibration_tokens,
        "selection_tokens": args.selection_tokens,
        "evaluation_tokens": args.evaluation_tokens,
        "ridge_values": ridge_values,
        "seed": args.seed,
        "compute_dtype": "float32",
        "jax_backend": jax.default_backend(),
        "variants": results,
        "raw_normalized_hidden_control": {
            "calibration": raw_calibration,
            "heldout": asdict(parity_metrics(teacher[evaluation], raw_prediction)),
        },
        "passed": passed and bool(np.all(np.isfinite(raw_prediction))),
        "notes": [
            "Ridge is selected on a suffix inside calibration, never on held-out evaluation.",
            "The selected ridge is refit on the complete calibration prefix.",
            "Only the linear readout is fitted; every Mamba recurrent parameter is frozen.",
            "Raw and residual readouts are nested-selected without evaluation-token access.",
        ],
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        output = Path(args.result_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"result_json={output.resolve()}")
    if not result["passed"]:
        raise SystemExit("MAMBA3-READOUT-PROBE-FAIL")
    print("MAMBA3-READOUT-PROBE-PASS")


if __name__ == "__main__":
    main()
